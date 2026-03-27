import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
import os
import argparse

# Hyperparameters (Must match trained model)
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tokenizer setup
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab + 1 # +1 for mask token
mask_token_id = enc.n_vocab

def encode(s):
    return enc.encode(s)

def decode(l):
    return enc.decode([t for t in l if t != mask_token_id])

# Architecture Components (Kaggle/CPU compatible)
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        # Manual RMSNorm
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Manual attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3) 
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        x = x + self.attn(norm_x, cos_sin)
        
        norm_x2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        x = x + self.mlp(norm_x2)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.rotary_seq_len = block_size * 2
        
        device_for_emb = torch.device(device)
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device_for_emb)
        inv_freq = 1.0 / (10000 ** (channel_range / head_dim))
        t = torch.arange(self.rotary_seq_len, dtype=torch.float32, device=device_for_emb)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
            
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        return self.lm_head(x), None

def generate_diffusion(model, prompt_tokens, max_new_tokens=100, temp=1.0):
    model.eval()
    prompt_len = len(prompt_tokens)
    all_tokens = prompt_tokens.copy()
    confidence_threshold = 0.95
    top_k = 3

    while len(all_tokens) - len(prompt_tokens) < max_new_tokens:
        block_len = min(block_size - prompt_len, len(prompt_tokens) + max_new_tokens - len(all_tokens))
        if block_len <= 0: break
        
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)
        
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len:prompt_len + block_len] = True

        while masked.any():
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(masked, confidences, torch.tensor(-float('inf')).to(device))
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(1, block_size)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask
            
        all_tokens.extend(x[0, prompt_len:prompt_len + block_len].tolist())
        prompt_len = len(all_tokens) # Update prompt_len for sliding window but NOT the terminal condition check
        
    return decode(all_tokens)

def main():
    parser = argparse.ArgumentParser(description="TinyStories Diffusion LM Inference CLI")
    parser.add_argument("prompt", type=str, help="First few words of the story as input")
    parser.add_argument("--tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--weights", type=str, default="model/tinystories_diffusion.pt", help="Path to the model weights file")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling")

    args = parser.parse_args()

    # Load Model
    model = Model().to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print(f"Error: Weights file '{args.weights}' not found.")
        return

    # Encode and Generate
    prompt_tokens = encode(args.prompt)
    output = generate_diffusion(model, prompt_tokens, max_new_tokens=args.tokens, temp=args.temp)
    
    print("\n--- Generated Story ---")
    print(output)
    print("-----------------------\n")

if __name__ == "__main__":
    main()
