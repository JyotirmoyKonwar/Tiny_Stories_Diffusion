import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import argparse
import os

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab + 1 # +1 for mask token
mask_token_id = enc.n_vocab

def encode(s):
    return enc.encode(s)

def decode(l):
    return enc.decode([t for t in l if t != mask_token_id])

def norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    return out.to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.config.n_head, self.config.head_dim)
        k = self.c_k(x).view(B, T, self.config.n_head, self.config.head_dim)
        v = self.c_v(x).view(B, T, self.config.n_head, self.config.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = int(8 * config.n_embd / 3) 
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(vocab_size, config.n_embd)
        self.time_emb = nn.Sequential(
            nn.Linear(1, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )
        self.rotary_seq_len = config.block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # tie weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / self.config.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, mask=None, mask_rate=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        if mask_rate is not None:
            t = mask_rate.float().unsqueeze(-1)  # (B, 1, 1)
            x = x + self.time_emb(t)
        x = norm(x)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.lm_head(x)
        return logits, None

class Config:
    def __init__(self, model_type):
        self.block_size = 512
        if model_type == 'medium':
            self.n_embd = 512
            self.n_head = 8
            self.n_layer = 8
            self.weights_path = "../tinystories_diffusion_med_dual.pt"
        elif model_type == 'gpt2':
            self.n_embd = 768
            self.n_head = 12
            self.n_layer = 12
            self.weights_path = "../tinystories_diffusion_GPT2_dual.pt"
        else:
            raise ValueError("model_type must be 'medium' or 'gpt2'")
        self.head_dim = self.n_embd // self.n_head

@torch.no_grad()
def generate(model, config, prompt_tokens, max_new_tokens, temp=1.0, confidence_threshold=0.95, top_k=3, device="cpu"):
    model.eval()
    prompt_len = len(prompt_tokens)
    all_tokens = prompt_tokens.copy()
    total_steps = 0

    while len(all_tokens) - prompt_len < max_new_tokens:
        block_len = min(config.block_size - prompt_len, prompt_len + max_new_tokens - len(all_tokens))
        if block_len <= 0: break
        
        x = torch.full((1, config.block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)
        
        masked = torch.zeros(1, config.block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len:prompt_len + block_len] = True

        while masked.any():
            total_steps += 1
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(masked, confidences, torch.tensor(-float('inf')).to(device))
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(1, config.block_size)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask
            
        all_tokens.extend(x[0, prompt_len:prompt_len + block_len].tolist())
        
    return decode(all_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for TinyStories Diffusion Models")
    parser.add_argument("--model", type=str, default="medium", choices=["medium", "gpt2"], help="Model type to use")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights. If not provided, uses default path for the model type.")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl who", help="Prompt to start the story")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    
    args = parser.parse_args()
    
    config = Config(args.model)
    weights_path = args.model_path if args.model_path else config.weights_path

    print(f"Loading {args.model} model onto {args.device}...")
    model = Model(config)
    
    # Load state dict
    try:
        state_dict = torch.load(weights_path, map_location=args.device, weights_only=True)
        
        # Handle 'module.' prefix from DataParallel if present
        unwrapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                unwrapped_state_dict[k[7:]] = v
            else:
                unwrapped_state_dict[k] = v
                
        model.load_state_dict(unwrapped_state_dict)
        print(f"Model loaded successfully from {weights_path}!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
        
    model.to(args.device)
    
    print(f"\\nPrompt: {args.prompt}")
    prompt_tokens = encode(args.prompt)
    
    print("Generating story...")
    output = generate(model, config, prompt_tokens, max_new_tokens=args.max_new_tokens, device=args.device)
    
    print(f"\\nFinal Output:\\n{output}")