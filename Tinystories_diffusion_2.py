import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import tiktoken
import time
import os

# hyperparameters
batch_size = 16
block_size = 256
max_iters = 5000
eval_interval = 200
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head
torch.manual_seed(1337)

# Load Custom TinyStories dataset (jsonl)
print("Loading custom TinyStories dataset...")
dataset = load_dataset("json", data_files="/kaggle/input/datasets/jyotirmoykonwar/tinystories-46k/tinystories_46k.jsonl", split="train")
text_key = "story" if "story" in dataset.features else "text"
text = "\n".join(dataset[text_key])

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab + 1 # +1 for mask token
mask_token_id = enc.n_vocab

def encode(s):
    return enc.encode(s)

def decode(l):
    return enc.decode([t for t in l if t != mask_token_id])

print("Tokenizing data...")
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train data tokens: {len(train_data)}, Val data tokens: {len(val_data)}")

def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = x.clone()
    
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = mask_token_id
    
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask

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
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        import math
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

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
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
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
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        x = norm(x)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            if mask is not None:
                mask_flat = mask.view(B * T)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss
    
@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens, temp=1.0, confidence_threshold=0.95, top_k=3):
    model.eval()
    prompt_len = len(prompt_tokens)
    all_tokens = prompt_tokens.copy()
    total_steps = 0

    while len(all_tokens) - prompt_len < max_new_tokens:
        block_len = min(block_size - prompt_len, prompt_len + max_new_tokens - len(all_tokens))
        
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)
        
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
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
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(1, block_size)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask
            
        all_tokens.extend(x[0, prompt_len:prompt_len + block_len].tolist())
        
    model.train()
    return decode(all_tokens)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            _, loss = model(X, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize model and optimizer
model = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Train the model
start = time.time()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {time.time()-start:.2f}s")
        
    xb, yb, mb = get_batch("train")
    logits, loss = model(xb, yb, mb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

        
print("Training Complete!")

# Save Model Weights
print("Saving model weights...")
torch.save(model.state_dict(), "tinystories_diffusion.pt")
print("Saved to tinystories_diffusion.pt")