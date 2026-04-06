import json

path = '/home/jyo/Desktop/Projects/Tiny_Stories_Diffusion/notebooks/tinystories-diffusion_medium.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])

    if "def get_batch(" in source:
        source = source.replace("mask_probs = torch.rand(batch_size, 1)", "mask_probs = torch.distributions.Beta(1, 3).sample((batch_size, 1))")
        source = source.replace("return x, y, mask", "mask_probs = mask_probs.to(device)\n    return x, y, mask, mask_probs")

    if "def estimate_loss(model):" in source:
        source = source.replace("X, Y, M = get_batch(split)", "X, Y, M, M_P = get_batch(split)")
        source = source.replace("_, loss = model(X, Y, M)", "_, loss = model(X, Y, M, mask_rate=M_P)")

    if "class MultiHeadAttention" in source:
        # replace attention
        att_old = "        import math\n        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))\n        att = F.softmax(att, dim=-1)\n        y = att @ v"
        att_new = "        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)"
        source = source.replace(att_old, att_new)

    if "class Model(" in source:
        source = source.replace("self.token_emb = nn.Embedding(vocab_size, n_embd)", "self.token_emb = nn.Embedding(vocab_size, n_embd)\n        self.time_emb = nn.Sequential(\n            nn.Linear(1, n_embd),\n            nn.SiLU(),\n            nn.Linear(n_embd, n_embd),\n        )")
        source = source.replace("self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)", "self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)\n        self.lm_head.weight = self.token_emb.weight  # tie weights")
        source = source.replace("def forward(self, idx, targets=None, mask=None):", "def forward(self, idx, targets=None, mask=None, mask_rate=None):")
        source = source.replace("x = self.token_emb(idx)\n        x = norm(x)", "x = self.token_emb(idx)\n        if mask_rate is not None:\n            t = mask_rate.float().unsqueeze(-1)  # (B, 1, 1)\n            x = x + self.time_emb(t)\n        x = norm(x)")

    if "optimizer = " in source and "Model" in source:
        opt_new = "from torch.optim.lr_scheduler import CosineAnnealingLR\n\noptimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))\nscheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=3e-5)\n\ndef get_lr(it, warmup=100):\n    if it < warmup:\n        return learning_rate * it / warmup\n    return learning_rate\n\nprint(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')"
        source = source.replace("optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)\nprint(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')", opt_new)

    if "xb, yb, mb = get_batch(\"train\")" in source:
        source = source.replace("xb, yb, mb = get_batch(\"train\")\n    logits, loss = model(xb, yb, mb)", "if iter < 100:\n        lr = get_lr(iter)\n        for param_group in optimizer.param_groups:\n            param_group['lr'] = lr\n\n    xb, yb, mb, mask_probs = get_batch(\"train\")\n    logits, loss = model(xb, yb, mb, mask_rate=mask_probs)")
        
        # backward and step
        source = source.replace("loss.backward()\n    optimizer.step()", "loss.backward()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n    optimizer.step()\n    if iter >= 100:\n        scheduler.step()")

    cell['source'] = source.splitlines(True)

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
