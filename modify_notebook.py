import json

with open('/home/jyo/Desktop/Projects/Tiny_Stories_Diffusion/notebooks/tinystories-diffusion_medium_dual.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Add GPU info
        if "import torch\n" in source and "import torch.nn as nn\n" in source:
            source += "\nprint(\"GPU count:\", torch.cuda.device_count())\nfor i in range(torch.cuda.device_count()):\n    print(i, torch.cuda.get_device_name(i))\n"
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

        # 2. Use DataParallel
        if "model = Model().to(device)" in source and "optimizer = torch.optim.AdamW" in source:
            # Replace lines
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("model = Model().to(device)"):
                    lines.insert(i+1, "if torch.cuda.device_count() > 1:")
                    lines.insert(i+2, "    print(f\"Using {torch.cuda.device_count()} GPUs!\")")
                    lines.insert(i+3, "    model = torch.nn.DataParallel(model)")
                    break
            source = '\n'.join(lines)
            # Fix split lines without trailing newlines mapping correctly by using standard splitlines and re-adding newlines correctly
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]] if source else []

        # 3 and 4. Mixed precision & gradient accumulation
        if "# Train the model" in source and "start = time.time()" in source:
            new_source = """# Train the model
start = time.time()
scaler = torch.cuda.amp.GradScaler()
grad_accum_steps = 4

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {time.time()-start:.2f}s")
        wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "iter": iter})
        
    if iter < 100:
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        xb, yb, mb, mask_probs = get_batch("train")
        with torch.cuda.amp.autocast():
            logits, loss = model(xb, yb, mb, mask_rate=mask_probs)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        loss_accum += loss.item() * grad_accum_steps

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    if iter >= 100:
        scheduler.step()
    
    if iter % 10 == 0:
        wandb.log({"step_loss": loss_accum})
        
print("Training Complete!")
wandb.finish()

# Save Model Weights
print("Saving model weights...")
if isinstance(model, torch.nn.DataParallel):
    torch.save(model.module.state_dict(), "tinystories_diffusion.pt")
else:
    torch.save(model.state_dict(), "tinystories_diffusion.pt")
print("Saved to tinystories_diffusion.pt")"""
            # Also clean up new_source logic to format properly
            source = new_source
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]] if source else []

with open('/home/jyo/Desktop/Projects/Tiny_Stories_Diffusion/notebooks/tinystories-diffusion_medium_dual.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
