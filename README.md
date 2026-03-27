# TinyStories Diffusion Language Model

This repository contains a PyTorch implementation of a diffusion-based language model trained on the TinyStories dataset. It uses a parallel decoding methodology via token masking and confidence-based sampling, similar to non-autoregressive language models.

## Features

- **Diffusion-style Token Generation:** Iterative decoding where the model predicts tokens over multiple steps instead of traditional autoregressive (left-to-right) generation.
- **SwiGLU Activation:** Employs the SwiGLU variant in the MLP blocks (`F.silu(self.w1(x)) * self.w2(x)`), maintaining an effective 8/3 expansion ratio.
- **Rotary Position Embeddings (RoPE):** Multi-Head Attention leverages rotary embeddings for relative positional encoding.
- **Custom Dataset Loading:** Trains on a local JSONL subset of TinyStories (`tinystories_46k.jsonl`).
- **Weights & Biases Integration:** Fully instrumented with `wandb` for logging training losses, validation metrics, and performance.
- **Built for Jupyter:** Provided as an easy-to-run Jupyter Notebook (`tinystories_diffusion.ipynb`).

## Requirements

The project dependencies are outlined within the notebook itself, but you will minimally need:

- Python 3.8+
- PyTorch
- Hugging Face `datasets`
- `tiktoken`
- `wandb`

## Usage

1. **Install dependencies:**  
   Open the notebook `tinystories_diffusion.ipynb`. The first cell contains the install commands necessary:
   ```bash
   pip install datasets tiktoken torch wandb
   ```

2. **Dataset Setup:**  
   Ensure your custom dataset `tinystories_46k.jsonl` is present in the root of the project directory alongside the notebook.

3. **WandB Setup:**  
   The notebook will prompt you to authenticate with Weights & Biases to track the training progress. Run `wandb login` with your unique API key.

4. **Training:**  
   Run all subsequent cells in `tinystories_diffusion.ipynb`. The notebook will initialize the model, begin the training loop, print updates, and log them to W&B. The model weights are automatically saved to `tinystories_diffusion.pt` at the end of training.

5. **Interactive Inference:**  
   Once the model is trained, open `inference.ipynb`. This dedicated notebook loads your saved model weights natively and provides an interactive prompt loop. You can input your custom prompt (e.g., 10 words):
   > "Once upon a time, there was a little girl who"  
   and see the generated tokens materialize through the masked-diffusion parallel decoding pipeline.

## Model Architecture Details

*   **Tokenizer:** TikToken (`gpt2` BPE encoding) with an extended vocabulary space adding a `[MASK]` token.
*   **Context Window:** 256 tokens (`block_size`)
*   **Dimensions:** $384$ embedding size, $6$ attention heads, and $6$ transformer layers.
*   **Loss Calculation:** Computed dynamically using Mean Cross-Entropy over randomly injected masked tokens.

## Parallel Decoding: GPT (Autoregressive) vs Diffusion LM

Below is a visual representation comparing standard autoregressive generation (GPT-style) against the parallel decoding method used in this diffusion language model.

![Autoregressive vs Diffusion Decoding Comparison](others/animation.gif)

Instead of strictly decoding the next token left-to-right, the diffusion generation algorithm progressively evaluates a fully masked block window and unmasks the tokens that yield a confidence score over a specified `confidence_threshold` iteratively until all variables within the context block are satisfied.
