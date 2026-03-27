# TinyStories Diffusion Language Model

This repository contains a PyTorch implementation of a Diffusion Language Model trained on the TinyStories dataset. It is only about 10M parameters big.

**Check out the live demo on Hugging Face Spaces:** [Tiny DiffLM Story Teller](https://huggingface.co/spaces/Jyo-K/Tiny_DiffLM_Story_Teller)

> The outputs on HF Spaces isn't great because of the limited compute resources(using only CPU inferencing). You can try it out on Colab or locally on a GPU for better results. 

**The model weights are availble [HERE](https://huggingface.co/spaces/Jyo-K/Tiny_DiffLM_Story_Teller/resolve/main/tinystories_diffusion.pt?download=true)**

## About

- **Diffusion-style Token Generation:**  Iterative decoding where the model predicts tokens over multiple steps instead of traditional autoregressive (left-to-right) generation.
- **SwiGLU Activation:** ⚡ Employs the SwiGLU variant in the MLP blocks (`F.silu(self.w1(x)) * self.w2(x)`), maintaining an effective 8/3 expansion ratio.
- **Rotary Position Embeddings (RoPE):**  Multi-Head Attention leverages rotary embeddings for relative positional encoding.

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
   pip install -r requirements.txt
   ```

2. **Dataset Setup:**  
   Ensure your custom dataset `tinystories_46k.jsonl` is present in the root of the project directory alongside the notebook.

   Or

   You can download using the script 
   ```bash
   python Tinystories_data_download.py
   ``` 
   where it will download 46k stories from the 3M+ stories available in the dataset, you can manaully increase of decrese the number of stories by changing the `limit` variable in the script.

3. **Training:**  
   Run all subsequent cells in `tinystories_diffusion.ipynb` or  use the script
   ```bash
   python Tinystories_diffusion.py
   ```   
   The training will take around 1hr on a T4 GPU(that's what it took me for a dataset of 46k datapoints and 5k iterations with a batch size of 16(VRAM constraints))

4. **Inferencing:**  
   If you only want to perform inferencing, you can use the script
   ```bash
   mkdir -p model && wget -P model https://huggingface.co/spaces/Jyo-K/Tiny_DiffLM_Story_Teller/resolve/main/tinystories_diffusion.pt?download=true
   python inference.py "Once upon a time, there was a little girl who" --tokens 150
   ```

## Model Architecture Details

*   **Tokenizer:** TikToken (`gpt2` BPE encoding) with an extended vocabulary space adding a `[MASK]` token.
*   **Context Window:** 256 tokens (`block_size`)
*   **Dimensions:** $384$ embedding size, $6$ attention heads, and $6$ transformer layers.
*   **Loss Calculation:** Computed dynamically using Mean Cross-Entropy over randomly injected masked tokens.

## Parallel Decoding: GPT (Autoregressive) vs Diffusion LM

Below is a visual representation comparing standard autoregressive generation (GPT-style) against the parallel decoding method used in this diffusion language model.

![Autoregressive vs Diffusion Decoding Comparison](others/animation.gif)

Instead of strictly decoding the next token left-to-right, the diffusion generation algorithm progressively evaluates a fully masked block window and unmasks the tokens that yield a confidence score over a specified `confidence_threshold` iteratively until all variables within the context block are satisfied.
