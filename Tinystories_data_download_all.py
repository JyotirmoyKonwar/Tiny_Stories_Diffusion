import json
import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_tinystories(token=None, output_file="data/tinystories_full.jsonl"):

    hf_token = token or os.getenv("HF_TOKEN")
    if hf_token:
        print(f"Logging in with token...")
        login(token=hf_token)
    else:
        print("Warning: No token provided. You may hit rate limits.")

    print(f"Downloading from karpathy/tinystories-gpt4-clean...")

    dataset = load_dataset("karpathy/tinystories-gpt4-clean", split="train")

    print(f"Writing to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            line = json.dumps({"text": entry["text"].strip()})
            f.write(line + '\n')

    print(f"Successfully saved {len(dataset)} datapoints to {output_file}")

MY_TOKEN = None

download_tinystories(token=MY_TOKEN)

