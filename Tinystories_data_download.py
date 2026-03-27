import json
import os
from datasets import load_dataset
from huggingface_hub import login

def download_tinystories_subset(token=None, output_file="tinystories_46k.jsonl", limit=46523):

    if token:
        login(token=token)
    elif "HF_TOKEN" in os.environ:
        print("Using token from environment variables...")
    else:
        print("Warning: No token provided. You may hit rate limits.")

    print(f"Downloading {limit} stories from karpathy/tinystories-gpt4-clean...")

    dataset = load_dataset("karpathy/tinystories-gpt4-clean", split=f"train[:{limit}]")

    print(f"Writing to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            line = json.dumps({"text": entry["text"].strip()})
            f.write(line + '\n')

    print(f"Successfully saved {len(dataset)} datapoints to {output_file}")

MY_TOKEN = None

download_tinystories_subset(token=MY_TOKEN)

