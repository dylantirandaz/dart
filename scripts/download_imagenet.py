"""Download ImageNet 256x256 from HuggingFace and convert to ImageFolder format.

Streams the dataset to avoid double-caching (parquet + arrow).
Saves directly to C:/imagenet256/train/class_XXXX/img_XXXXXXX.jpg

Usage:
    pip install datasets pillow
    python scripts/download_imagenet.py
"""

import os
import sys

OUTPUT_DIR = "C:/imagenet256/train"


def main():
    print("Downloading ImageNet 256x256 from HuggingFace (streaming)...")
    print("  Source: evanarlian/imagenet_1k_resized_256")
    print(f"  Output: {OUTPUT_DIR}\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install required: pip install datasets")
        sys.exit(1)

    # Stream to avoid caching the full dataset in Arrow format
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train", streaming=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for c in range(1000):
        os.makedirs(os.path.join(OUTPUT_DIR, f"class_{c:04d}"), exist_ok=True)

    count = 0
    for example in ds:
        label = example["label"]
        image = example["image"]
        path = os.path.join(OUTPUT_DIR, f"class_{label:04d}", f"img_{count:07d}.jpg")
        if not os.path.exists(path):
            image.save(path, quality=95)
        count += 1
        if count % 10000 == 0:
            print(f"  {count} images saved...")

    print(f"\nDone. {count} images saved to {OUTPUT_DIR}")
    print(f"\nTo train:")
    print(f"  python train/train.py --dataset folder --data-dir {OUTPUT_DIR} "
          f"--size small --num-steps 4 --steps 200000 --batch-size 8 --workers 0 "
          f"--output-dir C:/dart_checkpoints --save-every 10000 --sample-every 10000")


if __name__ == "__main__":
    main()
