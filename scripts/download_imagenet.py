"""Download ImageNet 256x256 from HuggingFace and convert to ImageFolder format.

Downloads evanarlian/imagenet_1k_resized_256 (25GB), then extracts to:
    C:/imagenet256/train/class_XXXX/img_XXXXXX.jpg

Usage:
    pip install datasets pillow
    python scripts/download_imagenet.py
"""

import os
import sys

OUTPUT_DIR = "C:/imagenet256/train"


def main():
    print("Downloading ImageNet 256x256 from HuggingFace...")
    print("  Source: evanarlian/imagenet_1k_resized_256")
    print("  This is ~25 GB, may take a while.\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install required: pip install datasets")
        sys.exit(1)

    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
    total = len(ds)
    print(f"  Downloaded {total} images.\n")

    print(f"Extracting to ImageFolder format: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create all class directories
    for c in range(1000):
        os.makedirs(os.path.join(OUTPUT_DIR, f"class_{c:04d}"), exist_ok=True)

    # Extract images
    for idx, example in enumerate(ds):
        label = example["label"]
        image = example["image"]
        path = os.path.join(OUTPUT_DIR, f"class_{label:04d}", f"img_{idx:07d}.jpg")
        if not os.path.exists(path):
            image.save(path, quality=95)
        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1}/{total} images ({100 * (idx + 1) / total:.1f}%)")

    print(f"\nDone. {total} images saved to {OUTPUT_DIR}")
    print(f"\nTo train: python train/train.py --dataset folder --data-dir {OUTPUT_DIR} "
          f"--size small --num-steps 4 --steps 200000 --batch-size 8 --workers 0 "
          f"--output-dir C:/dart_checkpoints --save-every 10000 --sample-every 10000")


if __name__ == "__main__":
    main()
