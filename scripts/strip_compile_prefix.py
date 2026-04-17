"""Strip the `_orig_mod.` prefix that torch.compile adds to state dict keys."""

import argparse
from safetensors.torch import load_file, save_file

PREFIX = "_orig_mod."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to safetensors file with _orig_mod. prefix")
    parser.add_argument("output", help="Path to write cleaned safetensors")
    args = parser.parse_args()

    tensors = load_file(args.input)
    cleaned = {}
    stripped = 0
    for k, v in tensors.items():
        if k.startswith(PREFIX):
            cleaned[k[len(PREFIX):]] = v
            stripped += 1
        else:
            cleaned[k] = v

    save_file(cleaned, args.output)
    print(f"Stripped {stripped}/{len(tensors)} keys. Wrote {args.output}")


if __name__ == "__main__":
    main()
