"""Generate a deterministic absolute image-path list for xslim calibration.

Usage:
    python generate_data_txt.py [image_dir] [output_file]

Defaults:
    image_dir  = VOC/quantify_dataset
    output_file = VOC/quantify_dataset/data.txt
"""

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an absolute image path list for xslim calibration."
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        default="VOC/quantify_dataset",
        help="Directory scanned recursively for calibration images.",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="VOC/quantify_dataset/data.txt",
        help="Output text file; one absolute image path per line.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    images = sorted(
        path.resolve()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No calibration images found under: {image_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(str(path) for path in images) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(images)} image paths to {output_file}")


if __name__ == "__main__":
    main()
