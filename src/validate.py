"""
Data validation script for the Chest X-Ray Pneumonia dataset.

This is the "DataOps" step: before we train anything, we confirm the
data is actually usable. It checks three things:
  1. Every image file can actually be opened (catches corrupt files)
  2. The class balance per split (train/test) — important because we
     saw NORMAL vs PNEUMONIA is imbalanced (~1:3 in training)
  3. Basic image properties (format, size range) so we know what
     preprocessing we'll need later (e.g., resizing)

Run this BEFORE writing any training code. If this script reports
problems, fix the data first — garbage in, garbage out.
"""

import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

# Path to the extracted dataset. Adjust if your folder structure differs.
DATA_ROOT = Path("data/chest_xray")
SPLITS = ["train", "test"]  # we're skipping "val" — only 16 images, too small
CLASSES = ["NORMAL", "PNEUMONIA"]


def validate_split(split: str):
    """Check one split (train or test) for corrupt files and class balance."""
    print(f"\n{'='*50}")
    print(f"Validating split: {split}")
    print(f"{'='*50}")

    counts = defaultdict(int)
    corrupt_files = []
    sizes = []
    formats = defaultdict(int)

    for cls in CLASSES:
        class_dir = DATA_ROOT / split / cls
        if not class_dir.exists():
            print(f"  WARNING: expected folder not found: {class_dir}")
            continue

        image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # checks the file isn't corrupt/truncated
                # re-open after verify() — verify() invalidates the file handle
                with Image.open(img_path) as img:
                    sizes.append(img.size)
                    formats[img.format] += 1
                counts[cls] += 1
            except Exception as e:
                corrupt_files.append((str(img_path), str(e)))

    # --- Report results ---
    total = sum(counts.values())
    print(f"\nClass counts:")
    for cls in CLASSES:
        pct = (counts[cls] / total * 100) if total else 0
        print(f"  {cls}: {counts[cls]} images ({pct:.1f}%)")

    if total > 0:
        imbalance_ratio = max(counts.values()) / max(min(counts.values()), 1)
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2:
            print("  -> Significant imbalance detected. Remember: track "
                  "precision/recall/F1, not just accuracy, when evaluating.")

    print(f"\nCorrupt/unreadable files: {len(corrupt_files)}")
    for path, err in corrupt_files[:5]:  # show at most 5
        print(f"  {path}: {err}")

    print(f"\nImage formats found: {dict(formats)}")
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"Image size range: width [{min(widths)}-{max(widths)}], "
              f"height [{min(heights)}-{max(heights)}]")
        print("  -> Sizes vary, so we'll need to resize all images to a "
              "consistent size (e.g., 224x224) before training.")

    return counts, corrupt_files


def main():
    if not DATA_ROOT.exists():
        print(f"ERROR: {DATA_ROOT} not found.")
        print("Make sure you extracted the Kaggle dataset into "
              "'data/chest_xray/' relative to where you run this script.")
        return

    all_corrupt = []
    for split in SPLITS:
        _, corrupt = validate_split(split)
        all_corrupt.extend(corrupt)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    if all_corrupt:
        print(f"Found {len(all_corrupt)} total corrupt files across all splits. "
              "Investigate before training.")
    else:
        print("No corrupt files found. Data looks structurally sound.")


if __name__ == "__main__":
    main()