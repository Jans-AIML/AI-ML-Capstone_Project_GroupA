"""
convert_coco_to_yolo.py
========================
Converts COCO 2017 JSON annotations to YOLO format .txt files.

COCO bbox format: [x_min, y_min, width, height] (absolute pixels)
YOLO bbox format: [x_center, y_center, width, height] (normalized 0-1)

"""

import json
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION - Update this path
# ============================================================
DIR = "./images"

# ============================================================
# COCO category ID to YOLO class ID mapping
# COCO uses non-contiguous IDs (1-90 with gaps)
# YOLO needs contiguous IDs (0-79)
# ============================================================
COCO_TO_YOLO_CLASS = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


def convert_coco_to_yolo(annotation_file: str, output_dir: str):
    """
    Convert a single COCO annotation JSON file to YOLO format .txt files.

    Args:
        annotation_file: Path to COCO instances JSON (e.g., instances_train2017.json)
        output_dir: Directory to save YOLO .txt label files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    # Build image ID -> image info lookup
    images = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image ID
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    print(f"Converting {len(images)} images...")
    skipped = 0
    converted = 0

    for image_id, img_info in tqdm(images.items()):
        img_width = img_info["width"]
        img_height = img_info["height"]
        file_name = Path(img_info["file_name"]).stem  # e.g., "000000000001"

        label_file = output_dir / f"{file_name}.txt"
        lines = []

        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                # Skip crowd annotations
                if ann.get("iscrowd", 0):
                    continue

                cat_id = ann["category_id"]
                if cat_id not in COCO_TO_YOLO_CLASS:
                    skipped += 1
                    continue

                yolo_class = COCO_TO_YOLO_CLASS[cat_id]

                # COCO bbox: [x_min, y_min, width, height] (absolute)
                x_min, y_min, bbox_w, bbox_h = ann["bbox"]

                # Convert to YOLO: [x_center, y_center, width, height] (normalized)
                x_center = (x_min + bbox_w / 2.0) / img_width
                y_center = (y_min + bbox_h / 2.0) / img_height
                norm_w = bbox_w / img_width
                norm_h = bbox_h / img_height

                # Clamp values to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                # Skip invalid boxes
                if norm_w <= 0 or norm_h <= 0:
                    skipped += 1
                    continue

                lines.append(
                    f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                converted += 1

        # Write label file (empty file if no annotations - YOLO treats as negative sample)
        with open(label_file, "w") as f:
            f.write("\n".join(lines))

    print(f"Done! Converted {converted} annotations, skipped {skipped}")
    print(f"Label files saved to: {output_dir}")


def main():
    coco_dir = Path(DIR)
    ann_dir = coco_dir / "annotations"

    # Convert train annotations
    train_ann = ann_dir / "instances_train2017.json"
    train_labels = coco_dir / "labels" / "train2017"

    if train_ann.exists():
        print("=" * 60)
        print("Converting TRAIN annotations")
        print("=" * 60)
        convert_coco_to_yolo(str(train_ann), str(train_labels))
    else:
        print(f"WARNING: Train annotations not found at {train_ann}")

    # Convert val annotations
    val_ann = ann_dir / "instances_val2017.json"
    val_labels = coco_dir / "labels" / "val2017"

    if val_ann.exists():
        print("\n" + "=" * 60)
        print("Converting VAL annotations")
        print("=" * 60)
        convert_coco_to_yolo(str(val_ann), str(val_labels))
    else:
        print(f"WARNING: Val annotations not found at {val_ann}")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nExpected directory structure:")
    print(f"  {coco_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train2017/   (118K images)")
    print(f"  │   └── val2017/     (5K images)")
    print(f"  ├── labels/")
    print(f"  │   ├── train2017/   (118K .txt files)")
    print(f"  │   └── val2017/     (5K .txt files)")
    print(f"  └── annotations/")
    print(f"      ├── instances_train2017.json")
    print(f"      └── instances_val2017.json")


if __name__ == "__main__":
    main()
