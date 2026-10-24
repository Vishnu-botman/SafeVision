"""
SafeVision: Automated Pascal VOC (SHWD - VOC2028) to YOLO Format Converter
Converts 7,581 images and Pascal VOC XML annotations into YOLOv8 format
with class harmonization:
  - 'hat'    -> class 2 ('helmet')
  - 'person' -> class 0 ('person')

Can also create a combined dataset merging CHV (1,330 images) + SHWD (7,581 images)
for an 8,911 image master PPE dataset.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


LABEL_MAP = {
    "hat": 2,       # helmet
    "person": 0     # person / head
}


def convert_xml_to_yolo(xml_file: Path, width: int, height: int) -> List[str]:
    """Parses a Pascal VOC XML file and returns normalized YOLO label lines."""
    yolo_lines = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in LABEL_MAP:
                continue

            cls_id = LABEL_MAP[name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            # Clamp bounding boxes
            xmin = max(0.0, min(float(width), xmin))
            ymin = max(0.0, min(float(height), ymin))
            xmax = max(0.0, min(float(width), xmax))
            ymax = max(0.0, min(float(height), ymax))

            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 1.0 or bh <= 1.0:
                continue

            xc = xmin + bw / 2.0
            yc = ymin + bh / 2.0

            # Normalize [0, 1]
            n_xc = xc / width
            n_yc = yc / height
            n_bw = bw / width
            n_bh = bh / height

            yolo_lines.append(f"{cls_id} {n_xc:.6f} {n_yc:.6f} {n_bw:.6f} {n_bh:.6f}")
    except Exception as e:
        print(f"[!] Error parsing {xml_file}: {e}")

    return yolo_lines


def prepare_shwd_yolo(voc_root: str = "VOC2028/VOC2028",
                      output_dir: str = "shwd_yolo",
                      max_images: int = 0) -> str:
    """
    Transforms VOC2028 into standard YOLOv8 layout.
    """
    voc_path = Path(voc_root)
    out_path = Path(output_dir).resolve()

    images_dir = voc_path / "JPEGImages"
    annots_dir = voc_path / "Annotations"
    splits_dir = voc_path / "ImageSets" / "Main"

    if not images_dir.exists() or not annots_dir.exists():
        raise FileNotFoundError(f"VOC2028 directory not found at: {voc_root}")

    print(f"[*] Preparing SHWD (VOC2028) YOLO dataset at: {out_path}")

    for split in ["train", "val", "test"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Use official splits
    split_files = {
        "train": splits_dir / "train.txt",
        "val": splits_dir / "val.txt",
        "test": splits_dir / "test.txt"
    }

    from PIL import Image

    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, s_file in split_files.items():
        if not s_file.exists():
            continue

        with open(s_file, "r", encoding="utf-8") as f:
            img_ids = [line.strip() for line in f if line.strip()]

        if max_images > 0:
            img_ids = img_ids[:max_images]

        for img_id in img_ids:
            img_path = images_dir / f"{img_id}.jpg"
            xml_path = annots_dir / f"{img_id}.xml"

            if not img_path.exists() or not xml_path.exists():
                continue

            dst_img = out_path / "images" / split_name / f"{img_id}.jpg"
            dst_txt = out_path / "labels" / split_name / f"{img_id}.txt"

            # Parse size
            try:
                tree = ET.parse(xml_path)
                size_elem = tree.getroot().find("size")
                w = int(size_elem.find("width").text)
                h = int(size_elem.find("height").text)
            except Exception:
                with Image.open(img_path) as im:
                    w, h = im.size

            yolo_boxes = convert_xml_to_yolo(xml_path, w, h)

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            with open(dst_txt, "w", encoding="utf-8") as tf:
                tf.write("\n".join(yolo_boxes) + "\n")

            counts[split_name] += 1

    print(f"[OK] SHWD YOLO Dataset Created: Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")

    yaml_content = f"""# SafeVision Harmonized SHWD (VOC2028) PPE Dataset
path: {out_path.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: vest
  2: helmet
"""
    yaml_file = out_path / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"[OK] Created dataset config: {yaml_file}")
    return str(yaml_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VOC2028 to YOLO format")
    parser.add_argument("--voc-dir", type=str, default="VOC2028/VOC2028", help="Path to VOC2028")
    parser.add_argument("--out-dir", type=str, default="shwd_yolo", help="Output YOLO directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images per split (0 for all)")
    args = parser.parse_args()

    prepare_shwd_yolo(args.voc_dir, args.out_dir, args.limit)
