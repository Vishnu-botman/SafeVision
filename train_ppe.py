"""
SafeVision: Automated YOLO Dataset Preparation and Model Training Pipeline
Specialized for Model B (PPE Detector) using harmonized CHV dataset (person, vest, helmet).

Usage:
    # 1. Prepare dataset and run a quick 5-epoch verification on CPU:
    python train_ppe.py --epochs 5 --batch 8 --model yolov8n.pt

    # 2. Prepare dataset only (e.g., to upload to Colab or run later):
    python train_ppe.py --prepare-only

    # 3. Full training from scratch or pretrained with custom epochs:
    python train_ppe.py --epochs 50 --batch 16 --model yolov8n.pt --device cpu
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


# Harmonization map from CHV raw IDs to our target 3-class ontology
# Raw: 0: person, 1: vest, 2: blue_helmet, 3: red_helmet, 4: white_helmet, 5: yellow_helmet
RAW_TO_HARMONIZED = {
    0: 0,  # person -> 0
    1: 1,  # vest   -> 1
    2: 2,  # blue_helmet -> 2 (helmet)
    3: 2,  # red_helmet  -> 2 (helmet)
    4: 2,  # white_helmet -> 2 (helmet)
    5: 2   # yellow_helmet -> 2 (helmet)
}

CLASS_NAMES = {
    0: "person",
    1: "vest",
    2: "helmet"
}


def prepare_yolo_dataset(raw_dir: str = "CHV_dataset/CHV_dataset",
                          output_dir: str = "chv_yolo") -> str:
    """
    Transforms raw CHV dataset into a standard YOLOv8 directory structure:
    chv_yolo/
      images/train, images/val, images/test
      labels/train, labels/val, labels/test
      dataset.yaml
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir).resolve()
    
    images_dir = raw_path / "images"
    annots_dir = raw_path / "annotations"
    split_dir = raw_path / "data split"

    if not images_dir.exists() or not annots_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found at: {raw_dir}")

    print(f"[*] Preparing YOLO-formatted dataset at: {out_path}")

    # Create directory tree
    for split in ["train", "val", "test"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Read split files
    split_files = {
        "train": split_dir / "train.txt",
        "val": split_dir / "valid.txt",
        "test": split_dir / "test.txt"
    }

    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, split_file in split_files.items():
        if not split_file.exists():
            print(f"[!] Warning: Split file {split_file} not found. Skipping {split_name}.")
            continue

        with open(split_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            # Expected format: CHV_dataset/images/ppe_XXXX.jpg or ppe_XXXX.jpg
            img_basename = Path(line).name
            stem = Path(img_basename).stem
            
            src_img = images_dir / img_basename
            src_annot = annots_dir / f"{stem}.txt"

            if not src_img.exists():
                continue

            dst_img = out_path / "images" / split_name / img_basename
            dst_annot = out_path / "labels" / split_name / f"{stem}.txt"

            # Copy image
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Harmonize and copy label
            if src_annot.exists():
                harmonized_lines = []
                with open(src_annot, "r", encoding="utf-8") as af:
                    for annot_line in af:
                        parts = annot_line.strip().split()
                        if len(parts) >= 5:
                            raw_cls = int(parts[0])
                            if raw_cls in RAW_TO_HARMONIZED:
                                harm_cls = RAW_TO_HARMONIZED[raw_cls]
                                harmonized_lines.append(f"{harm_cls} " + " ".join(parts[1:5]))

                with open(dst_annot, "w", encoding="utf-8") as out_af:
                    out_af.write("\n".join(harmonized_lines) + "\n")

            counts[split_name] += 1

    print(f"[OK] Dataset prepared: Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")

    # Create dataset.yaml
    yaml_content = f"""# SafeVision Harmonized CHV Dataset Configuration for PPE Detector
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


def train_model(dataset_yaml: str,
                epochs: int = 10,
                batch_size: int = 8,
                imgsz: int = 640,
                base_model: str = "yolov8n.pt",
                device: str = "auto",
                project_name: str = "safevision_training",
                run_name: str = "ppe_detector") -> str:
    """
    Trains the YOLO model using Ultralytics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Ultralytics is required. Please install via: pip install ultralytics")
        sys.exit(1)

    print("\n=======================================================")
    print("         SAFEVISION MODEL B (PPE) TRAINING             ")
    print("=======================================================")
    print(f"  Base Model:    {base_model}")
    print(f"  Dataset YAML:  {dataset_yaml}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Image Size:    {imgsz}")
    print(f"  Device:        {device}")
    print("=======================================================\n")

    model = YOLO(base_model)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project_name,
        name=run_name,
        exist_ok=True,
        verbose=True,
        workers=2,
        val=True
    )

    best_weights_path = Path(project_name) / run_name / "weights" / "best.pt"
    if best_weights_path.exists():
        print(f"\n[OK] Training complete! Best weights saved at: {best_weights_path}")
        
        # Save to safevision models weights directory for automatic deployment
        dest_dir = Path("safevision/models/weights")
        dest_dir.mkdir(parents=True, exist_ok=True)
        deployed_path = dest_dir / "ppe_detector_best.pt"
        shutil.copy2(best_weights_path, deployed_path)
        print(f"[OK] Automatically deployed best weights to: {deployed_path}")
        return str(deployed_path)
    else:
        print("[!] Note: Completed run, but best.pt was not found at standard path.")
        return str(best_weights_path)


def main():
    parser = argparse.ArgumentParser(description="SafeVision PPE Detector Training Script")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size in pixels (default: 640)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model checkpoint or YAML (default: yolov8n.pt)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu', '0', or 'auto')")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare the dataset without training")
    parser.add_argument("--raw-dir", type=str, default="CHV_dataset/CHV_dataset", help="Raw CHV directory")
    parser.add_argument("--out-dir", type=str, default="chv_yolo", help="Prepared YOLO dataset directory")

    args = parser.parse_args()

    # Step 1: Prepare dataset
    yaml_path = prepare_yolo_dataset(raw_dir=args.raw_dir, output_dir=args.out_dir)

    if args.prepare_only:
        print("\n[OK] Dataset preparation complete. Ready for training.")
        return

    # Step 2: Train model
    train_model(
        dataset_yaml=yaml_path,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        base_model=args.model,
        device=args.device
    )


if __name__ == "__main__":
    main()
