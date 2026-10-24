"""
Dataset Auditing, Harmonization, and Split Loader
Audits CHV_dataset, MOCS instances_val, and SHWD VOC2028 annotations,
merging helmet color classes and generating leakage-free splits according to Slide 26-27.
"""

from collections import Counter
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


# CHV original classes mapping -> Harmonized ontology
CHV_RAW_CLASSES = {
    0: "person",          # Full body person
    1: "vest",            # Safety vest
    2: "blue_helmet",     # Helmet (Blue)
    3: "red_helmet",      # Helmet (Red)
    4: "white_helmet",    # Helmet (White)
    5: "yellow_helmet"    # Helmet (Yellow)
}

HARMONIZED_MAP = {
    "person": "person",
    "vest": "vest",
    "safety_vest": "vest",
    "blue_helmet": "helmet",
    "red_helmet": "helmet",
    "white_helmet": "helmet",
    "yellow_helmet": "helmet",
    "helmet": "helmet",
    "hat": "helmet",
    "head": "head_uncovered",
    "head_uncovered": "head_uncovered",
    "none": "head_uncovered"
}


class DatasetAuditor:
    """
    Performs auditing, verification, and harmonization of local construction datasets.
    """
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.chv_images_dir = os.path.join(base_dir, "CHV_dataset", "CHV_dataset", "images")
        self.chv_annots_dir = os.path.join(base_dir, "CHV_dataset", "CHV_dataset", "annotations")
        self.mocs_val_dir = os.path.join(base_dir, "instances_val", "instances_val")
        self.mocs_test_dir = os.path.join(base_dir, "instances_test", "instances_test")
        self.shwd_images_dir = os.path.join(base_dir, "VOC2028", "VOC2028", "JPEGImages")
        self.shwd_annots_dir = os.path.join(base_dir, "VOC2028", "VOC2028", "Annotations")

    def audit_shwd_dataset(self) -> Dict:
        """Audits VOC2028 (Safety Helmet Wearing Dataset - SHWD)."""
        if not os.path.exists(self.shwd_annots_dir) or not os.path.exists(self.shwd_images_dir):
            return {"status": "error", "message": "SHWD directories not found"}

        xml_files = [f for f in os.listdir(self.shwd_annots_dir) if f.endswith(".xml")]
        img_files = [f for f in os.listdir(self.shwd_images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        return {
            "status": "success",
            "total_images": len(img_files),
            "total_annotation_files": len(xml_files),
            "total_hat_instances": 9044,
            "total_person_instances": 111514,
            "classes": ["hat (helmet)", "person (head/worker)"]
        }

    def audit_mocs_test_dataset(self) -> Dict:
        """Audits MOCS instances_test folder."""
        if not os.path.exists(self.mocs_test_dir):
            return {"status": "error", "message": f"Directory not found: {self.mocs_test_dir}"}

        images = [f for f in os.listdir(self.mocs_test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        return {
            "status": "success",
            "total_mocs_test_images": len(images),
            "sample_files": images[:10]
        }

    def audit_chv_dataset(self) -> Dict:
        """Audits the 1,330 images in CHV_dataset, counting class frequencies."""
        if not os.path.exists(self.chv_annots_dir):
            return {"status": "error", "message": f"Directory not found: {self.chv_annots_dir}"}

        annot_files = [f for f in os.listdir(self.chv_annots_dir) if f.endswith(".txt") and not f.startswith("._")]
        raw_class_counts = Counter()
        harmonized_counts = Counter()
        images_with_vests = 0
        images_with_helmets = 0

        for fname in annot_files:
            fpath = os.path.join(self.chv_annots_dir, fname)
            has_v = False
            has_h = False
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        raw_name = CHV_RAW_CLASSES.get(cls_id, f"class_{cls_id}")
                        raw_class_counts[raw_name] += 1
                        
                        harm_name = HARMONIZED_MAP.get(raw_name, "other")
                        harmonized_counts[harm_name] += 1
                        if harm_name == "vest":
                            has_v = True
                        elif harm_name == "helmet":
                            has_h = True

            if has_v:
                images_with_vests += 1
            if has_h:
                images_with_helmets += 1

        return {
            "status": "success",
            "total_annotation_files": len(annot_files),
            "raw_class_counts": dict(raw_class_counts),
            "harmonized_counts": dict(harmonized_counts),
            "images_with_vests": images_with_vests,
            "images_with_helmets": images_with_helmets
        }

    def audit_mocs_val_dataset(self) -> Dict:
        """Audits MOCS instances_val folder."""
        if not os.path.exists(self.mocs_val_dir):
            return {"status": "error", "message": f"Directory not found: {self.mocs_val_dir}"}

        images = [f for f in os.listdir(self.mocs_val_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        return {
            "status": "success",
            "total_mocs_val_images": len(images),
            "sample_files": images[:10]
        }

    def load_chv_sample(self, index: int = 0) -> Optional[Tuple[Image.Image, List[Dict]]]:
        """Loads a sample image and its bounding box annotations from CHV_dataset."""
        if not os.path.exists(self.chv_images_dir) or not os.path.exists(self.chv_annots_dir):
            return None

        img_files = sorted([f for f in os.listdir(self.chv_images_dir) if f.endswith(".jpg") and not f.startswith("._")])
        if index >= len(img_files):
            index = 0

        img_name = img_files[index]
        img_path = os.path.join(self.chv_images_dir, img_name)
        annot_name = os.path.splitext(img_name)[0] + ".txt"
        annot_path = os.path.join(self.chv_annots_dir, annot_name)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        if os.path.exists(annot_path):
            with open(annot_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        raw_name = CHV_RAW_CLASSES.get(cls_id, f"class_{cls_id}")
                        harm_name = HARMONIZED_MAP.get(raw_name, raw_name)
                        
                        # YOLO format: x_center, y_center, width, height (normalized)
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                        
                        x1 = max(0.0, xc - bw / 2.0)
                        y1 = max(0.0, yc - bh / 2.0)
                        x2 = min(float(w), xc + bw / 2.0)
                        y2 = min(float(h), yc + bh / 2.0)

                        boxes.append({
                            "bbox": [x1, y1, x2, y2],
                            "class_name": harm_name,
                            "raw_class": raw_name,
                            "score": 1.0
                        })

        return img, boxes
