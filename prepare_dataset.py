"""
Chuyển đổi LabelMe JSON → YOLO detection format
Cấu trúc output:
  dataset/
    images/train/  images/val/
    labels/train/  labels/val/
    dataset.yaml
"""

import json
import shutil
import random
from pathlib import Path

DATA_DIRS  = [
    Path(__file__).parent / "data",
    Path(__file__).parent / "data-alll",
]
OUT_DIR    = Path(__file__).parent / "dataset"
VAL_RATIO  = 0.15   # 15% validation
SEED       = 42


def labelme_to_yolo(points, img_w: int, img_h: int, shape_type: str) -> str | None:
    """
    Chuyển 1 shape LabelMe → YOLO line.
    Hỗ trợ rectangle và polygon.
    Trả về None nếu không hỗ trợ.
    """
    if shape_type == "rectangle":
        (x1, y1), (x2, y2) = points[0], points[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    elif shape_type == "polygon":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    return None


def prepare(val_ratio=VAL_RATIO, seed=SEED, verbose=True):
    random.seed(seed)

    # Thu thập tất cả JSON có ảnh tương ứng (quét đệ quy tất cả DATA_DIRS)
    pairs = []
    seen_stems = set()
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for j in sorted(data_dir.rglob("*.json")):
            img = j.with_suffix(".png")
            if not img.exists():
                img = j.with_suffix(".jpg")
            if img.exists() and img.stem not in seen_stems:
                pairs.append((j, img))
                seen_stems.add(img.stem)

    if not pairs:
        print("Không tìm thấy cặp json+ảnh nào!")
        return None

    # Thu thập tất cả labels (để build class map)
    all_labels = set()
    for jf, _ in pairs:
        d = json.load(open(jf, encoding="utf-8"))
        for s in d["shapes"]:
            all_labels.add(s["label"])
    class_names = sorted(all_labels)
    class_idx   = {c: i for i, c in enumerate(class_names)}

    # Shuffle và split
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * val_ratio))
    val_set = set(range(len(pairs) - n_val, len(pairs)))

    # Tạo thư mục
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skipped": 0, "boxes": 0}

    for i, (jf, img_f) in enumerate(pairs):
        split = "val" if i in val_set else "train"
        d = json.load(open(jf, encoding="utf-8"))
        img_w = d["imageWidth"]
        img_h = d["imageHeight"]

        lines = []
        for s in d["shapes"]:
            label = s["label"]
            cid   = class_idx[label]
            yolo  = labelme_to_yolo(s["points"], img_w, img_h, s["shape_type"])
            if yolo:
                lines.append(f"{cid} {yolo}")
                stats["boxes"] += 1

        if not lines:
            stats["skipped"] += 1
            continue

        label_text = "\n".join(lines)
        shutil.copy2(img_f, OUT_DIR / "images" / split / img_f.name)
        (OUT_DIR / "labels" / split / (img_f.stem + ".txt")).write_text(label_text)
        stats[split] += 1

    # Ghi dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    if verbose:
        print(f"Classes : {class_names}")
        print(f"Train   : {stats['train']} ảnh")
        print(f"Val     : {stats['val']} ảnh")
        print(f"Skipped : {stats['skipped']} ảnh (không có bbox)")
        print(f"Boxes   : {stats['boxes']}")
        print(f"YAML    : {yaml_path}")

    return str(yaml_path)


if __name__ == "__main__":
    prepare()
