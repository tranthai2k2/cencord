"""
NSFW Tool - phiên bản gộp: Blur nhanh (batch GPU) + Review vẽ-tay + Train.

Khác biệt so với nsfw_blur_ui.py:
  - Blur chạy BATCH nhiều ảnh/lần trên GPU (nhanh hơn nhiều), gộp mask nhiều model rồi blur 1 lần.
  - Tab Review cho phép KÉO 1 KHUNG lên vùng model bỏ sót -> blur ngay lên ảnh output
    + tự lưu label (LabelMe JSON) vào train-collect/ để train sau. => Không cần train lại mới có output sạch.
  - Tab Train chỉ chạy khi muốn nâng model (thỉnh thoảng), không bắt buộc mỗi lần.

Self-check:  python nsfw_tool.py --selftest
"""

import os
import sys
import json
import time
import queue
import shutil
import threading
import warnings
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BASE_DIR = Path(__file__).parent
YOLO_MODEL_DIR = BASE_DIR / "model-yolo"
TRAIN_COLLECT_DIR = BASE_DIR / "train-collect"      # label vẽ tay tích luỹ ở đây
DATASET_DIR = BASE_DIR / "dataset"
FIXED_GAUSSIAN_RADIUS = 4
FIXED_MOSAIC_CELL_SIZE = 9
MANIFEST_NAME = "blur_manifest.json"

# Class KHÔNG kiểm duyệt (khớp substring, chữ thường). Chương trình bỏ qua núm vú.
BLUR_SKIP_CLASSES = ("nipple",)
# Nhãn để vẽ tay trong Review. Ô combobox sửa được -> gõ nhãn khác cũng được (giữ nhất quán).
# "testicles" = dái. anus/testicles: model hiện chưa bắt -> phải vẽ tay, tích data để train sau.
LABEL_CHOICES = ["penis", "pussy", "anus", "testicles", "nipple", "breast", "buttocks"]

# Pre-import ultralytics, chặn output noise của torch/C-ext
_YOLO_CLASS = None
try:
    _old_err, _old_out = sys.stderr, sys.stdout
    sys.stderr = sys.stdout = open(os.devnull, "w")
    try:
        from ultralytics import YOLO as _YOLO_CLASS
    finally:
        sys.stdout.close()
        sys.stderr, sys.stdout = _old_err, _old_out
except Exception:
    pass


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return "0", torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu", "CPU"


# ─────────────────────────── BLUR CORE ───────────────────────────────

def _imread(path):
    """Đọc ảnh -> BGR ndarray, fallback PIL cho format lạ. None nếu fail."""
    import cv2
    import numpy as np
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(str(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _make_blur_full(img):
    """Gaussian + mosaic toàn ảnh (dùng làm nguồn blend theo mask, và cho ROI)."""
    import cv2
    h, w = img.shape[:2]
    ksize = FIXED_GAUSSIAN_RADIUS * 2 + 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    cell = max(1, FIXED_MOSAIC_CELL_SIZE)
    small = cv2.resize(blurred, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _blur_region(img, x1, y1, x2, y2):
    """Blur 1 hình chữ nhật tại chỗ (dùng cho vẽ tay)."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return img
    img[y1:y2, x1:x2] = _make_blur_full(img[y1:y2, x1:x2])
    return img


def _read_gif(path):
    from PIL import Image, ImageSequence
    with Image.open(path) as gif:
        loop = gif.info.get("loop")
        frames, durations = [], []
        for frame in ImageSequence.Iterator(gif):
            frames.append(frame.convert("RGBA"))
            durations.append(frame.info.get("duration", 100))
    return frames, durations, loop


def _gif_view(frame, boxes, index):
    import cv2
    import numpy as np
    view = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
    for start, end, box in boxes:
        if start <= index <= end:
            _blur_region(view, *box)
    return view


def _save_gif(path, frames, durations, loop, boxes):
    """Ghi GIF qua file tạm, chỉ thay file hiện tại khi encode thành công."""
    import cv2
    import tempfile
    from PIL import Image
    output = []
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(_gif_view(frame, boxes, i), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGBA")
        image.putalpha(frame.getchannel("A"))
        if frame.getchannel("A").getextrema() == (255, 255):
            image = image.convert("RGB")
        output.append(image)
    options = {"loop": loop} if loop is not None else {}
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, suffix=".gif", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        output[0].save(temporary, format="GIF", save_all=True, append_images=output[1:],
                       duration=durations, disposal=2, optimize=False, **options)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_mask(pred, img_w, img_h, dilate, feather, min_ratio, max_ratio):
    """Result 1 model -> (combined_mask float32 HxW, details list). Seg dùng pixel mask, detect dùng bbox."""
    import cv2
    import numpy as np
    combined = np.zeros((img_h, img_w), dtype=np.float32)
    details = []
    img_area = img_w * img_h
    if pred.boxes is None:
        return combined, details
    for i, box in enumerate(pred.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        ratio = max(1, (x2 - x1) * (y2 - y1)) / img_area
        if not (min_ratio <= ratio <= max_ratio):
            continue
        if any(s in pred.names[int(box.cls[0])].lower() for s in BLUR_SKIP_CLASSES):
            continue  # không kiểm duyệt núm vú
        if pred.masks is not None:
            m = pred.masks.data[i].cpu().numpy()
            mask_r = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_r = np.zeros((img_h, img_w), dtype=np.float32)
            mask_r[max(0, y1):y2, max(0, x1):x2] = 1.0
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
            mask_r = cv2.dilate(mask_r, k)
        if feather > 0:
            ks = feather * 2 + 1
            mask_r = cv2.GaussianBlur(mask_r, (ks, ks), 0)
        combined = np.maximum(combined, mask_r)
        details.append({"class": pred.names[int(box.cls[0])], "score": round(float(box.conf[0]), 3)})
    return combined, details


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _same_path(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return Path(a).absolute() == Path(b).absolute()


def blur_folder(images, input_dir, output_dir, models, params, device,
                log_cb=print, progress_cb=None, status_cb=None, cancel=None):
    """
    Blur batch. images: list[Path]. models: list[YOLO]. params: dict.
    Ghi output giữ cấu trúc thư mục + blur_manifest.json. Trả về stats dict.
    """
    import cv2
    import numpy as np
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    overwrite_input = _same_path(input_dir, output_dir)
    conf = params["conf"]; dilate = params["dilate"]; feather = params["feather"]
    min_r = params["min_ratio"]; max_r = params["max_ratio"]
    imgsz = params["imgsz"]; batch = params["batch"]
    half = device != "cpu"

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "overwrite": overwrite_input,
        "detections": {},
    }
    stats = {"blurred": 0, "clean": 0, "error": 0, "regions": 0, "written": 0, "skipped_clean": 0}
    total = len(images)
    done = 0
    t0 = time.perf_counter()

    for chunk in _chunks(images, batch):
        if cancel is not None and cancel.is_set():
            log_cb("⏹ Đã dừng."); break
        arrs, keep = [], []
        for p in chunk:
            im = _imread(p)
            if im is None:
                stats["error"] += 1
                log_cb(f"  LỖI đọc: {p.name}")
                done += 1
                continue
            arrs.append(im); keep.append(p)
        if not arrs:
            if progress_cb: progress_cb(done / total * 100)
            continue

        combined = [np.zeros(a.shape[:2], np.float32) for a in arrs]
        details = [[] for _ in arrs]
        for model in models:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.predict(arrs, imgsz=imgsz, conf=conf, half=half,
                                        device=device, batch=len(arrs), verbose=False)
            except Exception as e:
                log_cb(f"  model lỗi: {e}")
                continue
            for idx, pred in enumerate(res):
                h, w = arrs[idx].shape[:2]
                m, d = _build_mask(pred, w, h, dilate, feather, min_r, max_r)
                combined[idx] = np.maximum(combined[idx], m)
                details[idx].extend(d)

        for idx, p in enumerate(keep):
            im = arrs[idx]
            n = len(details[idx])
            if n > 0:
                blurred = _make_blur_full(im)
                a = combined[idx][:, :, None]
                im = (blurred * a + im * (1 - a)).astype(np.uint8)
                stats["blurred"] += 1; stats["regions"] += n
            else:
                stats["clean"] += 1
            rel = p.relative_to(input_dir)
            outp = output_dir / rel
            if n > 0 or not overwrite_input:
                outp.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(outp), im):
                    stats["written"] += 1
                else:
                    stats["error"] += 1
                    log_cb(f"  LỖI ghi: {outp}")
            else:
                stats["skipped_clean"] += 1
            manifest["detections"][rel.as_posix()] = details[idx]
            done += 1
            if status_cb:
                status_cb(f"[{done}/{total}] {p.name}  ({n} vùng)")
            if progress_cb:
                progress_cb(done / total * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    stats["wall"] = time.perf_counter() - t0
    stats["avg"] = stats["wall"] / max(1, done)
    return stats


# ─────────────────────────── LABEL COLLECT ───────────────────────────

def save_labelme_box(orig_img_path, box_img_coords, label, img_w, img_h):
    """Copy ảnh GỐC (chưa blur) + ghi/append 1 rectangle vào LabelMe JSON trong train-collect/."""
    import cv2
    TRAIN_COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    stem = orig_img_path.stem
    dst_img = TRAIN_COLLECT_DIR / f"{stem}.jpg"
    dst_json = TRAIN_COLLECT_DIR / f"{stem}.json"
    if not dst_img.exists():
        im = _imread(orig_img_path)
        if im is None:
            return False
        cv2.imwrite(str(dst_img), im)

    if dst_json.exists():
        data = json.loads(dst_json.read_text(encoding="utf-8"))
    else:
        data = {"version": "5.0.1", "flags": {}, "shapes": [],
                "imagePath": dst_img.name, "imageData": None,
                "imageHeight": img_h, "imageWidth": img_w}
    x1, y1, x2, y2 = box_img_coords
    data["shapes"].append({
        "label": label, "points": [[x1, y1], [x2, y2]],
        "group_id": None, "shape_type": "rectangle", "flags": {},
    })
    dst_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


# Gộp class từ nhiều model về nhãn chuẩn (để train 1 model đa-class nhất quán)
LABEL_NORM = {"Pussy": "pussy", "pussy": "pussy", "vagina_exposed": "pussy",
              "penis": "penis", "nipple": "nipple"}


def _norm_label(c):
    return LABEL_NORM.get(c, c.lower())


def write_labelme_boxes(orig_img_path, items, img_w, img_h):
    """Ghi ĐÈ LabelMe JSON với toàn bộ box (dùng cho seed). items: list (label,x1,y1,x2,y2)."""
    import cv2
    TRAIN_COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    stem = orig_img_path.stem
    dst_img = TRAIN_COLLECT_DIR / f"{stem}.jpg"
    if not dst_img.exists():
        im = _imread(orig_img_path)
        if im is None:
            return False
        cv2.imwrite(str(dst_img), im)
    shapes = [{"label": lb, "points": [[x1, y1], [x2, y2]], "group_id": None,
               "shape_type": "rectangle", "flags": {}} for (lb, x1, y1, x2, y2) in items]
    (TRAIN_COLLECT_DIR / f"{stem}.json").write_text(json.dumps(
        {"version": "5.0.1", "flags": {}, "shapes": shapes, "imagePath": dst_img.name,
         "imageData": None, "imageHeight": img_h, "imageWidth": img_w},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def seed_labels(images, models, conf, imgsz, batch, device, min_ratio, max_ratio,
                log_cb=print, progress_cb=None, cancel=None):
    """Chạy ensemble sinh nhãn (auto-label) -> LabelMe JSON trong train-collect/. Trả về (n_ảnh, n_box)."""
    half = device != "cpu"
    n_img = n_box = done = 0
    total = len(images)
    for chunk in _chunks(images, batch):
        if cancel is not None and cancel.is_set():
            log_cb("⏹ Đã dừng."); break
        arrs, keep = [], []
        for p in chunk:
            im = _imread(p)
            if im is None:
                done += 1; continue
            arrs.append(im); keep.append(p)
        if not arrs:
            continue
        per_img = [[] for _ in arrs]
        for model in models:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.predict(arrs, imgsz=imgsz, conf=conf, half=half,
                                        device=device, batch=len(arrs), verbose=False)
            except Exception as e:
                log_cb(f"  model lỗi: {e}"); continue
            for idx, pred in enumerate(res):
                if pred.boxes is None:
                    continue
                h, w = arrs[idx].shape[:2]; area = h * w
                for box in pred.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    ratio = max(1, (x2 - x1) * (y2 - y1)) / area
                    if not (min_ratio <= ratio <= max_ratio):
                        continue
                    lb = _norm_label(pred.names[int(box.cls[0])])
                    per_img[idx].append((lb, max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        for idx, p in enumerate(keep):
            h, w = arrs[idx].shape[:2]
            if per_img[idx]:
                write_labelme_boxes(p, per_img[idx], w, h)
                n_img += 1; n_box += len(per_img[idx])
            done += 1
            if progress_cb:
                progress_cb(done / total * 100)
    return n_img, n_box


def pop_labelme_box(orig_img_path):
    """Xoá shape cuối (dùng cho Undo). Xoá file nếu hết shape."""
    stem = orig_img_path.stem
    dst_json = TRAIN_COLLECT_DIR / f"{stem}.json"
    if not dst_json.exists():
        return
    data = json.loads(dst_json.read_text(encoding="utf-8"))
    if data.get("shapes"):
        data["shapes"].pop()
    if data.get("shapes"):
        dst_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        dst_json.unlink(missing_ok=True)
        (TRAIN_COLLECT_DIR / f"{stem}.jpg").unlink(missing_ok=True)


# ─────────────────────────── UI ──────────────────────────────────────

import tkinter as tk
from tkinter import ttk, filedialog


class App:
    def __init__(self, root):
        self.root = root
        root.title("NSFW Tool — Blur / Review / Train")
        root.geometry("1040x820")
        root.minsize(920, 700)
        self.C = {
            "bg": "#0d1017", "panel": "#161b26", "panel2": "#1e2533", "border": "#2b3446",
            "text": "#e8ecf4", "muted": "#8b96ab", "accent": "#8b93ff", "accent_d": "#4b53d6",
            "ok": "#3fb950", "danger": "#6e2b39", "log_bg": "#0a0d13", "log_fg": "#cdd6e6",
        }
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.models = {}          # relpath -> YOLO
        self.model_vars = {}      # relpath -> BooleanVar
        self.device, self.device_name = _device()
        self._theme()
        self._build()
        self._poll()

    # ---- theme ----
    def _theme(self):
        C = self.C
        self.root.configure(bg=C["bg"])
        s = ttk.Style(self.root)
        try: s.theme_use("clam")
        except tk.TclError: pass
        s.configure(".", font=("Segoe UI", 10))
        s.configure("TFrame", background=C["bg"])
        s.configure("Panel.TFrame", background=C["panel"])
        s.configure("TLabel", background=C["bg"], foreground=C["text"])
        s.configure("Muted.TLabel", background=C["panel"], foreground=C["muted"])
        s.configure("Panel.TLabel", background=C["panel"], foreground=C["text"])
        s.configure("Title.TLabel", background=C["bg"], foreground=C["text"], font=("Segoe UI Semibold", 17))
        s.configure("Card.TLabelframe", background=C["panel"], foreground=C["text"], bordercolor=C["border"], relief="solid")
        s.configure("Card.TLabelframe.Label", background=C["panel"], foreground=C["text"], font=("Segoe UI Semibold", 10))
        for e in ("TEntry", "TSpinbox", "TCombobox"):
            s.configure(e, fieldbackground="#0f151b", foreground=C["text"], bordercolor=C["border"], arrowcolor=C["text"])
        s.map("TCombobox", fieldbackground=[("readonly", "#0f151b")])
        s.configure("TButton", background=C["panel2"], foreground=C["text"], bordercolor=C["border"], padding=(11, 6))
        s.map("TButton", background=[("active", "#26313a"), ("disabled", "#151b20")], foreground=[("disabled", "#65727d")])
        s.configure("Accent.TButton", background=C["accent_d"], foreground="#f6f8ff", padding=(13, 7))
        s.map("Accent.TButton", background=[("active", "#5a63e8"), ("disabled", "#242b3a")])
        s.configure("Danger.TButton", background=C["danger"], foreground="#fff4f6", padding=(11, 7))
        s.map("Danger.TButton", background=[("active", "#793240"), ("disabled", "#25191c")])
        s.configure("TCheckbutton", background=C["panel"], foreground=C["text"])
        s.map("TCheckbutton", background=[("active", C["panel"])])
        s.configure("Horizontal.TProgressbar", troughcolor="#0b0f13", background=C["accent"], bordercolor=C["border"])
        s.configure("TNotebook", background=C["bg"], borderwidth=0)
        s.configure("TNotebook.Tab", background="#151b21", foreground=C["muted"], padding=(20, 9), font=("Segoe UI Semibold", 10))
        s.map("TNotebook.Tab", background=[("selected", C["panel"])], foreground=[("selected", C["text"]), ("active", C["text"])])
        s.configure("Vertical.TScrollbar", background=C["panel2"], troughcolor=C["bg"], bordercolor=C["border"], arrowcolor=C["muted"])

    def _card(self, parent, title):
        return ttk.LabelFrame(parent, text=title, padding=11, style="Card.TLabelframe")

    def _log_widget(self, parent):
        t = tk.Text(parent, state="disabled", bg=self.C["log_bg"], fg=self.C["log_fg"],
                    relief="flat", padx=11, pady=9, font=("Cascadia Mono", 9), wrap="word")
        sb = ttk.Scrollbar(parent, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); t.pack(fill="both", expand=True)
        return t

    def _build(self):
        shell = ttk.Frame(self.root)
        shell.pack(fill="both", expand=True, padx=16, pady=14)
        head = ttk.Frame(shell); head.pack(fill="x", pady=(0, 12))
        titlebox = ttk.Frame(head); titlebox.pack(side="left")
        ttk.Label(titlebox, text="NSFW Tool", style="Title.TLabel").pack(anchor="w")
        ttk.Label(titlebox, text="Blur hàng loạt (GPU) · Review sửa tay · Train",
                  style="Muted.TLabel", background=self.C["bg"]).pack(anchor="w")
        # device pill bên phải
        gpu = self.device != "cpu"
        pill = tk.Label(head, text=("● " if gpu else "○ ") + self.device_name,
                        bg=self.C["panel2"], fg=(self.C["ok"] if gpu else self.C["muted"]),
                        font=("Segoe UI Semibold", 10), padx=12, pady=6)
        pill.pack(side="right", pady=(4, 0))
        nb = ttk.Notebook(shell); nb.pack(fill="both", expand=True)
        self.t_blur = ttk.Frame(nb); self.t_review = ttk.Frame(nb); self.t_train = ttk.Frame(nb)
        nb.add(self.t_blur, text="  1 · Blur  ")
        nb.add(self.t_review, text="  2 · Review + sửa tay  ")
        nb.add(self.t_train, text="  3 · Train (thỉnh thoảng)  ")
        self._build_blur()
        self._build_review()
        self._build_train()

    # ==================== BLUR TAB ====================
    def _build_blur(self):
        r = self.t_blur
        r.columnconfigure(0, weight=1); r.rowconfigure(5, weight=1)

        f_in = self._card(r, "Thư mục ảnh đầu vào"); f_in.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        f_in.columnconfigure(0, weight=1)
        self.in_var = tk.StringVar()
        self._path_row(f_in, self.in_var, self._pick_in).grid(row=0, column=0, sticky="ew")
        self.out_var = tk.StringVar()
        self.overwrite_var = tk.BooleanVar(value=True)
        ttk.Label(
            f_in,
            text="Nhập output nếu muốn xuất ra thư mục khác, hoặc bật ghi đè để sửa trực tiếp ảnh gốc",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(8, 2))
        self.output_row = self._path_row(f_in, self.out_var, self._pick_out)
        self.output_row.grid(row=2, column=0, sticky="ew")
        ttk.Checkbutton(
            f_in,
            text="Ghi đè trực tiếp ảnh gốc",
            variable=self.overwrite_var,
            command=self._sync_output_mode,
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.recursive = tk.BooleanVar(value=False)
        ttk.Checkbutton(f_in, text="Bao gồm thư mục con", variable=self.recursive).grid(row=4, column=0, sticky="w", pady=(6, 0))
        self._sync_output_mode()

        f_m = self._card(r, "YOLO models (mặc định bật TẤT CẢ — gộp mask nhiều model)"); f_m.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        f_m.columnconfigure(0, weight=1)
        hdr = ttk.Frame(f_m, style="Panel.TFrame"); hdr.grid(row=0, column=0, sticky="ew")
        ttk.Button(hdr, text="Refresh", command=self._refresh_models).pack(side="left")
        ttk.Button(hdr, text="+ Thêm .pt", command=self._add_model).pack(side="left", padx=(6, 0))
        ttk.Button(hdr, text="✓ Chọn tất cả", command=lambda: self._toggle_all(True)).pack(side="left", padx=(6, 0))
        ttk.Button(hdr, text="✗ Bỏ tất cả", command=lambda: self._toggle_all(False)).pack(side="left", padx=(6, 0))
        # danh sách model có chiều cao cố định + scroll (nhiều model không đẩy nút Bắt đầu khỏi màn hình)
        wrap = tk.Frame(f_m, bg=self.C["log_bg"], highlightthickness=1, highlightbackground=self.C["border"])
        wrap.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        mcanvas = tk.Canvas(wrap, bg=self.C["log_bg"], height=150, highlightthickness=0)
        msb = ttk.Scrollbar(wrap, orient="vertical", command=mcanvas.yview)
        mcanvas.configure(yscrollcommand=msb.set)
        msb.pack(side="right", fill="y"); mcanvas.pack(side="left", fill="both", expand=True)
        self.model_box = tk.Frame(mcanvas, bg=self.C["log_bg"])
        mcanvas.create_window((0, 0), window=self.model_box, anchor="nw", tags="inner")
        self.model_box.bind("<Configure>", lambda e: mcanvas.configure(scrollregion=mcanvas.bbox("all")))
        mcanvas.bind("<Configure>", lambda e: mcanvas.itemconfigure("inner", width=e.width))
        mcanvas.bind("<Enter>", lambda e: mcanvas.bind_all(
            "<MouseWheel>", lambda ev: mcanvas.yview_scroll(-ev.delta // 120, "units")))
        mcanvas.bind("<Leave>", lambda e: mcanvas.unbind_all("<MouseWheel>"))
        self._refresh_models()

        f_s = self._card(r, "Cài đặt"); f_s.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        for c in (1, 3, 5, 7): f_s.columnconfigure(c, weight=1)
        self.conf = tk.DoubleVar(value=0.55); self.dilate = tk.IntVar(value=4); self.feather = tk.IntVar(value=4)
        self.min_r = tk.DoubleVar(value=0.0); self.max_r = tk.DoubleVar(value=1.0)
        self.imgsz = tk.IntVar(value=768); self.batch = tk.IntVar(value=16)
        self._spin(f_s, "Confidence", self.conf, 0, 0, 0.05, 1.0, 0.05, "%.2f")
        self._spin(f_s, "Batch", self.batch, 0, 2, 1, 64, 1)
        self._spin(f_s, "Imgsz", self.imgsz, 0, 4, 320, 1536, 64)
        self._spin(f_s, "Dilation", self.dilate, 1, 0, 0, 50, 1)
        self._spin(f_s, "Feather", self.feather, 1, 2, 0, 50, 1)
        self._spin(f_s, "Min ratio", self.min_r, 1, 4, 0.0, 1.0, 0.01, "%.2f")
        self._spin(f_s, "Max ratio", self.max_r, 1, 6, 0.0, 1.0, 0.01, "%.2f")

        f_b = ttk.Frame(r); f_b.grid(row=3, column=0, sticky="ew", padx=12, pady=6)
        self.start_btn = ttk.Button(f_b, text="▶  Bắt đầu Blur", command=self._start, style="Accent.TButton")
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(f_b, text="⏹  Dừng", command=lambda: self.cancel.set(), state="disabled", style="Danger.TButton")
        self.cancel_btn.pack(side="left", padx=(6, 0))

        f_p = self._card(r, "Tiến trình"); f_p.grid(row=4, column=0, sticky="ew", padx=12, pady=6)
        self.prog = tk.DoubleVar(value=0)
        ttk.Progressbar(f_p, variable=self.prog, maximum=100).pack(fill="x", pady=(0, 6))
        self.status = tk.StringVar(value="Sẵn sàng")
        ttk.Label(f_p, textvariable=self.status, style="Muted.TLabel").pack(fill="x")

        f_l = self._card(r, "Log"); f_l.grid(row=5, column=0, sticky="nsew", padx=12, pady=(6, 12))
        self.log = self._log_widget(f_l)

    def _spin(self, parent, label, var, row, col, frm, to, inc, fmt=None):
        ttk.Label(parent, text=label + ":", style="Panel.TLabel").grid(row=row, column=col, sticky="w", pady=(6, 0))
        kw = dict(from_=frm, to=to, increment=inc, textvariable=var, width=7)
        if fmt: kw["format"] = fmt
        ttk.Spinbox(parent, **kw).grid(row=row, column=col + 1, sticky="ew", padx=(6, 20), pady=(6, 0))

    def _path_row(self, parent, var, cmd):
        row = ttk.Frame(parent, style="Panel.TFrame"); row.columnconfigure(0, weight=1)
        entry = ttk.Entry(row, textvariable=var)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        button = ttk.Button(row, text="Chọn...", command=cmd)
        button.grid(row=0, column=1)
        row.entry = entry
        row.button = button
        return row

    def _load_models(self, sel, log):
        """Load các model relpath trong sel (cache lại). Trả list YOLO. Raise nếu lỗi."""
        out = []
        for rel in sel:
            if rel not in self.models:
                log(f"  Tải {rel} ...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.models[rel] = _YOLO_CLASS(str(YOLO_MODEL_DIR / rel))
            out.append(self.models[rel])
        return out

    def _refresh_models(self):
        found = []
        if YOLO_MODEL_DIR.exists():
            found = [p for p in sorted(YOLO_MODEL_DIR.rglob("*.pt"))
                     if "New folder" not in p.parts]  # bỏ thư mục backup trùng lặp
        for w in self.model_box.winfo_children():
            w.destroy()
        if not found:
            ttk.Label(self.model_box, text=f"  Không có .pt trong {YOLO_MODEL_DIR}",
                      style="Muted.TLabel").pack(anchor="w", padx=8, pady=8)
            return
        old = {k: v.get() for k, v in self.model_vars.items()}
        self.model_vars.clear()
        for p in found:
            rel = p.relative_to(YOLO_MODEL_DIR).as_posix()
            var = tk.BooleanVar(value=old.get(rel, True))  # luôn bật tất cả yolo mặc định
            self.model_vars[rel] = var
            row = tk.Frame(self.model_box, bg=self.C["log_bg"])
            row.pack(fill="x", padx=8, pady=2)
            ttk.Checkbutton(row, text=rel, variable=var).pack(side="left")
            if rel in self.models:
                m = self.models[rel]
                ttk.Label(row, text=f"[{m.task} {list(m.names.values())}]", style="Muted.TLabel",
                          font=("Cascadia Mono", 8)).pack(side="left", padx=(8, 0))

    def _toggle_all(self, on):
        for v in self.model_vars.values():
            v.set(on)

    def _add_model(self):
        path = filedialog.askopenfilename(title="Chọn .pt", filetypes=[("YOLO", "*.pt"), ("All", "*.*")])
        if path:
            dst = YOLO_MODEL_DIR / Path(path).name
            if not dst.exists():
                try: shutil.copy2(path, dst)
                except Exception as e: self.q.put(("log", f"✗ copy lỗi: {e}"))
            self._refresh_models()

    def _pick_in(self):
        d = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if d:
            self.in_var.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="Chọn thư mục output")
        if d: self.out_var.set(d)

    def _sync_output_mode(self):
        overwrite = self.overwrite_var.get()
        state = "disabled" if overwrite else "normal"
        self.output_row.entry.configure(state=state)
        self.output_row.button.configure(state=state)
        if overwrite:
            self.out_var.set("")

    def _start(self):
        in_dir = self.in_var.get().strip()
        if not in_dir or not Path(in_dir).is_dir():
            self.status.set("⚠ Chưa chọn thư mục đầu vào hợp lệ!"); return
        sel = [k for k, v in self.model_vars.items() if v.get()]
        if not sel:
            self.status.set("⚠ Chưa chọn model!"); return
        if _YOLO_CLASS is None:
            self.status.set("⚠ Chưa cài ultralytics!"); return
        self.cancel.clear()
        self.start_btn.config(state="disabled"); self.cancel_btn.config(state="normal")
        self.prog.set(0)
        self.log.config(state="normal"); self.log.delete("1.0", "end"); self.log.config(state="disabled")
        threading.Thread(target=self._worker, args=(sel,), daemon=True).start()

    def _worker(self, sel):
        in_dir = Path(self.in_var.get().strip())
        out_str = self.out_var.get().strip()
        overwrite = self.overwrite_var.get()
        out_dir = in_dir if overwrite else (Path(out_str) if out_str else in_dir.parent / f"{in_dir.name}_blurred")
        params = {
            "conf": round(self.conf.get(), 2), "dilate": self.dilate.get(), "feather": self.feather.get(),
            "min_ratio": round(self.min_r.get(), 2), "max_ratio": round(self.max_r.get(), 2),
            "imgsz": int(self.imgsz.get()), "batch": max(1, int(self.batch.get())),
        }
        if not 0.0 <= params["min_ratio"] <= params["max_ratio"] <= 1.0:
            self.q.put(("log", "Lỗi: cần 0 <= min_ratio <= max_ratio <= 1")); self.q.put(("done",)); return
        pattern = "**/*" if self.recursive.get() else "*"
        images = sorted(f for f in in_dir.glob(pattern) if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            self.q.put(("log", "Không tìm thấy ảnh!")); self.q.put(("done",)); return

        try:
            models = self._load_models(sel, lambda s: self.q.put(("log", s)))
        except Exception as e:
            self.q.put(("log", f"  Lỗi tải model: {e}")); self.q.put(("done",)); return
        self.q.put(("refresh_models",))
        self.q.put(("log", f"Models : {', '.join(sel)}"))
        self.q.put(("log", f"Device : {self.device_name}  |  imgsz={params['imgsz']} batch={params['batch']} half={self.device!='cpu'}"))
        self.q.put(("log", f"Chế độ : {'Ghi đè trực tiếp ảnh gốc' if overwrite else 'Xuất ra thư mục riêng'}"))
        self.q.put(("log", f"Ảnh    : {len(images)}   Output: {out_dir}"))
        self.q.put(("log", "─" * 60))

        stats = blur_folder(
            images, in_dir, out_dir, models, params, self.device,
            log_cb=lambda s: self.q.put(("log", s)),
            progress_cb=lambda v: self.q.put(("prog", v)),
            status_cb=lambda s: self.q.put(("status", s)),
            cancel=self.cancel,
        )
        self.q.put(("log",
            f"\n{'═'*60}\n"
            f"Đã blur : {stats['blurred']} ảnh ({stats['regions']} vùng)\n"
            f"Sạch    : {stats['clean']} ảnh   Lỗi: {stats['error']}\n"
            f"Ghi file: {stats['written']} ảnh"
            + (f"   Bỏ qua ảnh sạch: {stats['skipped_clean']}\n" if overwrite else "\n")
            + f"Thời gian: {stats['wall']:.1f}s  ({stats['avg']*1000:.0f} ms/ảnh)\n"
            f"Output  : {out_dir}\n"
            f"→ Sang tab Review để soi & sửa tay vùng sót.\n{'═'*60}"))
        self.q.put(("set_review_folder", str(out_dir)))
        self.q.put(("status", f"Xong · {stats['avg']*1000:.0f} ms/ảnh · {stats['wall']:.1f}s"))
        self.q.put(("prog", 100)); self.q.put(("done",))

    # ==================== REVIEW TAB ====================
    def _build_review(self):
        r = self.t_review
        r.columnconfigure(0, weight=1); r.rowconfigure(1, weight=1)
        f_f = self._card(r, "Thư mục output (đã blur)"); f_f.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        f_f.columnconfigure(0, weight=1)
        self.rev_folder = tk.StringVar()
        self._path_row(f_f, self.rev_folder, self._rev_pick).grid(row=0, column=0, sticky="ew")
        bar = ttk.Frame(f_f, style="Panel.TFrame"); bar.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(bar, text="Tải lại", command=self._rev_load).pack(side="left")
        self.only_miss = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Chỉ ảnh nghi sót (0 vùng)", variable=self.only_miss,
                        command=self._rev_load).pack(side="left", padx=(10, 0))

        main = ttk.Frame(r); main.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        main.columnconfigure(0, weight=1, minsize=250); main.columnconfigure(1, weight=3); main.rowconfigure(0, weight=1)

        f_list = self._card(main, "Ảnh"); f_list.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.rev_list = tk.Listbox(f_list, activestyle="dotbox", font=("Cascadia Mono", 9), selectmode="browse",
                                   bg=self.C["log_bg"], fg=self.C["log_fg"], selectbackground=self.C["accent_d"],
                                   selectforeground="#fff", relief="flat", highlightthickness=1,
                                   highlightbackground=self.C["border"])
        sb = ttk.Scrollbar(f_list, command=self.rev_list.yview)
        self.rev_list.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.rev_list.pack(fill="both", expand=True)
        self.rev_list.bind("<<ListboxSelect>>", lambda e: self._rev_sel())

        f_r = self._card(main, "Kéo chuột để khoanh vùng sót → blur ngay"); f_r.grid(row=0, column=1, sticky="nsew")
        f_r.columnconfigure(0, weight=1); f_r.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(f_r, bg=self.C["log_bg"], highlightthickness=1, highlightbackground=self.C["border"], cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._draw_start)
        self.canvas.bind("<B1-Motion>", self._draw_move)
        self.canvas.bind("<ButtonRelease-1>", self._draw_end)
        self.canvas.bind("<Configure>", lambda e: self._render() if getattr(self, "rev_view", None) is not None else None)

        ctl = ttk.Frame(f_r, style="Panel.TFrame"); ctl.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(ctl, text="Class:", style="Panel.TLabel").pack(side="left")
        self.rev_label = tk.StringVar(value=LABEL_CHOICES[0])
        ttk.Combobox(ctl, textvariable=self.rev_label, values=LABEL_CHOICES, width=14).pack(side="left", padx=(4, 12))
        ttk.Button(ctl, text="↶ Undo khung", command=self._undo_box).pack(side="left")
        ttk.Button(ctl, text="◀", command=lambda: self._rev_step(-1), width=3).pack(side="left", padx=(12, 2))
        ttk.Button(ctl, text="▶", command=lambda: self._rev_step(1), width=3).pack(side="left")
        ttk.Button(ctl, text="Mở file", command=self._rev_open).pack(side="left", padx=(12, 0))

        self.rev_info = tk.StringVar(value="—")
        ttk.Label(f_r, textvariable=self.rev_info, style="Muted.TLabel", font=("Cascadia Mono", 9)).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.gif_ctl = ttk.Frame(f_r, style="Panel.TFrame")
        self.gif_ctl.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.gif_follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.gif_ctl, text="Áp dụng từ khung này đến cuối GIF",
                        variable=self.gif_follow).pack(side="left")
        ttk.Button(self.gif_ctl, text="◀ Khung", command=lambda: self._gif_step(-1)).pack(side="left", padx=(8, 2))
        ttk.Button(self.gif_ctl, text="Khung ▶", command=lambda: self._gif_step(1)).pack(side="left")
        self.gif_info = tk.StringVar()
        ttk.Label(self.gif_ctl, textvariable=self.gif_info, style="Muted.TLabel").pack(side="left", padx=8)
        self.gif_ctl.grid_remove()

        # review runtime state
        self.rev_images = []; self.rev_input_dir = None; self.rev_dets = {}
        self.rev_idx = -1; self.rev_photo = None
        self.rev_base = None      # ndarray output gốc phiên hiện tại
        self.rev_view = None      # ndarray đang hiển thị (đã blur các khung)
        self.rev_boxes = []       # list (x1,y1,x2,y2) trong toạ độ ảnh
        self.rev_scale = 1.0; self.rev_off = (0, 0)
        self._drag = None
        self.gif_frames = []; self.gif_durations = []; self.gif_loop = None
        self.gif_idx = 0; self.gif_boxes = []

    def _rev_pick(self):
        d = filedialog.askdirectory(title="Chọn thư mục output đã blur")
        if d: self.rev_folder.set(d); self._rev_load()

    def _rev_load(self):
        folder = Path(self.rev_folder.get().strip())
        if not folder.is_dir(): return
        self.rev_dets = {}; self.rev_input_dir = None
        mf = folder / MANIFEST_NAME
        if mf.exists():
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                self.rev_dets = data.get("detections", {})
                self.rev_input_dir = Path(data["input_dir"]) if data.get("input_dir") else None
            except Exception:
                pass
        imgs = sorted(f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTS | {".gif"})
        if self.only_miss.get():
            imgs = [f for f in imgs if f.suffix.lower() == ".gif" or len(self.rev_dets.get(f.relative_to(folder).as_posix(), [1])) == 0]
        self.rev_images = imgs
        self.rev_list.delete(0, "end")
        for f in imgs:
            n = len(self.rev_dets.get(f.relative_to(folder).as_posix(), []))
            self.rev_list.insert("end", f"{'· ' if n else '⚠ '}{f.name}  ({n})")
        self.rev_info.set(f"{len(imgs)} ảnh" + ("  ·  input: " + str(self.rev_input_dir) if self.rev_input_dir else "  ·  (không có manifest → không lưu label được)"))
        if imgs:
            self.rev_list.selection_clear(0, "end"); self.rev_list.selection_set(0); self._rev_show(0)
        else:
            self.rev_idx = -1; self.rev_view = None; self.gif_frames = []
            self._drag = None; self.canvas.delete("all"); self.gif_ctl.grid_remove()

    def _rev_sel(self):
        s = self.rev_list.curselection()
        if s: self._rev_show(s[0])

    def _orig_path(self, out_path):
        """Ảnh gốc tương ứng output (theo manifest input_dir + relpath)."""
        if not self.rev_input_dir: return None
        folder = Path(self.rev_folder.get().strip())
        cand = self.rev_input_dir / out_path.relative_to(folder)
        if cand.exists(): return cand
        for ext in IMAGE_EXTS:  # output có thể khác đuôi
            c = cand.with_suffix(ext)
            if c.exists(): return c
        return None

    def _rev_show(self, idx):
        if not (0 <= idx < len(self.rev_images)): return
        # lưu phiên cũ trước khi chuyển
        self._flush_view()
        self.rev_idx = idx
        p = self.rev_images[idx]
        self._drag = None; self.rev_view = None; self.rev_boxes = []
        self.gif_frames = []; self.gif_boxes = []; self.gif_idx = 0
        self.gif_ctl.grid_remove()
        if p.suffix.lower() == ".gif":
            try:
                self.gif_frames, self.gif_durations, self.gif_loop = _read_gif(p)
            except Exception as exc:
                self.canvas.delete("all")
                self.rev_info.set(f"Không đọc được GIF: {exc}")
                return
            self.gif_ctl.grid()
            self._gif_refresh()
            return
        img = _imread(p)
        if img is None:
            self.canvas.delete("all"); self.canvas.create_text(20, 20, anchor="nw", text="Không đọc được ảnh", fill="red"); return
        self.rev_base = img.copy(); self.rev_view = img.copy(); self.rev_boxes = []
        self._render()
        orig = self._orig_path(p)
        self.rev_info.set(f"{p.name}   [{img.shape[1]}x{img.shape[0]}]   gốc: {'✓' if orig else '✗ (không lưu label được)'}")

    def _gif_refresh(self):
        self.rev_view = _gif_view(self.gif_frames[self.gif_idx], self.gif_boxes, self.gif_idx)
        self._render()
        self.gif_info.set(f"{self.gif_idx + 1}/{len(self.gif_frames)}")
        self.rev_info.set(f"{self.rev_images[self.rev_idx].name}   GIF · {len(self.gif_boxes)} vùng")

    def _gif_step(self, delta):
        if self.gif_frames and 0 <= self.gif_idx + delta < len(self.gif_frames):
            self._drag = None
            self.gif_idx += delta
            self._gif_refresh()

    def _render(self):
        import cv2
        from PIL import Image, ImageTk
        cw = max(self.canvas.winfo_width(), 100); ch = max(self.canvas.winfo_height(), 100)
        h, w = self.rev_view.shape[:2]
        self.rev_scale = min(cw / w, ch / h, 1.0)
        dw, dh = max(1, int(w * self.rev_scale)), max(1, int(h * self.rev_scale))
        disp = cv2.resize(self.rev_view, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.rev_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.rev_off = ((cw - dw) // 2, (ch - dh) // 2)
        self.canvas.delete("all")
        self.canvas.create_image(self.rev_off[0], self.rev_off[1], anchor="nw", image=self.rev_photo)

    def _canvas_to_img(self, cx, cy):
        x = (cx - self.rev_off[0]) / self.rev_scale
        y = (cy - self.rev_off[1]) / self.rev_scale
        return int(x), int(y)

    def _draw_start(self, e):
        if self.rev_view is None: return
        self._drag = (e.x, e.y)
        self._rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline=self.C["accent"], width=2)

    def _draw_move(self, e):
        if self._drag:
            self.canvas.coords(self._rect, self._drag[0], self._drag[1], e.x, e.y)

    def _draw_end(self, e):
        if not self._drag: return
        x0, y0 = self._drag; self._drag = None
        ix1, iy1 = self._canvas_to_img(min(x0, e.x), min(y0, e.y))
        ix2, iy2 = self._canvas_to_img(max(x0, e.x), max(y0, e.y))
        h, w = self.rev_view.shape[:2]
        ix1, iy1 = max(0, ix1), max(0, iy1); ix2, iy2 = min(w, ix2), min(h, iy2)
        if ix2 - ix1 < 3 or iy2 - iy1 < 3:
            self._render()
            return
        if self.gif_frames:
            end = len(self.gif_frames) - 1 if self.gif_follow.get() else self.gif_idx
            self.gif_boxes.append((self.gif_idx, end, (ix1, iy1, ix2, iy2)))
            try:
                self._save_view(self.rev_images[self.rev_idx])
            except Exception as exc:
                self.gif_boxes.pop()
                self._gif_refresh()
                self.rev_info.set(f"Lỗi lưu GIF: {exc}")
                return
            self._gif_refresh()
            self.rev_info.set(f"Đã lưu ✓ · vùng áp dụng cho khung {self.gif_idx + 1}–{end + 1}")
            return
        self.rev_boxes.append((ix1, iy1, ix2, iy2))
        _blur_region(self.rev_view, ix1, iy1, ix2, iy2)
        self._render()
        # lưu label (nếu có ảnh gốc) + lưu output ngay
        p = self.rev_images[self.rev_idx]
        orig = self._orig_path(p)
        if orig:
            save_labelme_box(orig, (ix1, iy1, ix2, iy2), self.rev_label.get().strip() or LABEL_CHOICES[0], w, h)
        self._save_view(p)
        self.rev_info.set(f"{p.name}   +1 khung ({len(self.rev_boxes)})   label={self.rev_label.get()}   {'lưu ✓' if orig else 'không có gốc → chỉ blur, KHÔNG lưu label'}")

    def _save_view(self, p):
        if self.gif_frames:
            _save_gif(p, self.gif_frames, self.gif_durations, self.gif_loop, self.gif_boxes)
            return
        import cv2
        cv2.imwrite(str(p), self.rev_view)

    def _flush_view(self):
        # đã lưu ngay khi vẽ; hàm này chừa chỗ nếu cần đồng bộ thêm
        pass

    def _undo_box(self):
        if self.gif_frames:
            if not self.gif_boxes: return
            box = self.gif_boxes.pop()
            try:
                self._save_view(self.rev_images[self.rev_idx])
            except Exception as exc:
                self.gif_boxes.append(box)
                self.rev_info.set(f"Lỗi lưu GIF: {exc}")
                return
            self._gif_refresh()
            return
        if self.rev_idx < 0 or not self.rev_boxes:
            return
        self.rev_boxes.pop()
        self.rev_view = self.rev_base.copy()
        for b in self.rev_boxes:
            _blur_region(self.rev_view, *b)
        self._render()
        p = self.rev_images[self.rev_idx]
        orig = self._orig_path(p)
        if orig:
            pop_labelme_box(orig)
        self._save_view(p)
        self.rev_info.set(f"{p.name}   undo → còn {len(self.rev_boxes)} khung")

    def _rev_step(self, d):
        i = self.rev_idx + d
        if 0 <= i < len(self.rev_images):
            self.rev_list.selection_clear(0, "end"); self.rev_list.selection_set(i); self.rev_list.see(i)
            self._rev_show(i)

    def _rev_open(self):
        if 0 <= self.rev_idx < len(self.rev_images):
            os.startfile(str(self.rev_images[self.rev_idx]))

    # ==================== TRAIN TAB ====================
    def _build_train(self):
        r = self.t_train
        r.columnconfigure(0, weight=1); r.rowconfigure(4, weight=1)
        f_d = self._card(r, "Data (LabelMe JSON + ảnh). Mặc định gồm train-collect/ vẽ tay.")
        f_d.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6)); f_d.columnconfigure(0, weight=1)
        self.data_var = tk.StringVar(value=str(TRAIN_COLLECT_DIR))
        self._path_row(f_d, self.data_var, self._pick_data).grid(row=0, column=0, sticky="ew")
        bar = ttk.Frame(f_d, style="Panel.TFrame"); bar.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.seed_btn = ttk.Button(bar, text="🤖 Seed nhãn từ ensemble", command=self._seed)
        self.seed_btn.pack(side="left")
        ttk.Button(bar, text="Mở AnyLabeling", command=self._open_anylabel).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Mở train-collect", command=lambda: self._open(TRAIN_COLLECT_DIR)).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Mở model-yolo", command=lambda: self._open(YOLO_MODEL_DIR)).pack(side="left", padx=(6, 0))
        ttk.Label(f_d, text="Seed = dùng model + conf/imgsz/batch/ratio ở tab Blur để auto-label 1 folder ảnh vào train-collect. "
                            "Sau đó sang Review sửa vùng sót rồi Prepare + Train.", style="Muted.TLabel",
                  wraplength=880, justify="left").grid(row=2, column=0, sticky="w", pady=(6, 0))

        f_p = self._card(r, "Tham số"); f_p.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        for c in (1, 3, 5): f_p.columnconfigure(c, weight=1)
        ttk.Label(f_p, text="Base:", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.tr_base = tk.StringVar(value="yolo11l.pt")  # GPU 17GB -> train được l/x, mạnh hơn nano nhiều
        ttk.Combobox(f_p, textvariable=self.tr_base, width=14,
                     values=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
                             "yolo11m-seg.pt", "yolo11l-seg.pt"]).grid(row=0, column=1, sticky="ew", padx=(6, 20))
        ttk.Label(f_p, text="Tên output:", style="Panel.TLabel").grid(row=0, column=2, sticky="w")
        self.tr_name = tk.StringVar(value="my_nsfw_model")
        ttk.Entry(f_p, textvariable=self.tr_name, width=18).grid(row=0, column=3, sticky="ew", padx=(6, 20))
        self.tr_epochs = tk.IntVar(value=100); self.tr_batch = tk.IntVar(value=16); self.tr_imgsz = tk.IntVar(value=768)
        self._spin(f_p, "Epochs", self.tr_epochs, 1, 0, 1, 1000, 1)
        self._spin(f_p, "Batch", self.tr_batch, 1, 2, 1, 64, 1)
        self._spin(f_p, "Imgsz", self.tr_imgsz, 1, 4, 320, 1280, 32)

        f_b = ttk.Frame(r); f_b.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        self.prep_btn = ttk.Button(f_b, text="⚙ Prepare Dataset", command=self._prepare)
        self.prep_btn.pack(side="left")
        self.train_btn = ttk.Button(f_b, text="▶ Train", command=self._train, style="Accent.TButton")
        self.train_btn.pack(side="left", padx=(6, 0))

        f_pr = self._card(r, "Tiến trình"); f_pr.grid(row=3, column=0, sticky="ew", padx=12, pady=6)
        self.tr_prog = tk.DoubleVar(value=0)
        ttk.Progressbar(f_pr, variable=self.tr_prog, maximum=100).pack(fill="x", pady=(0, 6))
        self.tr_status = tk.StringVar(value="Sẵn sàng")
        ttk.Label(f_pr, textvariable=self.tr_status, style="Muted.TLabel").pack(fill="x")
        f_l = self._card(r, "Log train"); f_l.grid(row=4, column=0, sticky="nsew", padx=12, pady=(6, 12))
        self.tr_log = self._log_widget(f_l)

    def _pick_data(self):
        d = filedialog.askdirectory(title="Chọn thư mục data")
        if d: self.data_var.set(d)

    def _seed(self):
        sel = [k for k, v in self.model_vars.items() if v.get()]
        if not sel:
            self.q.put(("tlog", "⚠ Chưa chọn model nào ở tab Blur")); return
        if _YOLO_CLASS is None:
            self.q.put(("tlog", "⚠ Chưa cài ultralytics")); return
        folder = filedialog.askdirectory(title="Chọn folder ẢNH GỐC để auto-label")
        if not folder:
            return
        self.cancel.clear()
        self.seed_btn.config(state="disabled")
        self.tr_log.config(state="normal"); self.tr_log.delete("1.0", "end"); self.tr_log.config(state="disabled")
        threading.Thread(target=self._seed_worker, args=(sel, Path(folder)), daemon=True).start()

    def _seed_worker(self, sel, folder):
        images = sorted(f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            self.q.put(("tlog", "Không tìm thấy ảnh!")); self.q.put(("seed_done",)); return
        try:
            models = self._load_models(sel, lambda s: self.q.put(("tlog", s)))
        except Exception as e:
            self.q.put(("tlog", f"Lỗi tải model: {e}")); self.q.put(("seed_done",)); return
        self.q.put(("refresh_models",))
        self.q.put(("tlog", f"Ensemble: {', '.join(sel)}"))
        self.q.put(("tlog", f"Auto-label {len(images)} ảnh → {TRAIN_COLLECT_DIR}"))
        self.q.put(("tstatus", "Đang seed..."))
        n_img, n_box = seed_labels(
            images, models, round(self.conf.get(), 2), int(self.imgsz.get()),
            max(1, int(self.batch.get())), self.device,
            round(self.min_r.get(), 2), round(self.max_r.get(), 2),
            log_cb=lambda s: self.q.put(("tlog", s)),
            progress_cb=lambda v: self.q.put(("tprog", v)),
            cancel=self.cancel,
        )
        self.q.put(("tlog", f"\n✓ Seed xong: {n_img} ảnh có nhãn, {n_box} box → {TRAIN_COLLECT_DIR}"))
        self.q.put(("tlog", "→ Sang Review (mở train-collect) sửa vùng sót, rồi Prepare + Train."))
        self.q.put(("tstatus", f"✓ Seed: {n_img} ảnh / {n_box} box"))
        self.q.put(("tprog", 100)); self.q.put(("seed_done",))

    def _open(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        try: os.startfile(str(path))
        except Exception as e: self.q.put(("tlog", f"✗ {e}"))

    def _open_anylabel(self):
        exe = BASE_DIR / "AnyLabeling-Windows-CPU-x64.exe"
        if exe.exists():
            import subprocess
            try: subprocess.Popen([str(exe)], cwd=str(BASE_DIR))
            except Exception as e: self.q.put(("tlog", f"✗ {e}"))
        else:
            self.q.put(("tlog", "✗ Không tìm thấy AnyLabeling exe"))

    def _prepare(self):
        data = Path(self.data_var.get().strip())
        if not data.is_dir():
            self.q.put(("tlog", f"✗ Data không hợp lệ: {data}")); return
        self.prep_btn.config(state="disabled")
        self.q.put(("tlog", f"⚙ Prepare từ: {data}"))

        def run():
            try:
                import prepare_dataset as pd
                pd.DATA_DIRS = [data]; pd.OUT_DIR = DATASET_DIR
                # bắt print của prepare vào log
                import io
                buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
                try:
                    yaml = pd.prepare(verbose=True)
                finally:
                    sys.stdout = old
                for line in buf.getvalue().splitlines():
                    self.q.put(("tlog", "  " + line))
                self.q.put(("tlog", f"✓ Dataset: {yaml}" if yaml else "✗ Không có cặp json+ảnh"))
            except Exception as e:
                self.q.put(("tlog", f"✗ Lỗi: {e}"))
            finally:
                self.q.put(("prep_done",))
        threading.Thread(target=run, daemon=True).start()

    def _train(self):
        if _YOLO_CLASS is None:
            self.tr_status.set("✗ Chưa cài ultralytics"); return
        yaml = DATASET_DIR / "dataset.yaml"
        if not yaml.exists():
            self.q.put(("tlog", "✗ Chưa có dataset.yaml → bấm Prepare trước")); return
        self.train_btn.config(state="disabled"); self.prep_btn.config(state="disabled")
        self.tr_prog.set(0)
        self.tr_log.config(state="normal"); self.tr_log.delete("1.0", "end"); self.tr_log.config(state="disabled")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        yaml = DATASET_DIR / "dataset.yaml"
        base = self.tr_base.get().strip(); name = self.tr_name.get().strip() or "my_nsfw_model"
        epochs = self.tr_epochs.get(); batch = self.tr_batch.get(); imgsz = self.tr_imgsz.get()
        self.q.put(("tlog", f"Base {base} · epochs={epochs} batch={batch} imgsz={imgsz} · device={self.device_name}"))
        self.q.put(("tstatus", "Đang train..."))
        try:
            import io
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _YOLO_CLASS(base)
            runs = BASE_DIR / "runs" / "train"

            class Cap(io.TextIOBase):
                def __init__(s, cb): s.cb = cb
                def write(s, x):
                    x = x.rstrip("\n")
                    if x.strip(): s.cb(x)
                    return len(x)
                def flush(s): pass
            old = sys.stdout
            sys.stdout = Cap(lambda x: self.q.put(("tlog", x)))
            try:
                model.train(data=str(yaml), epochs=epochs, batch=batch, imgsz=imgsz,
                            device=self.device, project=str(runs), name=name, exist_ok=True, verbose=True)
            finally:
                sys.stdout = old
            best = runs / name / "weights" / "best.pt"
            if best.exists():
                YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                dst = YOLO_MODEL_DIR / f"{name}.pt"
                shutil.copy2(best, dst)
                self.q.put(("tlog", f"\n✓ Lưu model: {dst}  → Refresh ở tab Blur để dùng"))
                self.q.put(("tstatus", f"✓ Xong: model-yolo/{name}.pt"))
            else:
                self.q.put(("tlog", f"✗ Không tìm thấy {best}")); self.q.put(("tstatus", "✗ Không có best.pt"))
        except Exception as e:
            self.q.put(("tlog", f"✗ Lỗi train: {e}")); self.q.put(("tstatus", f"✗ {e}"))
        self.q.put(("tprog", 100)); self.q.put(("train_done",))

    # ==================== QUEUE POLL ====================
    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait(); k = m[0]
                if k == "log":
                    self._append(self.log, m[1])
                elif k == "prog": self.prog.set(m[1])
                elif k == "status": self.status.set(m[1])
                elif k == "refresh_models": self._refresh_models()
                elif k == "set_review_folder": self.rev_folder.set(m[1])
                elif k == "done":
                    self.start_btn.config(state="normal"); self.cancel_btn.config(state="disabled")
                elif k == "tlog": self._append(self.tr_log, m[1])
                elif k == "tstatus": self.tr_status.set(m[1])
                elif k == "tprog": self.tr_prog.set(m[1])
                elif k == "prep_done": self.prep_btn.config(state="normal")
                elif k == "seed_done": self.seed_btn.config(state="normal")
                elif k == "train_done":
                    self.train_btn.config(state="normal"); self.prep_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _append(self, widget, text):
        widget.config(state="normal"); widget.insert("end", text + "\n"); widget.see("end"); widget.config(state="disabled")


# ─────────────────────────── SELF-TEST ───────────────────────────────

def _selftest():
    import numpy as np, cv2, tempfile
    d = Path(tempfile.mkdtemp())
    orig = d / "x.jpg"
    img = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(orig), img)

    # blur_region thực sự đổi pixel
    before = _imread(orig); after = before.copy()
    _blur_region(after, 10, 10, 80, 80)
    assert not np.array_equal(before[10:80, 10:80], after[10:80, 10:80]), "blur không đổi pixel"
    assert np.array_equal(before[0:5, 0:5], after[0:5, 0:5]), "blur tràn ra ngoài vùng"

    # labelme roundtrip: save 2 box rồi pop
    global TRAIN_COLLECT_DIR
    TRAIN_COLLECT_DIR = d / "collect"
    save_labelme_box(orig, (5, 6, 40, 50), "pussy", 160, 120)
    save_labelme_box(orig, (7, 8, 30, 33), "penis", 160, 120)
    js = TRAIN_COLLECT_DIR / "x.json"
    data = json.loads(js.read_text(encoding="utf-8"))
    assert len(data["shapes"]) == 2 and data["imageWidth"] == 160, "labelme sai"
    assert data["shapes"][0]["shape_type"] == "rectangle"
    pop_labelme_box(orig)
    assert len(json.loads(js.read_text())["shapes"]) == 1, "pop sai"
    pop_labelme_box(orig)
    assert not js.exists(), "file phải bị xoá khi hết shape"

    # chunk
    assert list(_chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    # class normalization + overwrite writer
    assert _norm_label("Pussy") == "pussy" and _norm_label("vagina_exposed") == "pussy"
    assert _norm_label("BUTTOCKS") == "buttocks"  # fallback lowercase
    write_labelme_boxes(orig, [("pussy", 1, 2, 3, 4), ("penis", 5, 6, 7, 8)], 160, 120)
    d2 = json.loads((TRAIN_COLLECT_DIR / "x.json").read_text())
    assert len(d2["shapes"]) == 2, "overwrite writer sai"
    write_labelme_boxes(orig, [("nipple", 1, 2, 3, 4)], 160, 120)
    assert len(json.loads((TRAIN_COLLECT_DIR / "x.json").read_text())["shapes"]) == 1, "phải ghi đè, không append"

    # blur_folder ghi đè trực tiếp folder: ảnh có detection bị ghi, ảnh sạch không bị re-save.
    work = d / "overwrite_case"
    work.mkdir()
    hit = work / "hit.png"
    clean = work / "clean.png"
    hit_before = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
    clean_before = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(hit), hit_before)
    cv2.imwrite(str(clean), clean_before)

    class FakeBox:
        xyxy = np.array([[10, 10, 60, 60]])
        cls = np.array([0])
        conf = np.array([0.95])

    class FakePred:
        names = {0: "penis"}
        masks = None

        def __init__(self, detected):
            self.boxes = [FakeBox()] if detected else []

    class FakeModel:
        def predict(self, arrs, **kwargs):
            return [FakePred(i == 0) for i in range(len(arrs))]

    written = []
    real_imwrite = cv2.imwrite
    def track_imwrite(path, im):
        written.append(Path(path).name)
        return real_imwrite(path, im)
    cv2.imwrite = track_imwrite
    try:
        stats = blur_folder(
            [hit, clean], work, work, [FakeModel()],
            {"conf": 0.5, "dilate": 0, "feather": 0, "min_ratio": 0.0, "max_ratio": 1.0, "imgsz": 80, "batch": 2},
            "cpu", log_cb=lambda s: None,
        )
    finally:
        cv2.imwrite = real_imwrite
    assert stats["written"] == 1 and stats["skipped_clean"] == 1, "ghi đè folder sai thống kê"
    assert written == ["hit.png"], "không được ghi lại ảnh sạch khi overwrite input"
    assert not np.array_equal(_imread(hit), hit_before), "ảnh có detection phải bị ghi đè"
    mf = json.loads((work / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert mf["overwrite"] is True and mf["output_dir"] == str(work), "manifest overwrite sai"

    # GIF: vùng từ khung được chọn tới cuối, giữ timing/loop và undo được.
    from PIL import Image
    rng = np.random.default_rng(42)
    gif_path = d / "animation.gif"
    frames = [Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)) for _ in range(3)]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=[40, 80, 120], loop=2)
    frames, durations, loop = _read_gif(gif_path)
    boxes = [(1, 2, (10, 10, 40, 35))]
    for i, frame in enumerate(frames):
        before = _gif_view(frame, [], i)
        after = _gif_view(frame, boxes, i)
        assert np.array_equal(before[:10], after[:10]), "GIF blur tràn vùng"
        assert np.array_equal(before, after) == (i == 0), "GIF sai phạm vi khung"
    single_box = [(1, 1, (10, 10, 40, 35))]
    assert np.array_equal(_gif_view(frames[2], single_box, 2), _gif_view(frames[2], [], 2))
    _save_gif(gif_path, frames, durations, loop, boxes)
    saved, timing, repeats = _read_gif(gif_path)
    assert len(saved) == 3 and timing == [40, 80, 120] and repeats == 2, "GIF mất animation/timing/loop"
    assert np.array_equal(np.array(saved[0]), np.array(frames[0])), "GIF đổi khung trước vùng chọn"
    assert not np.array_equal(np.array(saved[2]), np.array(frames[2])), "GIF chưa blur khung cuối"
    _save_gif(gif_path, frames, durations, loop, [])
    restored, _, _ = _read_gif(gif_path)
    assert all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(frames, restored)), "GIF undo sai"
    transparent = [f.copy() for f in frames]
    for frame in transparent:
        frame.putalpha(Image.fromarray(np.where(np.indices((48, 64))[1] < 5, 0, 255).astype(np.uint8)))
    _save_gif(gif_path, transparent, durations, None, boxes)
    saved, _, repeats = _read_gif(gif_path)
    assert repeats is None and all(f.getpixel((0, 0))[3] == 0 for f in saved), "GIF mất transparency hoặc tự lặp"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        root = tk.Tk()
        App(root)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
