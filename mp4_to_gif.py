"""mp4 -> gif có UI chọn vùng làm mờ (tkinter + cv2 + PIL, cùng stack với nsfw_tool.py).

Chạy:  python mp4_to_gif.py          (mở UI)
Tự kiểm: python mp4_to_gif.py --selftest

UI: Chọn thư mục -> click 1 video -> KÉO khung lên vùng cần mờ (kéo nhiều khung được) -> Chuyển GIF.
GIF lưu cạnh file mp4. Vùng mờ cố định trên mọi frame (không tracking).
"""

import sys
import tempfile
import threading
import zipfile
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# cùng thông số với nsfw_tool.py
GAUSSIAN_RADIUS = 4
MOSAIC_CELL = 9
CANVAS_W, CANVAS_H = 760, 460
MAX_FRAME_BYTES = 8 * 1024 * 1024   # giới hạn mỗi frame của pixiv ugoira


def blur_region(frame, box):
    """Gaussian + mosaic trong box=(x0,y0,x1,y1) theo tỉ lệ 0..1 của frame (in-place)."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return
    k = GAUSSIAN_RADIUS * 2 + 1
    roi = cv2.GaussianBlur(roi, (k, k), 0)
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (max(1, rw // MOSAIC_CELL), max(1, rh // MOSAIC_CELL)), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def _iter_source(src):
    """Trả (src_fps, iterator BGR). GIF đọc bằng PIL (cv2 đọc GIF không ổn định)."""
    if Path(src).suffix.lower() == ".gif":
        im = Image.open(src)
        fps = 1000 / (im.info.get("duration") or 100)   # ponytail: lấy delay frame đầu cho cả GIF

        def gen_gif():
            for k in range(getattr(im, "n_frames", 1)):
                im.seek(k)
                yield cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            im.close()

        return fps, gen_gif()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"không mở được {src}")

    def gen_cap():
        ok, bgr = cap.read()
        while ok:
            yield bgr
            ok, bgr = cap.read()
        cap.release()

    return cap.get(cv2.CAP_PROP_FPS), gen_cap()


def read_frames(src, fps, width):
    """Trả (frames, fps_thực). fps_thực = fps_gốc/step để tốc độ phát đúng."""
    src_fps, it = _iter_source(src)
    src_fps = src_fps or fps
    step = max(1, round(src_fps / fps))   # bỏ bớt frame để đạt ~fps
    frames = []
    for i, bgr in enumerate(it):
        if i % step == 0:
            h, w = bgr.shape[:2]
            if width and w > width:
                bgr = cv2.resize(bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            frames.append(bgr)
    if not frames:
        raise IOError(f"video rỗng: {src}")
    return frames, src_fps / step


def _jpeg_under_limit(frame, limit=MAX_FRAME_BYTES):
    for q in range(95, 20, -10):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        if len(buf) <= limit:
            return buf.tobytes()
    raise IOError("frame vẫn >8MB, hãy giảm 'Rộng (px)'")


def convert(src, dst, fps, width, boxes, ugoira=False):
    """dst là .gif, hoặc .zip (pixiv ugoira: 000000.jpg, 000001.jpg… mỗi frame ≤8MB, ≥2 frame)."""
    frames, fps = read_frames(src, fps, width)
    if len(frames) < 2:
        raise IOError("cần ≥2 frame (video quá ngắn hoặc FPS quá thấp)")
    for fr in frames:
        for b in boxes:
            blur_region(fr, b)
    if ugoira:
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:   # jpg đã nén, khỏi nén lại
            for i, fr in enumerate(frames):
                z.writestr(f"{i:06d}.jpg", _jpeg_under_limit(fr))
        return len(frames), fps
    # ponytail: giữ hết frame trong RAM; video rất dài thì stream bằng imageio.
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    imgs[0].save(dst, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(frames), fps


class App:
    def __init__(self):
        self.root = r = tk.Tk()
        r.title("MP4 -> GIF (làm mờ vùng chọn)")
        self.files, self.boxes, self.cur = [], {}, None   # boxes: {Path: [(x0,y0,x1,y1) tỉ lệ]}
        self.fps, self.width = tk.IntVar(value=12), tk.IntVar(value=480)
        self.same = tk.BooleanVar(value=False)
        self.ugoira = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Chọn thư mục chứa mp4.")

        left = ttk.Frame(r, padding=8); left.pack(side="left", fill="y")
        ttk.Button(left, text="Chọn thư mục…", command=self.pick_dir).pack(fill="x")
        self.lb = tk.Listbox(left, width=34, exportselection=False)
        self.lb.pack(fill="y", expand=True, pady=6)
        self.lb.bind("<<ListboxSelect>>", self.on_select)

        right = ttk.Frame(r, padding=8); right.pack(side="left", fill="both", expand=True)
        self.cv = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H, bg="#222", highlightthickness=0)
        self.cv.pack()
        self.cv.bind("<ButtonPress-1>", self.down)
        self.cv.bind("<B1-Motion>", self.drag)
        self.cv.bind("<ButtonRelease-1>", self.up)

        opt = ttk.Frame(right); opt.pack(fill="x", pady=6)
        ttk.Label(opt, text="FPS").pack(side="left")
        ttk.Spinbox(opt, from_=1, to=60, width=4, textvariable=self.fps).pack(side="left", padx=(2, 10))
        ttk.Label(opt, text="Rộng (px, 0=gốc)").pack(side="left")
        ttk.Spinbox(opt, from_=0, to=4000, increment=40, width=6, textvariable=self.width).pack(side="left", padx=(2, 10))
        ttk.Checkbutton(opt, text="Dùng vùng này cho mọi video", variable=self.same).pack(side="left")
        ttk.Checkbutton(opt, text="Xuất ZIP ugoira (pixiv)", variable=self.ugoira).pack(side="left", padx=10)

        btn = ttk.Frame(right); btn.pack(fill="x")
        ttk.Button(btn, text="Xoá vùng", command=self.clear).pack(side="left")
        ttk.Button(btn, text="Bỏ vùng cuối", command=self.undo).pack(side="left", padx=6)
        self.go = ttk.Button(btn, text="Chuyển GIF", command=self.run)
        self.go.pack(side="right")
        ttk.Label(right, textvariable=self.status).pack(fill="x", pady=(6, 0))

    # ---- chọn file / vẽ ----
    def pick_dir(self):
        d = filedialog.askdirectory(title="Chọn thư mục chứa mp4")
        if not d:
            return
        self.files = sorted(f for f in Path(d).iterdir() if f.suffix.lower() in (".mp4", ".gif"))
        self.boxes.clear()
        self.lb.delete(0, "end")
        for f in self.files:
            self.lb.insert("end", f.name)
        self.status.set(f"{len(self.files)} video." if self.files else "Không có .mp4/.gif trong thư mục.")
        if self.files:
            self.lb.selection_set(0)
            self.on_select()

    def on_select(self, _=None):
        sel = self.lb.curselection()
        if not sel:
            return
        self.cur = self.files[sel[0]]
        cap = cv2.VideoCapture(str(self.cur))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            self.status.set(f"Không đọc được {self.cur.name}")
            return
        h, w = bgr.shape[:2]
        s = min(CANVAS_W / w, CANVAS_H / h)
        self.dw, self.dh = int(w * s), int(h * s)
        im = Image.fromarray(cv2.cvtColor(cv2.resize(bgr, (self.dw, self.dh)), cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(im)   # giữ tham chiếu
        self.redraw()

    def redraw(self):
        self.cv.delete("all")
        self.cv.create_image(0, 0, image=self.photo, anchor="nw")
        for x0, y0, x1, y1 in self.boxes.get(self.cur, []):
            self.cv.create_rectangle(x0 * self.dw, y0 * self.dh, x1 * self.dw, y1 * self.dh,
                                     outline="#ff3355", width=2, fill="#ff3355", stipple="gray25")

    def _clamp(self, e):
        return min(max(e.x, 0), self.dw), min(max(e.y, 0), self.dh)

    def down(self, e):
        if self.cur:
            self.p0, self.tmp = self._clamp(e), None

    def drag(self, e):
        if not self.cur:
            return
        self.cv.delete(self.tmp)
        x, y = self._clamp(e)
        self.tmp = self.cv.create_rectangle(*self.p0, x, y, outline="#ffcc00", width=2)

    def up(self, e):
        if not self.cur:
            return
        x, y = self._clamp(e)
        (ax, ay) = self.p0
        if abs(x - ax) > 4 and abs(y - ay) > 4:
            self.boxes.setdefault(self.cur, []).append(
                (min(ax, x) / self.dw, min(ay, y) / self.dh, max(ax, x) / self.dw, max(ay, y) / self.dh))
        self.redraw()

    def clear(self):
        self.boxes.pop(self.cur, None)
        if self.cur:
            self.redraw()

    def undo(self):
        if self.boxes.get(self.cur):
            self.boxes[self.cur].pop()
            self.redraw()

    # ---- chuyển ----
    def run(self):
        if not self.files:
            self.status.set("Chưa chọn thư mục.")
            return
        fps, width = max(1, self.fps.get()), max(0, self.width.get())
        shared = self.boxes.get(self.cur, []) if self.same.get() else None
        jobs = [(f, shared if shared is not None else self.boxes.get(f, [])) for f in self.files]
        self.go.state(["disabled"])
        threading.Thread(target=self._work, args=(jobs, fps, width, self.ugoira.get()), daemon=True).start()

    def _work(self, jobs, fps, width, ugoira):
        say = lambda t: self.root.after(0, self.status.set, t)
        ext = ".zip" if ugoira else ".gif"
        err, last, eff = 0, "", fps
        for i, (f, boxes) in enumerate(jobs, 1):
            say(f"[{i}/{len(jobs)}] {f.name}…")
            try:
                dst = f.with_suffix(ext) if f.suffix.lower() != ext else f.with_name(f.stem + "_out" + ext)  # không ghi đè nguồn
                _, eff = convert(f, dst, fps, width, boxes, ugoira)
            except Exception as e:
                err += 1
                last = f" | LỖI {f.name}: {e}"
        hint = f" Delay mỗi frame trên pixiv: {round(1000 / eff)} ms." if ugoira else ""
        say(f"Xong {len(jobs) - err}/{len(jobs)} {ext} (lưu cạnh file gốc).{hint}{last}")
        self.root.after(0, lambda: self.go.state(["!disabled"]))


def _selftest():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    g = f.copy()
    blur_region(g, (0.5, 0.5, 1.0, 1.0))
    assert (g[:24] == f[:24]).all() and (g[:, :32] == f[:, :32]).all(), "ngoài vùng bị đổi"
    assert not (g[24:, 32:] == f[24:, 32:]).all(), "trong vùng không mờ"
    with tempfile.TemporaryDirectory() as d:
        mp4, gif = Path(d) / "t.mp4", Path(d) / "t.gif"
        vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
        for k in range(24):
            vw.write(np.full((48, 64, 3), k * 10, np.uint8))
        vw.release()
        n, eff = convert(mp4, gif, 12, 32, [(0, 0, 0.5, 0.5)])
        assert eff == 12, eff
        with Image.open(gif) as im:   # đóng file, nếu không Windows không xoá được thư mục tạm
            assert n == 12 and im.n_frames == 12 and im.size == (32, 24), (n, im.n_frames, im.size)
        z = Path(d) / "t.zip"
        convert(mp4, z, 12, 32, [], ugoira=True)
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        assert len(names) == 12 and names[0] == "000000.jpg", names
        g2 = Path(d) / "from_gif.zip"   # nguồn GIF -> ugoira
        assert convert(gif, g2, 12, 0, [], ugoira=True)[0] == 12
        try:
            convert(mp4, Path(d) / "x.gif", 1, 32, [])   # 1fps -> 1 frame
            assert False, "phải báo lỗi <2 frame"
        except IOError:
            pass
    print("OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        App().root.mainloop()
