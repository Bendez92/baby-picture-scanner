"""Tkinter desktop application for finding likely baby pictures."""

from __future__ import annotations

import os
import json
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .detector import BabyDetector, Detection, image_paths


class BabyPictureScanner(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self._configure_dark_theme()
        self.title("Baby Picture Scanner")
        self.geometry("980x650")
        self.minsize(760, 500)
        self.configure(bg="#202124")
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            try:
                self.iconbitmap(str(icon_path))
            except tk.TclError:
                pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Return>", lambda _event: self._start_scan())
        self._closing = False
        self._ui_queue: queue.Queue[tuple[object, tuple]] = queue.Queue()
        self._scan_active = False
        self._config_path = Path.home() / ".baby_picture_scanner.json"
        self._config = self._load_config()
        self._detector = BabyDetector()
        self._cancel_event = threading.Event()
        self._scan_thread: threading.Thread | None = None
        self._detections: list[Detection] = []
        self._selected: set[int] = set()
        self._thumb: ImageTk.PhotoImage | None = None
        self._preview_image: Image.Image | None = None
        self._preview_geometry: tuple[int, int, int, int, int, int] | None = None
        self._crop_start: tuple[int, int] | None = None
        self._crop_selection: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._crop_rect: int | None = None
        self._model_ready = False
        self._model_loaded = False
        self._download_thread: threading.Thread | None = None
        self._scanned_count = 0
        self._build_start()
        self._queue_after_id = self.after(50, self._drain_ui_queue)
        self._prepare_after_id = self.after(50, self._prepare_model)

    def _load_config(self) -> dict:
        try:
            with self._config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            return config if isinstance(config, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_config(self) -> None:
        try:
            threshold = max(1, min(99, float(self.threshold_var.get()))) if hasattr(self, "threshold_var") else 50
        except (TypeError, ValueError):
            threshold = 50
        config = {
            "folder": self.folder_var.get() if hasattr(self, "folder_var") else "",
            "recursive": bool(self.recursive_var.get()) if hasattr(self, "recursive_var") else True,
            "mode": self.mode_var.get() if hasattr(self, "mode_var") else "Kids 8 and under",
            "threshold": threshold,
        }
        self._config = config
        try:
            self._config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _post_ui(self, callback, *args) -> None:
        if self._closing:
            return
        self._ui_queue.put((callback, args))

    def _drain_ui_queue(self) -> None:
        if self._closing:
            return
        while True:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
                if getattr(callback, "__name__", "") == "_scan_progress":
                    self.update_idletasks()
                    break
            except tk.TclError:
                return
            except Exception as error:
                if not self._closing:
                    messagebox.showerror("Unexpected error", str(error))
        self._queue_after_id = self.after(50, self._drain_ui_queue)

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self._cancel_event.set()
            self._save_config()
            for after_id in (
                getattr(self, "_queue_after_id", None),
                getattr(self, "_prepare_after_id", None),
            ):
                if after_id:
                    try:
                        self.after_cancel(after_id)
                    except Exception:
                        pass
            self.destroy()
            current = threading.current_thread()
            for worker in (self._scan_thread, self._download_thread):
                if worker is not None and worker is not current and worker.is_alive():
                    worker.join(0.25)
        finally:
            os._exit(0)

    def _configure_dark_theme(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#202124", foreground="#f1f3f4")
        style.configure("TFrame", background="#202124")
        style.configure("TLabel", background="#202124", foreground="#f1f3f4")
        style.configure("TLabelframe", background="#202124", foreground="#f1f3f4")
        style.configure("TLabelframe.Label", background="#202124", foreground="#f1f3f4")
        style.configure("TButton", background="#3c4043", foreground="#f1f3f4", padding=(10, 6))
        style.map("TButton", background=[("active", "#5f6368"), ("disabled", "#303134")])
        style.configure("TCheckbutton", background="#202124", foreground="#f1f3f4")
        style.configure("TEntry", fieldbackground="#303134", foreground="#f1f3f4")
        style.configure("TSpinbox", fieldbackground="#303134", foreground="#f1f3f4")
        style.configure("TCombobox", fieldbackground="#303134", foreground="#f1f3f4")
        style.map("TCombobox", fieldbackground=[("readonly", "#303134")], foreground=[("readonly", "#f1f3f4")])
        style.configure("TProgressbar", background="#8ab4f8", troughcolor="#303134", bordercolor="#202124")
        style.configure("Treeview", background="#303134", fieldbackground="#303134", foreground="#f1f3f4", rowheight=28)
        style.configure("Treeview.Heading", background="#3c4043", foreground="#f1f3f4")
        style.map("Treeview", background=[("selected", "#3c5a80")], foreground=[("selected", "#ffffff")])
        style.configure("TPanedwindow", background="#202124")

    def _build_start(self) -> None:
        self._clear()
        outer = ttk.Frame(self, padding=35)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Baby Picture Scanner", font=("TkDefaultFont", 22, "bold")).pack(pady=(35, 6))
        ttk.Label(outer, text="Find likely baby pictures in a folder using a local age-classification model.").pack(pady=(0, 28))
        card = ttk.LabelFrame(outer, text="Scan location", padding=18)
        card.pack(fill="x", padx=80)
        self.folder_var = tk.StringVar(value=self._config.get("folder", ""))
        ttk.Entry(card, textvariable=self.folder_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(card, text="Browse…", command=self._browse).pack(side="left")
        options = ttk.Frame(outer)
        options.pack(pady=18)
        self.recursive_var = tk.BooleanVar(value=self._config.get("recursive", True))
        ttk.Checkbutton(options, text="Include subfolders", variable=self.recursive_var).pack(side="left", padx=12)
        ttk.Label(options, text="Scan for:").pack(side="left", padx=(8, 4))
        mode = self._config.get("mode", "Kids 8 and under")
        self.mode_var = tk.StringVar(value=mode if mode in ("Kids 8 and under", "Babies (0-2)") else "Kids 8 and under")
        self.mode_combo = ttk.Combobox(
            options,
            textvariable=self.mode_var,
            values=("Kids 8 and under", "Babies (0-2)"),
            state="readonly",
            width=18,
        )
        self.mode_combo.pack(side="left", padx=(0, 12))
        ttk.Label(options, text="Match confidence threshold:").pack(side="left")
        try:
            threshold = max(1, min(99, float(self._config.get("threshold", 50))))
        except (TypeError, ValueError):
            threshold = 50
        self.threshold_var = tk.DoubleVar(value=threshold)
        ttk.Spinbox(options, from_=1, to=99, increment=1, width=5, textvariable=self.threshold_var).pack(side="left", padx=5)
        ttk.Label(options, text="%").pack(side="left")
        self.start_status = ttk.Label(outer, text="Choose a folder to begin.")
        self.start_status.pack(pady=14)
        self.start_progress = ttk.Progressbar(outer, mode="determinate", length=420)
        self.start_progress.pack(pady=6)
        self.retry_button = ttk.Button(outer, text="Retry model download", command=self._prepare_model)
        buttons = ttk.Frame(outer)
        buttons.pack(pady=10)
        self.scan_button = ttk.Button(buttons, text="Scan", command=self._start_scan)
        self.scan_button.pack(side="left", padx=5)
        if not self._model_loaded:
            self.scan_button.configure(state="disabled")
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel_scan, state="disabled")
        self.cancel_button.pack(side="left", padx=5)

    def _clear(self) -> None:
        for child in self.winfo_children():
            child.destroy()

    def _browse(self) -> None:
        initial = self.folder_var.get() if Path(self.folder_var.get()).is_dir() else str(Path.home())
        selected = filedialog.askdirectory(title="Choose a folder to scan", initialdir=initial)
        if selected:
            self.folder_var.set(selected)

    def _prepare_model(self) -> None:
        if self._download_thread and self._download_thread.is_alive():
            return
        self.retry_button.pack_forget()
        self.scan_button.configure(state="disabled")
        self.start_progress.configure(mode="determinate", maximum=1, value=0)
        self.start_status.configure(text="Checking local model cache…")
        self._download_thread = threading.Thread(target=self._model_worker, daemon=True)
        self._download_thread.start()

    def _model_worker(self) -> None:
        def progress(downloaded: int, total: int) -> None:
            self._post_ui(self._download_progress, downloaded, total)

        try:
            self._detector.prepare_model(progress)
            self._post_ui(self._model_loading_ui)
            self._detector.load_model()
        except Exception as error:
            self._post_ui(self._model_failed, error)
        else:
            self._post_ui(self._model_ready_ui)

    def _download_progress(self, downloaded: int, total: int) -> None:
        total = max(total, 1)
        self.start_progress.configure(maximum=total, value=min(downloaded, total))
        self.start_status.configure(
            text=f"Downloading age model… {downloaded / 1024**2:.1f} / {total / 1024**2:.1f} MB"
        )

    def _model_ready_ui(self) -> None:
        self._model_ready = True
        self._model_loaded = True
        self.start_progress.stop()
        self.start_progress.configure(mode="determinate", maximum=1, value=1)
        self.start_status.configure(text="Model ready. Choose a folder to begin.")
        self.scan_button.configure(state="normal")

    def _model_loading_ui(self) -> None:
        self.start_progress.configure(mode="indeterminate")
        self.start_progress.start(12)
        self.start_status.configure(text="Loading model into memory…")

    def _model_failed(self, error: Exception) -> None:
        self._model_ready = False
        self._model_loaded = False
        self.start_progress.stop()
        self.start_progress.configure(mode="determinate", maximum=1, value=0)
        self.start_status.configure(
            text=f"Model download/load failed. Internet access is required on first run: {error}"
        )
        self.retry_button.pack(pady=(8, 0))

    def _start_scan(self) -> None:
        if self._scan_active:
            return
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.is_dir():
            messagebox.showerror("Folder required", "Please choose an existing folder.")
            return
        paths = image_paths(folder, self.recursive_var.get())
        if not paths:
            messagebox.showinfo("No images", "No supported images were found in that folder.")
            return
        try:
            threshold = max(0.01, min(0.99, float(self.threshold_var.get()) / 100))
        except (TypeError, ValueError):
            messagebox.showerror("Invalid threshold", "Enter a confidence threshold from 1 to 99.")
            return
        self._cancel_event.clear()
        self._save_config()
        self._scan_active = True
        self.scan_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.start_progress.stop()
        self.start_progress.configure(mode="determinate", maximum=len(paths), value=0)
        self.start_status.configure(text=f"Scanning 0/{len(paths)} images")
        mode = "baby" if self.mode_var.get() == "Babies (0-2)" else "kids"
        self._scan_thread = threading.Thread(target=self._scan_worker, args=(paths, threshold, mode), daemon=True)
        self._scan_thread.start()

    def _scan_worker(self, paths: list[Path], threshold: float, mode: str) -> None:
        def progress(done: int, total: int, path: Path | None) -> None:
            self._post_ui(self._scan_progress, done, total, path)

        try:
            results = self._detector.scan(paths, threshold, self._cancel_event, progress, mode)
        except Exception as error:
            self._post_ui(self._scan_failed, error)
            return
        self._post_ui(self._scan_finished, results, len(paths))

    def _scan_progress(self, done: int, total: int, path: Path | None = None) -> None:
        self._scanned_count = done
        self.start_progress.configure(value=done)
        self.start_status.configure(text=f"Scanning {done}/{total} images")

    def _cancel_scan(self) -> None:
        self._cancel_event.set()
        self.start_status.configure(text="Cancelling after the current image…")
        self.cancel_button.configure(state="disabled")

    def _scan_failed(self, error: Exception) -> None:
        self._scan_active = False
        self.scan_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        messagebox.showerror("Scan failed", f"The scan could not be completed:\n{error}")

    def _scan_finished(self, results: list[Detection], total: int) -> None:
        self._scan_active = False
        if self._cancel_event.is_set():
            self._save_config()
            self._build_start()
            self.start_status.configure(text="Scan cancelled.")
            return
        self._scanned_count = total
        self._detections = results
        self._selected = set(range(len(results)))
        self._build_results()

    def _build_results(self) -> None:
        self._clear()
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(
            top,
            text=f"Results — {len(self._detections)} flagged of {self._scanned_count} scanned",
            font=("TkDefaultFont", 15, "bold"),
        ).pack(side="left")
        ttk.Button(top, text="New Scan", command=self._new_scan).pack(side="right")
        body = ttk.PanedWindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        body.add(left, weight=3)
        columns = ("selected", "filename", "path", "confidence")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="extended")
        headings = {"selected": "✓", "filename": "Filename", "path": "Path", "confidence": "Confidence"}
        widths = {"selected": 42, "filename": 180, "path": 350, "confidence": 95}
        for column in columns:
            self.tree.heading(column, text=headings[column], command=lambda c=column: self._sort_results(c))
            self.tree.column(column, width=widths[column], anchor="w")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._show_preview)
        self.tree.bind("<Button-1>", self._toggle_mark)
        self.tree.bind("<Double-1>", lambda _event: self._open_selected())
        self.tree.bind("<Delete>", lambda _event: self._delete_selected())
        for index, detection in enumerate(self._detections):
            self.tree.insert("", "end", iid=str(index), values=("✓", detection.path.name, str(detection.path), f"{detection.confidence:.1%}"))
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Mark All", command=self._mark_all).pack(side="left", padx=(0, 5))
        ttk.Button(actions, text="Unmark All", command=self._unmark_all).pack(side="left", padx=5)
        ttk.Button(actions, text="Reveal in file manager", command=self._reveal_selected).pack(side="right")
        ttk.Button(actions, text="Delete Selected", command=self._delete_selected).pack(side="right", padx=5)
        preview = ttk.LabelFrame(body, text="Preview (drag to crop)", padding=12)
        body.add(preview, weight=1)
        self.preview_canvas = tk.Canvas(
            preview, width=360, height=450, bg="#303134", highlightthickness=0
        )
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<ButtonPress-1>", self._crop_press)
        self.preview_canvas.bind("<B1-Motion>", self._crop_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._crop_release)
        crop_actions = ttk.Frame(preview)
        crop_actions.pack(fill="x", pady=(8, 0))
        self.crop_button = ttk.Button(
            crop_actions, text="Crop & Overwrite Original", command=self._crop_overwrite, state="disabled"
        )
        self.crop_button.pack(side="left")
        ttk.Button(crop_actions, text="Reset", command=self._reset_crop).pack(side="right")
        self.preview_canvas.create_text(180, 225, text="Select a row to preview", fill="#f1f3f4")
        if self._detections:
            self.tree.selection_set("0")

    def _mark_all(self) -> None:
        self._selected = set(range(len(self._detections)))
        self._refresh_marks()

    def _new_scan(self) -> None:
        self._save_config()
        self._build_start()

    def _sort_results(self, column: str, descending: bool | None = None) -> None:
        if not self._detections:
            return
        selected_paths = [self._detections[int(item)].path for item in self.tree.selection()]
        marked_paths = {d.path for index, d in enumerate(self._detections) if index in self._selected}
        current = getattr(self, "_sort_state", ("filename", False))
        descending = (not current[1]) if descending is None and current[0] == column else bool(descending)
        self._sort_state = (column, descending)
        key = {
            "filename": lambda d: d.path.name.lower(),
            "path": lambda d: str(d.path).lower(),
            "confidence": lambda d: d.confidence,
            "selected": lambda d: d.path.name.lower(),
        }[column]
        self._detections.sort(key=key, reverse=descending)
        self._selected = {index for index, detection in enumerate(self._detections) if detection.path in marked_paths}
        for index, detection in enumerate(self._detections):
            self.tree.item(
                str(index),
                values=("✓" if index in self._selected else "", detection.path.name, str(detection.path), f"{detection.confidence:.1%}"),
            )
        self.tree.selection_set(
            [str(index) for index, detection in enumerate(self._detections) if detection.path in selected_paths]
        )
        self._show_preview()

    def _toggle_mark(self, event) -> str | None:
        region = self.tree.identify("region", event.x, event.y)
        column = self.tree.identify_column(event.x)
        item = self.tree.identify_row(event.y)
        if region == "cell" and column == "#1" and item:
            index = int(item)
            if index in self._selected:
                self._selected.remove(index)
            else:
                self._selected.add(index)
            self._refresh_marks()
            return "break"
        return None

    def _unmark_all(self) -> None:
        self._selected.clear()
        self._refresh_marks()

    def _refresh_marks(self) -> None:
        for index in range(len(self._detections)):
            values = list(self.tree.item(str(index), "values"))
            values[0] = "✓" if index in self._selected else ""
            self.tree.item(str(index), values=values)

    def _show_preview(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        index = int(selected[0])
        path = self._detections[index].path
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            self._preview_image = image
            self._reset_crop()
            self._draw_preview()
        except Exception as error:
            self._preview_image = None
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(180, 225, text=f"Preview unavailable\n{error}", fill="#f1f3f4")

    def _draw_preview(self) -> None:
        if self._preview_image is None:
            return
        self.preview_canvas.delete("all")
        image = self._preview_image.copy()
        image.thumbnail((350, 430))
        width, height = image.size
        canvas_width = max(self.preview_canvas.winfo_width(), 360)
        canvas_height = max(self.preview_canvas.winfo_height(), 450)
        origin_x = (canvas_width - width) // 2
        origin_y = (canvas_height - height) // 2
        self._preview_geometry = (
            self._preview_image.width,
            self._preview_image.height,
            width,
            height,
            origin_x,
            origin_y,
        )
        self._thumb = ImageTk.PhotoImage(image)
        self.preview_canvas.create_image(origin_x, origin_y, image=self._thumb, anchor="nw")

    def _crop_press(self, event) -> None:
        if self._preview_geometry is None:
            return
        self._reset_crop()
        self._crop_start = self._preview_point(event.x, event.y)

    def _crop_drag(self, event) -> None:
        if self._crop_start is None:
            return
        point = self._preview_point(event.x, event.y)
        if point is None:
            return
        if self._crop_rect is not None:
            self.preview_canvas.delete(self._crop_rect)
        self._crop_rect = self.preview_canvas.create_rectangle(
            self._crop_start[0], self._crop_start[1], point[0], point[1],
            outline="#8ab4f8", width=2,
        )
        self.crop_button.configure(state="normal" if self._valid_crop(self._crop_start, point) else "disabled")
        self._crop_selection = (self._crop_start, point)

    def _crop_release(self, event) -> None:
        self._crop_drag(event)
        self._crop_start = None

    def _preview_point(self, x: int, y: int) -> tuple[int, int] | None:
        if self._preview_geometry is None:
            return None
        _, _, width, height, origin_x, origin_y = self._preview_geometry
        px = min(max(x, origin_x), origin_x + width)
        py = min(max(y, origin_y), origin_y + height)
        return px, py

    @staticmethod
    def _valid_crop(start: tuple[int, int], end: tuple[int, int]) -> bool:
        return abs(end[0] - start[0]) >= 3 and abs(end[1] - start[1]) >= 3

    def _reset_crop(self) -> None:
        if hasattr(self, "preview_canvas") and self._crop_rect is not None:
            self.preview_canvas.delete(self._crop_rect)
        self._crop_rect = None
        self._crop_start = None
        self._crop_selection = None
        if hasattr(self, "crop_button"):
            self.crop_button.configure(state="disabled")

    def _crop_overwrite(self) -> None:
        if self._preview_geometry is None or self._crop_selection is None:
            return
        start, point = self._crop_selection
        if not self._valid_crop(start, point):
            messagebox.showinfo("Select a crop", "Drag a rectangle over the preview first.")
            return
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Select an image", "Select an image first.")
            return
        index = int(selected[0])
        path = self._detections[index].path
        original_width, original_height, display_width, display_height, origin_x, origin_y = self._preview_geometry
        left, right = sorted((start[0], point[0]))
        top, bottom = sorted((start[1], point[1]))
        box = (
            int((left - origin_x) * original_width / display_width),
            int((top - origin_y) * original_height / display_height),
            int((right - origin_x) * original_width / display_width),
            int((bottom - origin_y) * original_height / display_height),
        )
        if not messagebox.askyesno("Overwrite original", f"Crop and overwrite {path.name}?"):
            return
        try:
            with Image.open(path) as source:
                image_format = source.format or path.suffix.lstrip(".").upper()
                if image_format.upper() == "JPG":
                    image_format = "JPEG"
                cropped = source.crop(box)
                if image_format.upper() in {"JPEG", "BMP"} and cropped.mode not in {"RGB", "L"}:
                    cropped = cropped.convert("RGB")
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    cropped.save(temporary_path, format=image_format)
                    os.replace(temporary_path, path)
                finally:
                    temporary_path.unlink(missing_ok=True)
            self._show_preview()
            messagebox.showinfo("Crop complete", f"Updated {path.name}.")
        except Exception as error:
            messagebox.showerror("Could not overwrite image", str(error))

    def _reveal_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Select an image", "Select an image first.")
            return
        path = self._detections[int(selected[0])].path
        try:
            self._open_path(path.parent)
        except OSError as error:
            messagebox.showerror("Could not reveal file", str(error))

    def _open_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        path = self._detections[int(selected[0])].path
        try:
            self._open_path(path)
        except OSError as error:
            messagebox.showerror("Could not open image", str(error))

    @staticmethod
    def _open_path(path: Path) -> None:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _delete_selected(self) -> None:
        selected = sorted(self._selected)
        if not selected:
            messagebox.showinfo("Nothing selected", "Mark images for deletion first.")
            return
        if not messagebox.askyesno("Delete selected", f"Move {len(selected)} image(s) to the trash?"):
            return
        try:
            from send2trash import send2trash
        except ImportError:
            send2trash = None
        failures = []
        deleted = set()
        for index in selected:
            path = self._detections[index].path
            try:
                if send2trash:
                    send2trash(str(path))
                    deleted.add(index)
                else:
                    if messagebox.askyesno("Trash unavailable", "send2trash is not installed. Permanently delete instead?"):
                        os.remove(path)
                        deleted.add(index)
                    else:
                        continue
            except OSError:
                failures.append(path.name)
        self._detections = [d for i, d in enumerate(self._detections) if i not in deleted]
        self._selected = set(range(len(self._detections)))
        self._build_results()
        if failures:
            messagebox.showwarning("Some files could not be deleted", "\n".join(failures))


def main() -> None:
    app = BabyPictureScanner()
    try:
        app.mainloop()
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
