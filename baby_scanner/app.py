"""Tkinter desktop application for finding likely baby pictures."""

from __future__ import annotations

import os
import subprocess
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
        self._detector = BabyDetector()
        self._cancel_event = threading.Event()
        self._scan_thread: threading.Thread | None = None
        self._detections: list[Detection] = []
        self._selected: set[int] = set()
        self._thumb: ImageTk.PhotoImage | None = None
        self._model_ready = False
        self._model_loaded = False
        self._download_thread: threading.Thread | None = None
        self._build_start()
        self.after(50, self._prepare_model)

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
        self.folder_var = tk.StringVar()
        ttk.Entry(card, textvariable=self.folder_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(card, text="Browse…", command=self._browse).pack(side="left")
        options = ttk.Frame(outer)
        options.pack(pady=18)
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Include subfolders", variable=self.recursive_var).pack(side="left", padx=12)
        ttk.Label(options, text="Scan for:").pack(side="left", padx=(8, 4))
        self.mode_var = tk.StringVar(value="Kids 8 and under")
        self.mode_combo = ttk.Combobox(
            options,
            textvariable=self.mode_var,
            values=("Kids 8 and under", "Babies (0-2)"),
            state="readonly",
            width=18,
        )
        self.mode_combo.pack(side="left", padx=(0, 12))
        ttk.Label(options, text="Match confidence threshold:").pack(side="left")
        self.threshold_var = tk.DoubleVar(value=50)
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
        selected = filedialog.askdirectory(title="Choose a folder to scan")
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
            self.after(0, lambda: self._download_progress(downloaded, total))

        try:
            self._detector.prepare_model(progress)
            self.after(0, self._model_loading_ui)
            self._detector.load_model()
        except Exception as error:
            self.after(0, lambda error=error: self._model_failed(error))
        else:
            self.after(0, self._model_ready_ui)

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
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.is_dir():
            messagebox.showerror("Folder required", "Please choose an existing folder.")
            return
        paths = image_paths(folder, self.recursive_var.get())
        if not paths:
            messagebox.showinfo("No images", "No supported images were found in that folder.")
            return
        self._cancel_event.clear()
        self.scan_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.start_progress.configure(maximum=len(paths), value=0)
        self.start_status.configure(text=f"Scanning {len(paths)} image(s)…")
        threshold = max(0.01, min(0.99, float(self.threshold_var.get()) / 100))
        mode = "baby" if self.mode_var.get() == "Babies (0-2)" else "kids"
        self._scan_thread = threading.Thread(target=self._scan_worker, args=(paths, threshold, mode), daemon=True)
        self._scan_thread.start()

    def _scan_worker(self, paths: list[Path], threshold: float, mode: str) -> None:
        def progress(done: int, total: int) -> None:
            self.after(0, lambda: self._scan_progress(done, total))

        results = self._detector.scan(paths, threshold, self._cancel_event, progress, mode)
        self.after(0, lambda: self._scan_finished(results))

    def _scan_progress(self, done: int, total: int) -> None:
        self.start_progress.configure(value=done)
        self.start_status.configure(text=f"Scanning image {done} of {total}…")

    def _cancel_scan(self) -> None:
        self._cancel_event.set()
        self.start_status.configure(text="Cancelling after the current image…")
        self.cancel_button.configure(state="disabled")

    def _scan_finished(self, results: list[Detection]) -> None:
        if self._cancel_event.is_set():
            self._build_start()
            self.start_status.configure(text="Scan cancelled.")
            return
        self._detections = results
        self._selected = set(range(len(results)))
        self._build_results()

    def _build_results(self) -> None:
        self._clear()
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text=f"Results — {len(self._detections)} likely match(es)", font=("TkDefaultFont", 15, "bold")).pack(side="left")
        ttk.Button(top, text="New Scan", command=self._build_start).pack(side="right")
        body = ttk.PanedWindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        body.add(left, weight=3)
        columns = ("selected", "filename", "path", "confidence")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="extended")
        headings = {"selected": "✓", "filename": "Filename", "path": "Path", "confidence": "Confidence"}
        widths = {"selected": 42, "filename": 180, "path": 350, "confidence": 95}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._show_preview)
        self.tree.bind("<Button-1>", self._toggle_mark)
        self.tree.bind("<Double-1>", lambda _event: self._reveal_selected())
        for index, detection in enumerate(self._detections):
            self.tree.insert("", "end", iid=str(index), values=("✓", detection.path.name, str(detection.path), f"{detection.confidence:.1%}"))
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Mark All", command=self._mark_all).pack(side="left", padx=(0, 5))
        ttk.Button(actions, text="Unmark All", command=self._unmark_all).pack(side="left", padx=5)
        ttk.Button(actions, text="Reveal in file manager", command=self._reveal_selected).pack(side="right")
        ttk.Button(actions, text="Delete Selected", command=self._delete_selected).pack(side="right", padx=5)
        preview = ttk.LabelFrame(body, text="Preview", padding=12)
        body.add(preview, weight=1)
        self.preview_label = ttk.Label(preview, text="Select a row to preview", anchor="center")
        self.preview_label.pack(fill="both", expand=True)
        if self._detections:
            self.tree.selection_set("0")

    def _mark_all(self) -> None:
        self._selected = set(range(len(self._detections)))
        self._refresh_marks()

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
            image = Image.open(path).convert("RGB")
            image.thumbnail((330, 450))
            self._thumb = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self._thumb, text="")
        except Exception as error:
            self.preview_label.configure(image="", text=f"Preview unavailable\n{error}")

    def _reveal_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Select an image", "Select an image first.")
            return
        path = self._detections[int(selected[0])].path
        try:
            subprocess.run(["xdg-open", str(path.parent)], check=False)
        except OSError as error:
            messagebox.showerror("Could not reveal file", str(error))

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
        for index in selected:
            path = self._detections[index].path
            try:
                if send2trash:
                    send2trash(str(path))
                else:
                    if messagebox.askyesno("Trash unavailable", "send2trash is not installed. Permanently delete instead?"):
                        os.remove(path)
                    else:
                        continue
            except OSError:
                failures.append(path.name)
        self._detections = [d for i, d in enumerate(self._detections) if i not in self._selected]
        self._selected = set(range(len(self._detections)))
        self._build_results()
        if failures:
            messagebox.showwarning("Some files could not be deleted", "\n".join(failures))


def main() -> None:
    app = BabyPictureScanner()
    app.mainloop()


if __name__ == "__main__":
    main()
