"""Local image classification for baby-picture discovery."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL = "nateraw/vit-age-classifier"


@dataclass
class Detection:
    path: Path
    confidence: float
    label: str


class BabyDetector:
    """Lazy-loading HuggingFace age classifier.

    The model labels used by nateraw/vit-age-classifier are age ranges. The
    model's 0-2 and 3-9 age ranges can be selected independently or
    combined as the app's approximate "kids 8 and under" mode.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._pipeline = None
        self._pipeline_lock = Lock()

    def _load(self):
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    import torch
                    from transformers import pipeline

                    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                    self._pipeline = pipeline("image-classification", model=self.model_name)
        return self._pipeline

    def load_model(self) -> None:
        """Load the classifier into memory once after its files are cached."""
        self._load()

    def prepare_model(self, on_progress: Callable[[int, int], None] | None = None) -> None:
        """Ensure model files are cached before the UI allows scanning."""
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm

        try:
            snapshot_download(self.model_name, local_files_only=True)
            if on_progress:
                on_progress(1, 1)
            return
        except Exception:
            pass

        dry_run = snapshot_download(self.model_name, dry_run=True)
        total = sum(info.file_size or 0 for info in dry_run)
        state = {"downloaded": sum(info.file_size or 0 for info in dry_run if info.is_cached)}
        lock = Lock()
        callback = on_progress or (lambda _downloaded, _total: None)

        class ProgressTqdm(tqdm):
            def update(self, n=1):
                previous = self.n
                result = super().update(n)
                delta = self.n - previous
                if delta:
                    with lock:
                        state["downloaded"] += delta
                        callback(state["downloaded"], total)
                return result

        snapshot_download(self.model_name, tqdm_class=ProgressTqdm)
        if on_progress:
            on_progress(total, total)

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.lower().replace(" ", "")

    @classmethod
    def _is_baby_label(cls, label: str) -> bool:
        return cls._normalize_label(label) in {
            "0-2", "0–2", "1-2", "1–2", "0to2", "1to2",
            "0-2years", "0–2years", "1-2years", "1–2years",
        }

    @classmethod
    def _is_toddler_label(cls, label: str) -> bool:
        return cls._normalize_label(label) in {"3-9", "3–9", "3to9", "3-9years", "3–9years"}

    @classmethod
    def _score_predictions(cls, predictions, mode: str) -> tuple[float, str]:
        scores = {cls._normalize_label(item["label"]): float(item["score"]) for item in predictions}
        baby = sum(score for label, score in scores.items() if cls._is_baby_label(label))
        if mode == "kids":
            toddler = sum(score for label, score in scores.items() if cls._is_toddler_label(label))
            return baby + toddler, "0-2 + 3-9"
        return baby, "0-2"

    def _classify_inputs(self, image_inputs: list, mode: str = "kids") -> list[tuple[float, str]]:
        classifier = self._load()
        predictions = classifier(image_inputs, top_k=None, batch_size=min(16, len(image_inputs)))
        if len(image_inputs) == 1 and predictions and isinstance(predictions[0], dict):
            predictions = [predictions]
        return [self._score_predictions(items, mode) for items in predictions]

    def _classify_input(self, image_input, mode: str = "kids") -> tuple[float, str]:
        return self._classify_inputs([image_input], mode)[0]

    @staticmethod
    def _face_crops(image: Image.Image) -> list[Image.Image]:
        """Return face crops with context around each detected face."""
        import cv2
        import numpy as np

        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        # A slightly more permissive detector catches small faces in photos
        # and paintings. The age classifier filters remaining false positives.
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(32, 32))
        height, width = rgb.shape[:2]
        crops = []
        for x, y, face_width, face_height in faces:
            # Evaluate both a tight crop and a contextual crop. The tight
            # crop helps paintings and small faces where surrounding content
            # can dominate the age estimate.
            for margin_x_ratio, margin_y_ratio in ((0.0, 0.0), (0.5, 0.6)):
                margin_x = int(face_width * margin_x_ratio)
                margin_y = int(face_height * margin_y_ratio)
                left = max(0, x - margin_x)
                top = max(0, y - margin_y)
                right = min(width, x + face_width + margin_x)
                bottom = min(height, y + face_height + margin_y)
                crops.append(Image.fromarray(rgb[top:bottom, left:right]))
        return crops

    def _prepare_image(self, image_path: Path) -> tuple[Path, Image.Image, list[Image.Image]]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        return image_path, image, self._face_crops(image)

    def classify(self, image_path: Path, mode: str = "kids") -> tuple[float, str]:
        """Classify the strongest face or whole-image candidate."""
        _, image, crops = self._prepare_image(image_path)
        predictions = self._classify_inputs(crops + [image], mode)
        return max(predictions, key=lambda prediction: prediction[0])

    def scan(
        self,
        paths: Iterable[Path],
        threshold: float = 0.50,
        cancel_event=None,
        on_progress: Callable[[int, int, Path | None], None] | None = None,
        mode: str = "kids",
    ) -> list[Detection]:
        image_paths = list(paths)
        results: list[Detection] = []
        prepared = []
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            futures = [executor.submit(self._prepare_image, path) for path in image_paths]
            for future in futures:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    prepared.append(future.result())
                except Exception:
                    prepared.append(None)

        face_inputs = []
        face_owners = []
        for item in prepared:
            if item is None:
                continue
            path, image, crops = item
            if crops:
                face_inputs.extend(crops)
                face_owners.extend([path] * len(crops))

        face_scores: dict[Path, tuple[float, str]] = {}
        if face_inputs:
            for owner, score in zip(face_owners, self._classify_inputs(face_inputs, mode)):
                if score[0] > face_scores.get(owner, (0.0, ""))[0]:
                    face_scores[owner] = score

        whole_images = []
        for item in prepared:
            if item is None:
                continue
            path, image, crops = item
            whole_images.append((path, image))
        whole_scores = self._classify_inputs([image for _, image in whole_images], mode) if whole_images else []
        whole_by_path = dict(zip((path for path, _ in whole_images), whole_scores))

        for index, item in enumerate(prepared, 1):
            if cancel_event is not None and cancel_event.is_set():
                break
            if item is None:
                if on_progress:
                    on_progress(index, len(image_paths), None)
                continue
            path, _image, crops = item
            candidates = [face_scores.get(path, (0.0, "unknown")), whole_by_path.get(path, (0.0, "unknown"))]
            confidence, label = max(candidates, key=lambda prediction: prediction[0])
            if confidence >= threshold:
                results.append(Detection(path, confidence, label))
            if on_progress:
                on_progress(index, len(image_paths), path)
        return results


def image_paths(folder: Path, recursive: bool = False) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
