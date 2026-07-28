"""Local image classification for baby-picture discovery."""

from __future__ import annotations

from dataclasses import dataclass
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

    The model labels used by nateraw/vit-age-classifier are age ranges. A
    range whose lower bound is 0 or 1 and upper bound is no greater than 2 is
    treated as a baby range.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline("image-classification", model=self.model_name)
        return self._pipeline

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
    def _is_baby_label(label: str) -> bool:
        normalized = label.lower().replace(" ", "")
        return normalized in {
            "0-2", "0–2", "1-2", "1–2", "0to2", "1to2",
            "0-2years", "0–2years", "1-2years", "1–2years",
        }

    def _classify_input(self, image_input) -> tuple[float, str]:
        classifier = self._load()
        predictions = classifier(image_input, top_k=None)
        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            predictions = predictions[0]
        baby = next((item for item in predictions if self._is_baby_label(item["label"])), None)
        if baby is None:
            return 0.0, str(predictions[0]["label"]) if predictions else "unknown"
        return float(baby["score"]), str(baby["label"])

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

    def classify(self, image_path: Path) -> tuple[float, str]:
        """Classify an image using the strongest detected face, if available."""
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        crops = self._face_crops(image)
        if not crops:
            return self._classify_input(str(image_path))
        predictions = [self._classify_input(crop) for crop in crops]
        strongest_face = max(predictions, key=lambda prediction: prediction[0])
        # Sleeping or heavily occluded faces can be detected but provide no
        # useful age signal. Preserve the whole-image detector as a safety net
        # in that case, while still preferring any positive face estimate.
        if strongest_face[0] > 0:
            return strongest_face
        return self._classify_input(str(image_path))

    def scan(
        self,
        paths: Iterable[Path],
        threshold: float = 0.50,
        cancel_event=None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Detection]:
        image_paths = list(paths)
        results: list[Detection] = []
        for index, path in enumerate(image_paths, 1):
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                confidence, label = self.classify(path)
                if confidence >= threshold:
                    results.append(Detection(path, confidence, label))
            except Exception:
                # A corrupt or unsupported image should not abort a long scan.
                pass
            finally:
                if on_progress:
                    on_progress(index, len(image_paths))
        return results


def image_paths(folder: Path, recursive: bool = False) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
