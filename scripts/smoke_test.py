"""Small headless checks for the scanner."""

from pathlib import Path
import sys
import tempfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_scanner.detector import BabyDetector, image_paths


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.new("RGB", (64, 64), (240, 180, 150)).save(root / "sample.png")
        (root / "ignore.txt").write_text("not an image")
        paths = image_paths(root)
        assert paths == [root / "sample.png"]
        detector = BabyDetector()
        confidence, label = detector.classify(paths[0])
        assert 0 <= confidence <= 1
        assert label
        print(f"Classifier smoke test passed: {label} ({confidence:.1%})")


if __name__ == "__main__":
    main()
