"""Image output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    srgb = np.power(np.clip(image, 0.0, 1.0), 1.0 / 2.2)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels = to_uint8(image)

    if output.suffix.lower() == ".ppm":
        _save_ppm(output, pixels)
        return output

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Saving PNG/JPEG requires Pillow. Install `lingraphics[png]` "
            "or choose a `.ppm` output path."
        ) from error

    Image.fromarray(pixels, mode="RGB").save(output)
    return output


def _save_ppm(path: Path, pixels: np.ndarray) -> None:
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("PPM output requires an RGB image")
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(pixels).tobytes())
