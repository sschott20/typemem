"""Disk-backed image store with TTL-based cleanup."""

import json
import logging
import os
import time
import uuid
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

def _detect_encoder():
    try:
        import cv2
        return "cv2"
    except ImportError:
        return "pil"

_ENCODER = _detect_encoder()


def _encode_jpeg(frame: np.ndarray) -> bytes:
    if _ENCODER == "cv2":
        import cv2
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
    from PIL import Image
    import io
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _decode_jpeg(data: bytes) -> np.ndarray:
    if _ENCODER == "cv2":
        import cv2
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(data))
    return np.array(img)


class FrameStore:
    """Disk-backed image store with TTL-based cleanup."""

    def __init__(self, store_dir: str, default_ttl: float = 600.0):
        self._store_dir = store_dir
        self._default_ttl = default_ttl
        os.makedirs(store_dir, exist_ok=True)

    def store(self, frame: np.ndarray, timestamp: float, waypoint: Optional[int] = None) -> str:
        frame_id = f"f_{uuid.uuid4().hex[:12]}"
        img_path = os.path.join(self._store_dir, f"{frame_id}.jpg")
        meta_path = os.path.join(self._store_dir, f"{frame_id}.json")

        jpeg_bytes = _encode_jpeg(frame)
        with open(img_path, "wb") as f:
            f.write(jpeg_bytes)

        meta = {"timestamp": timestamp, "frame_id": frame_id}
        if waypoint is not None:
            meta["waypoint"] = waypoint
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        return frame_id

    def load(self, frame_id: str) -> Optional[np.ndarray]:
        img_path = os.path.join(self._store_dir, f"{frame_id}.jpg")
        if not os.path.exists(img_path):
            return None
        with open(img_path, "rb") as f:
            data = f.read()
        return _decode_jpeg(data)

    def cleanup(self, max_age: Optional[float] = None) -> int:
        ttl = max_age if max_age is not None else self._default_ttl
        cutoff = time.time() - ttl
        removed = 0

        for fname in os.listdir(self._store_dir):
            if not fname.endswith(".json"):
                continue
            meta_path = os.path.join(self._store_dir, fname)
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("timestamp", 0) < cutoff:
                    frame_id = meta["frame_id"]
                    img_path = os.path.join(self._store_dir, f"{frame_id}.jpg")
                    try:
                        os.remove(img_path)
                    except FileNotFoundError:
                        pass
                    os.remove(meta_path)
                    removed += 1
            except Exception as e:
                logger.warning("Failed to process %s during cleanup: %s", fname, e)

        return removed
