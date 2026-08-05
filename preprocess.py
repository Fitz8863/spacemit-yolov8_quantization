"""Calibration preprocessing for Ultralytics YOLOv8 ONNX quantization.

Must match the training-time preprocessing exactly:
  - LetterBox resize to (input_h, input_w), padding_value=114, center=True
  - BGR → RGB channel swap
  - Normalize to [0, 1] by dividing by 255
  - Output layout: NCHW float32 tensor

Matches: Ultralytics YOLOv8 default (rect=False, bgr=0.0, imgsz=640).
"""

from typing import Sequence

import cv2
import numpy as np
import torch


def _letterbox(img: np.ndarray, target_h: int, target_w: int,
               padding_value: int = 114) -> np.ndarray:
    """Letterbox resize matching Ultralytics LetterBox(center=True).

    Scales the image so the long side fits inside (target_h, target_w),
    then pads both sides symmetrically with padding_value.
    """
    src_h, src_w = img.shape[:2]
    ratio = min(target_h / src_h, target_w / src_w)

    # Scaled (unpadded) size
    new_w = round(src_w * ratio)
    new_h = round(src_h * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute symmetric padding (matches Ultralytics center=True rounding)
    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2
    top    = round(pad_h - 0.1)
    bottom = round(pad_h + 0.1)
    left   = round(pad_w - 0.1)
    right  = round(pad_w + 0.1)

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(padding_value, padding_value, padding_value),
    )
    return img


def preprocess_impl(
    path_list: Sequence[str], input_parametr: dict
) -> torch.Tensor:
    """Load images and apply YOLOv8-compatible preprocessing.

    Args:
        path_list:     List of image file paths for this calibration batch.
        input_parametr: Dict injected by xslim; must contain 'input_shape'
                        ([N, C, H, W] or [C, H, W]).  Optional keys
                        'mean_value' and 'std_value' are accepted but unused
                        (YOLOv8 training used only /255 normalization).

    Returns:
        Float32 NCHW tensor of shape [len(path_list), 3, H, W] in [0, 1].
    """
    # Resolve target spatial size from xslim-injected input_shape.
    input_shape = input_parametr.get("input_shape", [1, 3, 640, 640])
    target_h = int(input_shape[-2])
    target_w = int(input_shape[-1])

    batch_list = []
    for file_path in path_list:
        img = cv2.imread(file_path)  # uint8 BGR HWC
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {file_path}")

        # 1. Letterbox resize (aspect-ratio preserving, pad with 114)
        img = _letterbox(img, target_h, target_w, padding_value=114)

        # 2. BGR → RGB  (training used bgr=0.0, i.e. always RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. HWC → CHW, normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)

        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
        batch_list.append(tensor)

    return torch.cat(batch_list, dim=0)  # (N, 3, H, W)
