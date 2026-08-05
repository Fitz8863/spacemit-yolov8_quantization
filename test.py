"""Run a YOLOv8 ONNX model on one image and draw detections.

Default example:
    python test.py

Custom example:
    python test.py --model models/yolov8n_relu_110.q.onnx \
        --image bus.jpg --output bus_result.jpg --conf 0.25 --iou 0.45
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 ONNX single-image inference and visualization."
    )
    parser.add_argument(
        "--model",
        default="models/yolov8n_silu_110.q.onnx",
        help="FP32 or quantized ONNX model path.",
    )
    parser.add_argument("--image", default="bus.jpg", help="Input image path.")
    parser.add_argument(
        "--output", default="bus_result_silu.jpg", help="Annotated image output path."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Detection confidence threshold."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="Class-aware NMS IoU threshold."
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="Maximum detections to draw."
    )
    parser.add_argument(
        "--bgr",
        action="store_true",
        help="Feed BGR instead of RGB. Normally leave disabled for Ultralytics models.",
    )
    return parser.parse_args()


def letterbox(image, target_height, target_width, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio and symmetric padding."""
    source_height, source_width = image.shape[:2]
    ratio = min(target_height / source_height, target_width / source_width)
    resized_width = round(source_width * ratio)
    resized_height = round(source_height * ratio)

    if (resized_width, resized_height) != (source_width, source_height):
        image = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )

    pad_width = (target_width - resized_width) / 2
    pad_height = (target_height - resized_height) / 2
    top = round(pad_height - 0.1)
    bottom = round(pad_height + 0.1)
    left = round(pad_width - 0.1)
    right = round(pad_width + 0.1)

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return image, ratio, (pad_width, pad_height)


def preprocess(image, input_height, input_width, use_bgr=False):
    padded, ratio, pad = letterbox(image, input_height, input_width)
    if not use_bgr:
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), ratio, pad


def xywh_to_xyxy(boxes):
    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def nms(boxes, scores, iou_threshold):
    """NumPy NMS. Boxes passed here are already offset by class."""
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(scores)[::-1]
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    keep = []

    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def decode_output(output, conf_threshold, iou_threshold, max_det):
    """Decode a raw YOLOv8 detection output into xyxy boxes, scores and classes."""
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected output rank/shape [1, C, N] or [1, N, C], got {output.shape}")

    # Support both common layouts: [1, 4+nc, N] and [1, N, 4+nc].
    if output.shape[1] < output.shape[2]:
        predictions = output[0].T
    else:
        predictions = output[0]
    if predictions.shape[1] <= 4:
        raise ValueError(f"Output has no class scores: {output.shape}")

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    scores = np.max(class_scores, axis=1)
    selected = scores >= conf_threshold

    if not np.any(selected):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            class_scores.shape[1],
        )

    boxes = xywh_to_xyxy(predictions[selected, :4])
    scores = scores[selected]
    class_ids = class_ids[selected]

    # Class-aware NMS: boxes of different classes cannot suppress one another.
    max_wh = 7680.0
    offset_boxes = boxes + class_ids[:, None].astype(np.float32) * max_wh
    keep = nms(offset_boxes, scores, iou_threshold)[:max_det]
    return boxes[keep], scores[keep], class_ids[keep], class_scores.shape[1]


def restore_boxes(boxes, ratio, pad, original_shape):
    boxes = boxes.copy()
    if boxes.size == 0:
        return boxes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    height, width = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def class_color(class_id):
    """Generate a stable, visually distinct BGR color for a class."""
    return (
        int((37 * class_id + 80) % 255),
        int((17 * class_id + 160) % 255),
        int((29 * class_id + 220) % 255),
    )


def draw_detections(image, boxes, scores, class_ids, class_names):
    result = image.copy()
    line_width = max(round(sum(result.shape[:2]) / 2 * 0.003), 2)
    font_scale = line_width / 3
    font_thickness = max(line_width - 1, 1)

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = class_color(int(class_id))
        name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        label = f"{name} {score:.2f}"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)

        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_y = max(y1, text_height + 6)
        cv2.rectangle(
            result,
            (x1, text_y - text_height - 6),
            (x1 + text_width + 4, text_y),
            color,
            thickness=-1,
        )
        cv2.putText(
            result,
            label,
            (x1 + 2, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    return result


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be in [0, 1]")
    if args.max_det <= 0:
        raise ValueError("--max-det must be greater than 0")

    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(model_path), providers=providers)
    if len(session.get_inputs()) != 1:
        raise ValueError(f"Expected one model input, got {len(session.get_inputs())}")

    input_info = session.get_inputs()[0]
    input_shape = input_info.shape
    if len(input_shape) != 4:
        raise ValueError(f"Expected NCHW model input, got {input_shape}")
    input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
    input_width = input_shape[3] if isinstance(input_shape[3], int) else 640

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV cannot decode image: {image_path}")

    tensor, ratio, pad = preprocess(
        image, input_height, input_width, use_bgr=args.bgr
    )

    start = time.perf_counter()
    output = session.run(None, {input_info.name: tensor})[0]
    inference_ms = (time.perf_counter() - start) * 1000

    boxes, scores, class_ids, num_classes = decode_output(
        output, args.conf, args.iou, args.max_det
    )
    boxes = restore_boxes(boxes, ratio, pad, image.shape[:2])
    class_names = VOC_CLASSES if num_classes == len(VOC_CLASSES) else tuple(
        f"class_{i}" for i in range(num_classes)
    )
    result = draw_detections(image, boxes, scores, class_ids, class_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to save result image: {output_path}")

    print(f"Model:       {model_path}")
    print(f"Provider:    {session.get_providers()[0]}")
    print(f"Input image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"Model input: {input_width}x{input_height}, RGB={not args.bgr}, letterbox=True")
    print(f"Raw output:  {tuple(output.shape)} ({num_classes} classes)")
    print(f"Inference:   {inference_ms:.2f} ms (session.run only)")
    print(f"Detections:  {len(boxes)}")
    for box, score, class_id in zip(boxes, scores, class_ids):
        name = class_names[class_id]
        coords = ", ".join(f"{value:.1f}" for value in box)
        print(f"  {name:<12} conf={score:.4f} box=[{coords}]")
    print(f"Saved:       {output_path}")


if __name__ == "__main__":
    main()
