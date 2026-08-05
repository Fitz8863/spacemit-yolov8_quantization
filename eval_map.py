"""Evaluate mAP@0.5 and mAP@0.5:0.95 for a YOLOv8 ONNX model on a YOLO-format dataset.

Self-contained: only needs onnxruntime, opencv, numpy, tqdm.
Dataset layout expected:
    <root>/images/*.jpg (or .jpeg / .png / .bmp)
    <root>/labels/*.txt   # each line: cls xc yc w h  (normalized)

Preprocessing defaults match Ultralytics YOLOv8 training defaults
(verified from args.yaml: rect=False, bgr=0.0, imgsz=640):
  - LetterBox resize to model input size (aspect-ratio preserving, pad=114, center=True)
  - BGR → RGB channel swap
  - Normalize to [0, 1] by dividing by 255
Use --no-letterbox --bgr only if the model was explicitly trained that way.
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate mAP@0.5 and mAP@0.5:0.95 for a YOLOv8 ONNX model."
    )
    parser.add_argument("--model", required=True, help="Path to the ONNX model.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root containing images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most this many images. 0 means all.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold. Keep it low for mAP; raise it for speed.",
    )
    parser.add_argument(
        "--iou", type=float, default=0.7, help="NMS IoU threshold."
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="Max detections kept per image."
    )
    parser.add_argument(
        "--letterbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Letterbox resize (keeps aspect ratio). Use --no-letterbox to match "
        "a calibration preprocess that stretched images instead.",
    )
    parser.add_argument(
        "--bgr",
        action="store_true",
        help="Feed BGR instead of RGB. Use only if the model was explicitly trained with bgr>0.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Optional file with one class name per line. Defaults to VOC 20 classes.",
    )
    parser.add_argument(
        "--save-report",
        default=None,
        metavar="FILE",
        help="Write a Markdown evaluation report to FILE (e.g. eval_report.md).",
    )
    parser.add_argument(
        "--ort-threads",
        type=int,
        default=1,
        help=(
            "ONNX Runtime intra-op CPU threads per worker. The default 1 "
            "keeps quantized-model results reproducible; increase only after "
            "checking accuracy."
        ),
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help=(
            "OpenCV worker threads per evaluator worker. Keep this at 1 when "
            "using multiple --workers to avoid CPU oversubscription."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=(
            "Parallel image workers. Each worker owns one ORT session. "
            "Default 4 is usually faster on a CPU server; use 1 for the "
            "strict sequential reference run."
        ),
    )
    return parser.parse_args()


def load_class_names(names_path, num_classes):
    if names_path is not None:
        names = [
            line.strip()
            for line in Path(names_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) != num_classes:
            raise ValueError(
                f"{names_path} has {len(names)} names but model outputs {num_classes} classes"
            )
        return names
    if num_classes == len(VOC_CLASSES):
        return list(VOC_CLASSES)
    return [f"class_{i}" for i in range(num_classes)]


def collect_samples(dataset_root, limit=0):
    root = Path(dataset_root).expanduser().resolve()
    image_dir = root / "images"
    label_dir = root / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {label_dir}")

    samples = []
    missing_labels = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            continue
        samples.append((image_path, label_path))
        if limit > 0 and len(samples) >= limit:
            break
    return samples, missing_labels


def read_labels(label_path, width, height):
    """Read YOLO-format labels and return (classes, xyxy boxes in pixels)."""
    classes = []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        classes.append(cls)
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 4), dtype=np.float32)
    return (
        np.array(classes, dtype=np.int32),
        np.array(boxes, dtype=np.float32),
    )


def letterbox(image, new_shape, color=(114, 114, 114)):
    """Letterbox resize matching Ultralytics LetterBox(center=True, padding_value=114).

    Scales the image so the long side fits inside new_shape, then pads both sides
    symmetrically.  The rounding logic mirrors Ultralytics augment.py exactly.
    """
    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    unpad_w, unpad_h = round(width * ratio), round(height * ratio)
    pad_w = (new_shape[1] - unpad_w) / 2
    pad_h = (new_shape[0] - unpad_h) / 2

    if (width, height) != (unpad_w, unpad_h):
        image = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    # Symmetric padding: matches Ultralytics center=True (round -0.1 / +0.1 trick)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, ratio, (pad_w, pad_h)


def preprocess(image, input_height, input_width, use_letterbox, use_bgr):
    """Preprocess an image for YOLOv8 inference.

    Default mode (use_letterbox=True, use_bgr=False) matches Ultralytics training:
      - LetterBox resize (aspect-ratio preserving, pad=114, center=True, INTER_LINEAR)
      - BGR → RGB channel swap  (training: bgr=0.0, always RGB)
      - Normalize to [0, 1] by dividing by 255
    """
    if use_letterbox:
        padded, ratio, pad = letterbox(image, (input_height, input_width))
    else:
        height, width = image.shape[:2]
        padded = cv2.resize(
            image, (input_width, input_height), interpolation=cv2.INTER_LINEAR
        )
        # Independent x/y scales; keep them separately for coordinate recovery.
        ratio = (input_width / width, input_height / height)
        pad = (0.0, 0.0)

    if not use_bgr:
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0), ratio, pad


def xywh2xyxy(boxes):
    result = np.empty_like(boxes)
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    result[:, 0] = boxes[:, 0] - half_w
    result[:, 1] = boxes[:, 1] - half_h
    result[:, 2] = boxes[:, 0] + half_w
    result[:, 3] = boxes[:, 1] + half_h
    return result


def nms(boxes, scores, iou_threshold):
    """Plain single-class NMS. Returns kept indices, highest score first."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def postprocess(output, conf_threshold, iou_threshold, max_det):
    """Decode YOLOv8 output [1, 4+nc, N] into (boxes_xyxy, scores, classes)."""
    predictions = np.squeeze(output, axis=0).T  # [N, 4+nc]
    class_scores = predictions[:, 4:]
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = confidences > conf_threshold
    if not mask.any():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    boxes = xywh2xyxy(predictions[mask, :4])
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Class-aware NMS: offset boxes per class so classes never suppress each other.
    offsets = class_ids[:, None].astype(np.float32) * 8192.0
    keep = nms(boxes + offsets, confidences, iou_threshold)[:max_det]
    return boxes[keep], confidences[keep], class_ids[keep]


def scale_boxes(boxes, ratio, pad, original_shape, use_letterbox):
    if boxes.shape[0] == 0:
        return boxes
    boxes = boxes.copy()
    if use_letterbox:
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= ratio
    else:
        scale_x, scale_y = ratio
        boxes[:, [0, 2]] /= scale_x
        boxes[:, [1, 3]] /= scale_y
    height, width = original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def box_iou_matrix(boxes_a, boxes_b):
    """IoU between every pair. boxes_a [M,4], boxes_b [N,4] -> [M,N]."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clip(0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clip(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clip(0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clip(0)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_predictions(pred_classes, pred_boxes, gt_classes, gt_boxes, iou_levels):
    """Greedy IoU matching per threshold. Returns bool [n_pred, n_levels]."""
    correct = np.zeros((pred_classes.shape[0], iou_levels.size), dtype=bool)
    if gt_classes.size == 0 or pred_classes.size == 0:
        return correct

    iou = box_iou_matrix(gt_boxes, pred_boxes)
    # Zero out cross-class pairs so they can never match.
    iou = iou * (gt_classes[:, None] == pred_classes[None, :])

    for level_index, threshold in enumerate(iou_levels):
        gt_idx, pred_idx = np.nonzero(iou >= threshold)
        if gt_idx.size == 0:
            continue
        matches = np.stack((gt_idx, pred_idx), axis=1)
        if matches.shape[0] > 1:
            scores = iou[gt_idx, pred_idx]
            matches = matches[scores.argsort()[::-1]]
            # One prediction per GT, and one GT per prediction.
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1], level_index] = True
    return correct


def compute_ap(recall, precision):
    """101-point interpolated AP, the COCO convention."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    points = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(points, mrec, mpre), points))


def ap_per_class(correct, confidence, pred_classes, target_classes, num_classes):
    """Returns per-class AP [n_classes, n_iou_levels] and support counts."""
    order = confidence.argsort()[::-1]
    correct = correct[order]
    pred_classes = pred_classes[order]

    ap = np.zeros((num_classes, correct.shape[1]), dtype=np.float64)
    gt_counts = np.bincount(target_classes, minlength=num_classes)
    evaluated = np.zeros(num_classes, dtype=bool)

    for class_id in range(num_classes):
        n_gt = gt_counts[class_id]
        if n_gt == 0:
            continue
        mask = pred_classes == class_id
        if not mask.any():
            evaluated[class_id] = True  # AP stays 0: GT exists but nothing predicted.
            continue
        evaluated[class_id] = True

        true_positive = correct[mask].cumsum(axis=0)
        false_positive = (~correct[mask]).cumsum(axis=0)
        recall = true_positive / n_gt
        precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)

        for level in range(correct.shape[1]):
            ap[class_id, level] = compute_ap(recall[:, level], precision[:, level])

    return ap, gt_counts, evaluated


def _write_report(path, model_path, args, input_height, input_width,
                  num_images, class_names, ap, gt_counts, evaluated,
                  map50, map5095):
    """Write a Markdown evaluation report to *path*."""
    from datetime import datetime
    import os

    lines = [
        "# YOLOv8 mAP Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model",
        f"- Path: `{model_path}`",
        f"- Input size: {input_height}×{input_width}",
        "",
        "## Preprocessing",
        f"- Resize: {'letterbox (aspect-ratio preserving, pad=114)' if args.letterbox else 'stretch resize'}",
        f"- Channel order: {'BGR' if args.bgr else 'RGB'}",
        "- Normalization: ÷255",
        "",
        "## Evaluation settings",
        f"- Confidence threshold: {args.conf}",
        f"- NMS IoU threshold: {args.iou}",
        f"- Max detections: {args.max_det}",
        f"- Images evaluated: {num_images}",
        "",
        "## Results",
        "",
        "| Class | GT | AP@0.5 | AP@0.5:0.95 |",
        "|---|---:|---:|---:|",
    ]

    for class_id, name in enumerate(class_names):
        if not evaluated[class_id]:
            continue
        lines.append(
            f"| {name} | {gt_counts[class_id]} "
            f"| {ap[class_id, 0]:.4f} | {ap[class_id].mean():.4f} |"
        )

    present = evaluated
    total_gt = int(gt_counts[present].sum()) if present.any() else 0
    lines += [
        f"| **all** | **{total_gt}** | **{map50:.4f}** | **{map5095:.4f}** |",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| mAP@0.5 | **{map50:.6f}** |",
        f"| mAP@0.5:0.95 | **{map5095:.6f}** |",
        "",
    ]

    out_path = os.path.expanduser(path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


_worker_state = threading.local()


def _create_session(model_path, providers, ort_threads):
    """Create one CPU-friendly ORT session for an evaluator worker."""
    session_options = ort.SessionOptions()
    if ort_threads > 0:
        session_options.intra_op_num_threads = ort_threads
    # Each worker processes one image at a time.
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=session_options, providers=providers
    )


def _init_worker(
    model_path,
    providers,
    input_height,
    input_width,
    use_letterbox,
    use_bgr,
    conf_threshold,
    iou_threshold,
    max_det,
    iou_levels,
    ort_threads,
    opencv_threads,
):
    """Initialize thread-local model state for a parallel evaluation worker."""
    cv2.setNumThreads(opencv_threads)
    session = _create_session(model_path, providers, ort_threads)
    input_name = session.get_inputs()[0].name
    _worker_state.session = session
    _worker_state.input_name = input_name
    _worker_state.input_height = input_height
    _worker_state.input_width = input_width
    _worker_state.use_letterbox = use_letterbox
    _worker_state.use_bgr = use_bgr
    _worker_state.conf_threshold = conf_threshold
    _worker_state.iou_threshold = iou_threshold
    _worker_state.max_det = max_det
    _worker_state.iou_levels = iou_levels


def _evaluate_one(sample):
    """Evaluate one image using the calling worker's thread-local session."""
    image_path, label_path = sample
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return image_path, None, None, None, None

    state = _worker_state
    original_shape = image.shape[:2]
    tensor, ratio, pad = preprocess(
        image,
        state.input_height,
        state.input_width,
        state.use_letterbox,
        state.use_bgr,
    )
    output = state.session.run(None, {state.input_name: tensor})[0]
    boxes, scores, classes = postprocess(
        output,
        state.conf_threshold,
        state.iou_threshold,
        state.max_det,
    )
    boxes = scale_boxes(
        boxes, ratio, pad, original_shape, state.use_letterbox
    )
    gt_classes, gt_boxes = read_labels(
        label_path, original_shape[1], original_shape[0]
    )

    correct = None
    if classes.size:
        correct = match_predictions(
            classes, boxes, gt_classes, gt_boxes, state.iou_levels
        )
    return image_path, gt_classes, correct, scores, classes


def evaluate(args):
    if args.ort_threads < 0:
        raise ValueError("--ort-threads must be >= 0")
    if args.opencv_threads < 0:
        raise ValueError("--opencv-threads must be >= 0")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than 0")

    model_path = Path(args.model).expanduser().resolve()
    samples, missing_labels = collect_samples(args.dataset, args.limit)
    if not samples:
        raise RuntimeError("No image/label pairs found.")

    available_providers = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers
        else ["CPUExecutionProvider"]
    )

    # One metadata session is used to validate the model and determine its input
    # shape. For workers=1, reuse this session instead of loading the model twice.
    cv2.setNumThreads(args.opencv_threads)
    metadata_session = _create_session(model_path, providers, args.ort_threads)
    input_meta = metadata_session.get_inputs()[0]
    input_shape = input_meta.shape
    if len(input_shape) != 4:
        raise ValueError(f"Expected NCHW input [N,C,H,W], got {input_shape}")
    input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
    input_width = input_shape[3] if isinstance(input_shape[3], int) else 640

    output_shape = metadata_session.get_outputs()[0].shape
    if len(output_shape) != 3 or not isinstance(output_shape[1], int):
        raise ValueError(
            f"Expected YOLO output [1,4+nc,N], got {output_shape}"
        )
    num_classes = output_shape[1] - 4
    if num_classes <= 0:
        raise ValueError(f"Model output has invalid class count: {output_shape}")
    class_names = load_class_names(args.names, num_classes)

    print(f"Model:      {model_path}")
    print(f"Providers:  {metadata_session.get_providers()}")
    print(f"Input:      {input_meta.name} {input_shape} -> {input_height}x{input_width}")
    print(f"Output:     {output_shape}  ({num_classes} classes)")
    print(
        f"Preprocess: {'letterbox' if args.letterbox else 'stretch resize'}, "
        f"{'BGR' if args.bgr else 'RGB'}, scale 1/255"
    )
    print(f"NMS:        conf>{args.conf} iou={args.iou} max_det={args.max_det}")
    print(
        f"CPU workers: {args.workers}; ORT intra per worker: "
        f"{args.ort_threads or 'auto'}; OpenCV per worker: {args.opencv_threads}"
    )
    if args.workers > 1 and args.ort_threads > 1:
        print(
            "Warning: --workers > 1 and --ort-threads > 1 may oversubscribe "
            "the CPU and can change quantized-model results."
        )
    if missing_labels:
        print(f"Skipped images without labels: {missing_labels}")

    iou_levels = np.linspace(0.5, 0.95, 10)
    all_correct = []
    all_confidence = []
    all_pred_classes = []
    all_target_classes = []
    decode_failures = 0

    started = time.perf_counter()
    progress = tqdm(total=len(samples), dynamic_ncols=True, desc=f"eval n={len(samples)}")
    if args.workers == 1:
        _worker_state.session = metadata_session
        _worker_state.input_name = input_meta.name
        _worker_state.input_height = input_height
        _worker_state.input_width = input_width
        _worker_state.use_letterbox = args.letterbox
        _worker_state.use_bgr = args.bgr
        _worker_state.conf_threshold = args.conf
        _worker_state.iou_threshold = args.iou
        _worker_state.max_det = args.max_det
        _worker_state.iou_levels = iou_levels
        result_iter = map(_evaluate_one, samples)
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(
                model_path,
                providers,
                input_height,
                input_width,
                args.letterbox,
                args.bgr,
                args.conf,
                args.iou,
                args.max_det,
                iou_levels,
                args.ort_threads,
                args.opencv_threads,
            ),
        )
        result_iter = executor.map(_evaluate_one, samples)

    try:
        for image_path, gt_classes, correct_i, scores_i, classes_i in result_iter:
            progress.update(1)
            if gt_classes is None:
                decode_failures += 1
                continue
            all_target_classes.append(gt_classes)
            if correct_i is not None:
                all_correct.append(correct_i)
                all_confidence.append(scores_i)
                all_pred_classes.append(classes_i)
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(wait=True)

    if decode_failures:
        print(f"Skipped unreadable images: {decode_failures}")

    target_classes = (
        np.concatenate(all_target_classes)
        if all_target_classes
        else np.zeros(0, np.int32)
    )
    if not all_correct:
        raise RuntimeError("Model produced no detections above the confidence threshold.")

    correct = np.concatenate(all_correct)
    confidence = np.concatenate(all_confidence)
    pred_classes = np.concatenate(all_pred_classes)

    ap, gt_counts, evaluated = ap_per_class(
        correct, confidence, pred_classes, target_classes, num_classes
    )

    print(f"\n{'Class':<16}{'GT':>8}{'AP@0.5':>10}{'AP@0.5:0.95':>14}")
    print("-" * 48)
    for class_id in range(num_classes):
        if not evaluated[class_id]:
            continue
        print(
            f"{class_names[class_id]:<16}{gt_counts[class_id]:>8}"
            f"{ap[class_id, 0]:>10.4f}{ap[class_id].mean():>14.4f}"
        )

    present = evaluated
    map50 = ap[present, 0].mean() if present.any() else 0.0
    map5095 = ap[present].mean() if present.any() else 0.0

    elapsed = time.perf_counter() - started
    images_evaluated = len(samples) - decode_failures
    print("-" * 48)
    print(f"{'all':<16}{gt_counts[present].sum():>8}{map50:>10.4f}{map5095:>14.4f}")
    print(f"\nImages evaluated: {images_evaluated}")
    print(f"Elapsed:         {elapsed:.2f} s ({images_evaluated / elapsed:.2f} images/s)")
    print(f"mAP@0.5:         {map50:.6f}")
    print(f"mAP@0.5:0.95:    {map5095:.6f}")

    if args.save_report:
        _write_report(
            path=args.save_report,
            model_path=model_path,
            args=args,
            input_height=input_height,
            input_width=input_width,
            num_images=images_evaluated,
            class_names=class_names,
            ap=ap,
            gt_counts=gt_counts,
            evaluated=evaluated,
            map50=map50,
            map5095=map5095,
        )
        print(f"Report saved → {args.save_report}")

def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
