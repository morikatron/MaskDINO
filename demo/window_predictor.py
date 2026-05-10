import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from maskdino import add_maskdino_config
from maskdino.utils.border_inference import predict_border_with_padding


WINDOW_NAME = "MaskDINO Viewer"


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if hasattr(cfg.MODEL, "MaskDINO"):
        cfg.defrost()
        cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
        cfg.freeze()
    cfg.freeze()
    return cfg


def build_parser():
    parser = argparse.ArgumentParser(description="Windowed predictor for MaskDINO and border head")
    parser.add_argument(
        "--config-file",
        required=True,
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input image path(s) or one glob pattern",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for shown instances",
    )
    parser.add_argument(
        "--border-threshold",
        type=float,
        default=0.5,
        help="Threshold for border binary preview",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=1600,
        help="Resize the composed canvas for display if it is too large",
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using KEY VALUE pairs",
    )
    return parser


def resolve_inputs(input_args):
    if len(input_args) == 1:
        expanded = glob.glob(os.path.expanduser(input_args[0]))
        if expanded:
            return sorted(expanded)
    return [os.path.expanduser(path) for path in input_args]


def create_instance_panel(image_bgr, predictions, metadata, threshold):
    image_rgb = image_bgr[:, :, ::-1]
    visualizer = Visualizer(image_rgb, metadata=metadata, instance_mode=ColorMode.IMAGE)

    if "instances" not in predictions:
        return image_bgr.copy()

    instances = predictions["instances"].to(torch.device("cpu"))
    if instances.has("scores"):
        keep = instances.scores >= threshold
        instances = instances[keep]
    if len(instances) == 0:
        return image_bgr.copy()

    rendered = visualizer.draw_instance_predictions(instances).get_image()
    return rendered[:, :, ::-1]


def colorize_border_prob(prob_map):
    prob_map = np.asarray(prob_map, dtype=np.float32)
    max_value = float(prob_map.max()) if prob_map.size else 0.0
    if max_value <= 0.0:
        heat = np.zeros_like(prob_map, dtype=np.uint8)
    else:
        heat = np.clip((prob_map / max_value) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)


def create_border_panels(image_bgr, predictions, border_threshold):
    if "border_sem_seg" not in predictions:
        empty = np.zeros_like(image_bgr)
        cv2.putText(
            empty,
            "No border head output",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return empty, empty, "Border Heatmap", "Border Binary"

    border = predictions["border_sem_seg"]
    if isinstance(border, torch.Tensor):
        border = border.detach().to(torch.device("cpu")).numpy()
    if border.ndim == 3:
        border = border[0]

    border = np.clip(border, 0.0, 1.0)
    max_prob = float(border.max()) if border.size else 0.0
    mean_prob = float(border.mean()) if border.size else 0.0
    heatmap = colorize_border_prob(border)
    heat_overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0.0)

    effective_threshold = border_threshold
    adaptive = False
    if max_prob > 0.0 and max_prob < border_threshold:
        effective_threshold = max_prob * 0.5
        adaptive = True

    binary = (border >= effective_threshold).astype(np.uint8) * 255
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_overlay = cv2.addWeighted(image_bgr, 0.55, binary_bgr, 0.45, 0.0)
    heat_title = f"Border Heatmap max={max_prob:.4f} mean={mean_prob:.4f}"
    binary_title = f"Border Binary thr={effective_threshold:.4f}"
    if adaptive:
        binary_title += " adaptive"
    return heat_overlay, binary_overlay, heat_title, binary_title


def add_panel_title(image, title):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (image.shape[1], 44), (20, 20, 20), thickness=-1)
    cv2.putText(
        canvas,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_grid(original, instance_panel, border_heat, border_binary, heat_title, binary_title):
    top = np.concatenate(
        [
            add_panel_title(original, "Original"),
            add_panel_title(instance_panel, "Instances"),
        ],
        axis=1,
    )
    bottom = np.concatenate(
        [
            add_panel_title(border_heat, heat_title),
            add_panel_title(border_binary, binary_title),
        ],
        axis=1,
    )
    return np.concatenate([top, bottom], axis=0)


def resize_for_window(image, max_long_edge):
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def show_images(args):
    logger = setup_logger()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    metadata = MetadataCatalog.get(metadata_name)

    paths = resolve_inputs(args.input)
    if not paths:
        raise ValueError("No input images found.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    index = 0

    while 0 <= index < len(paths):
        path = paths[index]
        image = read_image(path, format="BGR")

        start_time = time.time()
        predictions = predictor(image)
        fixed_size = int(getattr(cfg.INPUT.BORDER_SEMANTIC, "FIXED_SIZE", 0))
        if fixed_size > 0 and "border_sem_seg" in predictions:
            restored_border = predict_border_with_padding(
                predictor,
                image,
                fixed_size,
                int(getattr(cfg.INPUT.BORDER_SEMANTIC, "PAD_VALUE", 255)),
            )
            if restored_border is not None:
                predictions = dict(predictions)
                predictions["border_sem_seg"] = restored_border
        elapsed = time.time() - start_time

        instance_panel = create_instance_panel(image, predictions, metadata, args.confidence_threshold)
        border_heat, border_binary, heat_title, binary_title = create_border_panels(
            image, predictions, args.border_threshold
        )
        canvas = compose_grid(image, instance_panel, border_heat, border_binary, heat_title, binary_title)
        canvas = resize_for_window(canvas, args.max_long_edge)

        status = f"[{index + 1}/{len(paths)}] {os.path.basename(path)}  {elapsed:.2f}s"
        cv2.setWindowTitle(WINDOW_NAME, status)
        cv2.imshow(WINDOW_NAME, canvas)
        logger.info(status)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (81, ord("a"), ord("h")):
            index = max(0, index - 1)
            continue
        index = min(len(paths) - 1, index + 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = build_parser().parse_args()
    show_images(args)
