import cv2
import numpy as np
import torch


def resize_and_pad_border_image(image_bgr, fixed_size, pad_value=255):
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(fixed_size / float(orig_h), fixed_size / float(orig_w))
    new_h = max(1, int(round(orig_h * scale)))
    new_w = max(1, int(round(orig_w * scale)))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((fixed_size, fixed_size, 3), pad_value, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "fixed_size": fixed_size,
    }
    return padded, meta


def restore_border_prediction(border_pred, meta):
    if isinstance(border_pred, torch.Tensor):
        border_pred = border_pred.detach().to(torch.device("cpu")).numpy()
    if border_pred.ndim == 3:
        border_pred = border_pred[0]

    cropped = border_pred[: meta["new_h"], : meta["new_w"]].astype(np.float32, copy=False)
    restored = cv2.resize(
        cropped,
        (meta["orig_w"], meta["orig_h"]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(restored, 0.0, 1.0)


def predict_border_with_padding(predictor, image_bgr, fixed_size, pad_value=255):
    padded_image, meta = resize_and_pad_border_image(image_bgr, fixed_size, pad_value)
    padded_predictions = predictor(padded_image)
    if "border_sem_seg" not in padded_predictions:
        return None
    return restore_border_prediction(padded_predictions["border_sem_seg"], meta)
