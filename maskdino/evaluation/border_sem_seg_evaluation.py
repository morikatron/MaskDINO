import copy
import json
import logging
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager


class BorderSemSegEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, threshold=0.5, distributed=True, output_dir=None):
        self._dataset_name = dataset_name
        self._threshold = float(threshold)
        self._distributed = distributed
        self._output_dir = output_dir
        self._logger = logging.getLogger(__name__)
        self._dataset_records = list(DatasetCatalog.get(dataset_name))
        self.reset()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input_record, output_record in zip(inputs, outputs):
            border_pred = output_record.get("border_sem_seg")
            if border_pred is None:
                continue
            if isinstance(border_pred, torch.Tensor):
                border_pred = border_pred.detach().cpu().numpy()
            if border_pred.ndim == 3:
                border_pred = border_pred[0]
            self._predictions.append(
                {
                    "file_name": input_record["file_name"],
                    "border_prob": np.asarray(border_pred, dtype=np.float32),
                }
            )

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            gathered = comm.gather(self._predictions, dst=0)
            if not comm.is_main_process():
                return {}
            predictions = []
            for chunk in gathered:
                predictions.extend(chunk)
        else:
            predictions = self._predictions

        by_file = {item["file_name"]: item["border_prob"] for item in predictions}
        totals = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "prob_sum": 0.0,
            "pixel_count": 0,
            "image_count": 0,
        }

        for record in self._dataset_records:
            file_name = record["file_name"]
            if file_name not in by_file:
                continue

            mask_path = record["sem_seg_file_name"]
            with PathManager.open(mask_path, "rb") as handle:
                gt_mask = np.frombuffer(handle.read(), dtype=np.uint8)
            gt_mask = cv2.imdecode(gt_mask, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                raise ValueError(f"failed to load border ground truth: {mask_path}")

            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_prob = by_file[file_name]
            if pred_prob.shape != gt_binary.shape:
                raise ValueError(
                    f"shape mismatch for {file_name}: pred {pred_prob.shape}, gt {gt_binary.shape}"
                )
            pred_binary = (pred_prob >= self._threshold).astype(np.uint8)

            tp = int(np.logical_and(pred_binary == 1, gt_binary == 1).sum())
            fp = int(np.logical_and(pred_binary == 1, gt_binary == 0).sum())
            fn = int(np.logical_and(pred_binary == 0, gt_binary == 1).sum())
            tn = int(np.logical_and(pred_binary == 0, gt_binary == 0).sum())

            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            totals["tn"] += tn
            totals["prob_sum"] += float(pred_prob.sum())
            totals["pixel_count"] += int(pred_prob.size)
            totals["image_count"] += 1

        tp = totals["tp"]
        fp = totals["fp"]
        fn = totals["fn"]
        tn = totals["tn"]
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        iou = tp / max(tp + fp + fn, 1)
        mean_prob = totals["prob_sum"] / max(totals["pixel_count"], 1)

        results = OrderedDict(
            {
                "border_sem_seg": {
                    "border_accuracy": 100.0 * accuracy,
                    "border_precision": 100.0 * precision,
                    "border_recall": 100.0 * recall,
                    "border_f1": 100.0 * f1,
                    "border_iou": 100.0 * iou,
                    "border_mean_prob": mean_prob,
                    "evaluated_images": totals["image_count"],
                    "threshold": self._threshold,
                }
            }
        )

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            serializable = copy.deepcopy(results["border_sem_seg"])
            metrics_path = os.path.join(self._output_dir, f"{self._dataset_name}_border_metrics.json")
            with PathManager.open(metrics_path, "w") as handle:
                json.dump(serializable, handle, indent=2)

        return results
