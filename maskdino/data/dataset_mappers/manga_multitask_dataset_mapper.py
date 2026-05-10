import copy

import torch

from detectron2.config import configurable
from detectron2.structures import Boxes, Instances

from .coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

__all__ = ["MangaMultiTaskDatasetMapper"]


class MangaMultiTaskDatasetMapper:
    @configurable
    def __init__(self, *, instance_mapper, semantic_mapper):
        self.instance_mapper = instance_mapper
        self.semantic_mapper = semantic_mapper

    @classmethod
    def from_config(cls, cfg, is_train=True):
        semantic_cfg = cfg.clone()
        semantic_cfg.defrost()
        semantic_cfg.DATASETS.TRAIN = ("manga_border_semantic_train_split",)
        semantic_cfg.freeze()
        return {
            "instance_mapper": COCOInstanceNewBaselineDatasetMapper(cfg, is_train),
            "semantic_mapper": MaskFormerSemanticDatasetMapper(semantic_cfg, is_train),
        }

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        if "annotations" in dataset_dict:
            mapped = self.instance_mapper(dataset_dict)
            mapped["task_type"] = "instance"
            return mapped

        mapped = self.semantic_mapper(dataset_dict)
        image_shape = (mapped["image"].shape[-2], mapped["image"].shape[-1])
        empty_instances = Instances(image_shape)
        empty_instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
        empty_instances.gt_masks = torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.bool)
        empty_instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
        mapped["instances"] = empty_instances
        mapped["task_type"] = "border"
        return mapped
