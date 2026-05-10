# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["MaskFormerSemanticDatasetMapper"]


class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        binary_from_255=False,
        threshold=127,
        upscale_factor=1.0,
        fixed_size=0,
        pad_value=255,
        dilate_kernel=0,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.binary_from_255 = binary_from_255
        self.threshold = threshold
        self.upscale_factor = upscale_factor
        self.fixed_size = fixed_size
        self.pad_value = pad_value
        self.dilate_kernel = dilate_kernel

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "binary_from_255": cfg.INPUT.BORDER_SEMANTIC.BINARY_FROM_255,
            "threshold": cfg.INPUT.BORDER_SEMANTIC.THRESHOLD,
            "upscale_factor": cfg.INPUT.BORDER_SEMANTIC.UPSCALE_FACTOR,
            "fixed_size": cfg.INPUT.BORDER_SEMANTIC.FIXED_SIZE,
            "pad_value": cfg.INPUT.BORDER_SEMANTIC.PAD_VALUE,
            "dilate_kernel": cfg.INPUT.BORDER_SEMANTIC.DILATE_KERNEL,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        if self.binary_from_255:
            sem_seg_gt = (sem_seg_gt >= self.threshold).astype("double")

        if self.upscale_factor != 1.0:
            new_h = max(1, int(round(image.shape[0] * self.upscale_factor)))
            new_w = max(1, int(round(image.shape[1] * self.upscale_factor)))

            image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()
            image_tensor = F.interpolate(
                image_tensor,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            image = image_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()

            sem_seg_tensor = torch.as_tensor(np.ascontiguousarray(sem_seg_gt)).unsqueeze(0).unsqueeze(0).float()
            sem_seg_tensor = F.interpolate(
                sem_seg_tensor,
                size=(new_h, new_w),
                mode="nearest",
            )
            sem_seg_gt = sem_seg_tensor.squeeze(0).squeeze(0).numpy()

        if self.fixed_size > 0:
            image, sem_seg_gt = self._resize_and_pad(image, sem_seg_gt, self.fixed_size)

        if self.dilate_kernel > 1:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
            sem_seg_gt = cv2.dilate(sem_seg_gt.astype(np.uint8), kernel, iterations=1).astype(sem_seg_gt.dtype)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            pad_h = (self.size_divisibility - image_size[0] % self.size_divisibility) % self.size_divisibility
            pad_w = (self.size_divisibility - image_size[1] % self.size_divisibility) % self.size_divisibility
            padding_size = [
                0,
                pad_w,
                0,
                pad_h,
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0,4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"] = instances

        return dataset_dict

    def _resize_and_pad(self, image, sem_seg_gt, size):
        scale = min(size / float(image.shape[0]), size / float(image.shape[1]))
        new_h = max(1, int(round(image.shape[0] * scale)))
        new_w = max(1, int(round(image.shape[1] * scale)))

        image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()
        image_tensor = F.interpolate(
            image_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        image_resized = image_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        image_padded = np.full((size, size, 3), self.pad_value, dtype=np.uint8)
        image_padded[:new_h, :new_w] = image_resized

        sem_seg_tensor = torch.as_tensor(np.ascontiguousarray(sem_seg_gt)).unsqueeze(0).unsqueeze(0).float()
        if self.binary_from_255:
            sem_seg_tensor = F.interpolate(
                sem_seg_tensor,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            sem_seg_tensor = F.interpolate(
                sem_seg_tensor,
                size=(new_h, new_w),
                mode="nearest",
            )
        sem_seg_resized = sem_seg_tensor.squeeze(0).squeeze(0).numpy()
        if self.binary_from_255:
            sem_seg_resized = (sem_seg_resized > 0).astype(np.float32)
        sem_seg_padded = np.zeros((size, size), dtype=sem_seg_resized.dtype)
        sem_seg_padded[:new_h, :new_w] = sem_seg_resized
        return image_padded, sem_seg_padded
