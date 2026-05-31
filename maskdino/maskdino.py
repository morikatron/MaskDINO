# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.comm import get_world_size
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils import box_ops


@META_ARCH_REGISTRY.register()
class MaskDINO(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        instance_box_source: str = "pred",
        instance_mask_source: str = "auto",
        semantic_ce_loss: bool = False,
        border_head_enabled: bool = False,
        border_train_only: bool = False,
        border_loss_weight: float = 1.0,
        border_bce_weight: float = 1.0,
        border_dice_weight: float = 1.0,
        border_pos_weight: float = 8.0,
        instance_mask_refine_enabled: bool = False,
        instance_mask_refine_backbone_downscale_factor: float = 0.5,
        instance_mask_refine_loss_weight: float = 1.0,
        instance_mask_refine_mask_weight: float = 5.0,
        instance_mask_refine_dice_weight: float = 5.0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            transform_eval: transform sigmoid score into softmax score to make score sharper
            semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.instance_box_source = instance_box_source
        self.instance_mask_source = instance_mask_source
        self.semantic_ce_loss = semantic_ce_loss
        self.border_head_enabled = border_head_enabled
        self.border_train_only = border_train_only
        self.border_loss_weight = border_loss_weight
        self.border_bce_weight = border_bce_weight
        self.border_dice_weight = border_dice_weight
        self.border_pos_weight = border_pos_weight
        self.instance_mask_refine_enabled = instance_mask_refine_enabled
        self.instance_mask_refine_backbone_downscale_factor = instance_mask_refine_backbone_downscale_factor
        self.instance_mask_refine_loss_weight = instance_mask_refine_loss_weight
        self.instance_mask_refine_mask_weight = instance_mask_refine_mask_weight
        self.instance_mask_refine_dice_weight = instance_mask_refine_dice_weight

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print('criterion.weight_dict ', self.criterion.weight_dict)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["labels","boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks","boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]
        # building criterion
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "instance_box_source": cfg.MODEL.MaskDINO.TEST.BOX_INFERENCE_SOURCE,
            "instance_mask_source": cfg.MODEL.MaskDINO.TEST.MASK_INFERENCE_SOURCE,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "border_head_enabled": cfg.MODEL.BORDER_HEAD.ENABLED,
            "border_train_only": cfg.MODEL.BORDER_HEAD.TRAIN_ONLY,
            "border_loss_weight": cfg.MODEL.BORDER_HEAD.LOSS_WEIGHT,
            "border_bce_weight": cfg.MODEL.BORDER_HEAD.BCE_WEIGHT,
            "border_dice_weight": cfg.MODEL.BORDER_HEAD.DICE_WEIGHT,
            "border_pos_weight": cfg.MODEL.BORDER_HEAD.POS_WEIGHT,
            "instance_mask_refine_enabled": cfg.MODEL.INSTANCE_MASK_REFINE.ENABLED,
            "instance_mask_refine_backbone_downscale_factor": cfg.MODEL.INSTANCE_MASK_REFINE.BACKBONE_DOWNSCALE_FACTOR,
            "instance_mask_refine_loss_weight": cfg.MODEL.INSTANCE_MASK_REFINE.LOSS_WEIGHT,
            "instance_mask_refine_mask_weight": cfg.MODEL.INSTANCE_MASK_REFINE.MASK_WEIGHT,
            "instance_mask_refine_dice_weight": cfg.MODEL.INSTANCE_MASK_REFINE.DICE_WEIGHT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    @staticmethod
    def _infer_task_type(input_per_image):
        if "task_type" in input_per_image:
            return input_per_image["task_type"]
        if "sem_seg" in input_per_image:
            return "border"
        return "instance"

    @staticmethod
    def _slice_batch_outputs(outputs, indices, batch_size):
        if len(indices) == batch_size:
            return outputs

        def _slice_value(value, index_tensor):
            if torch.is_tensor(value):
                if value.dim() > 0 and value.shape[0] == batch_size:
                    return value.index_select(0, index_tensor)
                return value
            if isinstance(value, list):
                return [_slice_value(item, index_tensor) for item in value]
            if isinstance(value, tuple):
                return tuple(_slice_value(item, index_tensor) for item in value)
            if isinstance(value, dict):
                return {key: _slice_value(item, index_tensor) for key, item in value.items()}
            return value

        device = None
        for value in outputs.values():
            if torch.is_tensor(value):
                device = value.device
                break
            if isinstance(value, dict):
                for nested_value in value.values():
                    if torch.is_tensor(nested_value):
                        device = nested_value.device
                        break
                if device is not None:
                    break
        if device is None:
            device = torch.device("cpu")

        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        return {key: _slice_value(value, index_tensor) for key, value in outputs.items()}

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        fullres_images = [x["image"].to(self.device) for x in batched_inputs]
        fullres_images = [(x - self.pixel_mean) / self.pixel_std for x in fullres_images]
        images_fullres = ImageList.from_tensors(fullres_images, self.size_divisibility)

        if self.instance_mask_refine_enabled:
            downscale = float(self.instance_mask_refine_backbone_downscale_factor)
            backbone_images = [
                F.interpolate(
                    image.unsqueeze(0),
                    scale_factor=downscale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                ).squeeze(0)
                for image in fullres_images
            ]
        else:
            backbone_images = fullres_images

        images = ImageList.from_tensors(backbone_images, self.size_divisibility)
        features = self.backbone(images.tensor)

        if self.training:
            # dn_args={"scalar":30,"noise_scale":0.4}
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                if 'detr' in self.data_loader:
                    targets = self.prepare_targets_detr(gt_instances, images_fullres)
                else:
                    targets = self.prepare_targets(gt_instances, images_fullres)
            else:
                targets = None
            outputs,mask_dict = self.sem_seg_head(features, targets=targets, images=images_fullres.tensor)
            task_types = [self._infer_task_type(x) for x in batched_inputs]
            instance_indices = [i for i, task_type in enumerate(task_types) if task_type == "instance"]
            border_indices = [i for i, task_type in enumerate(task_types) if task_type == "border"]
            losses = {}
            if not self.border_train_only and len(instance_indices) > 0:
                # bipartite matching-based loss
                instance_outputs = self._slice_batch_outputs(outputs, instance_indices, len(batched_inputs))
                instance_targets = [targets[i] for i in instance_indices]
                losses = self.criterion(instance_outputs, instance_targets, None)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                if self.instance_mask_refine_enabled and "pred_masks_refined" in instance_outputs:
                    losses.update(self.compute_instance_refine_losses(instance_outputs, instance_targets))
            if self.border_head_enabled and len(border_indices) > 0:
                border_logits = outputs["border_logits"]
                if border_logits is not None and len(border_indices) != len(batched_inputs):
                    border_logits = border_logits[border_indices]
                border_inputs = [batched_inputs[i] for i in border_indices]
                losses.update(self.compute_border_losses(border_logits, border_inputs))
            return losses
        else:
            outputs, _ = self.sem_seg_head(features, images=images_fullres.tensor)
            border_logits_results = outputs.get("border_logits")
            mask_cls_results = outputs.get("pred_logits")
            mask_pred_results = self.select_instance_mask_predictions(outputs)
            mask_box_results = outputs.get("pred_boxes")
            # upsample masks
            if mask_pred_results is not None:
                upsample_target = (
                    (images_fullres.tensor.shape[-2], images_fullres.tensor.shape[-1])
                    if self.instance_mask_refine_enabled
                    else (images.tensor.shape[-2], images.tensor.shape[-1])
                )
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=upsample_target,
                    mode="bilinear",
                    align_corners=False,
                )

            del outputs

            processed_results = []
            result_image_sizes = images_fullres.image_sizes if self.instance_mask_refine_enabled else images.image_sizes
            for image_index, (input_per_image, image_size) in enumerate(zip(batched_inputs, result_image_sizes)):
                mask_cls_result = None if mask_cls_results is None else mask_cls_results[image_index]
                mask_pred_result = None if mask_pred_results is None else mask_pred_results[image_index]
                mask_box_result = None if mask_box_results is None else mask_box_results[image_index]
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = image_size

                if mask_pred_result is not None and self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # mask_box_result = mask_box_result.to(mask_pred_result)
                    # mask_box_result = self.box_postprocess(mask_box_result, height, width)

                if border_logits_results is not None:
                    border_pred_result = border_logits_results[image_index, :, : image_size[0], : image_size[1]]
                    if self.sem_seg_postprocess_before_inference:
                        border_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            border_pred_result, image_size, height, width
                        )
                    processed_results[-1]["border_sem_seg"] = border_pred_result.sigmoid()

                # semantic segmentation inference
                if self.semantic_on and mask_pred_result is not None:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on and mask_pred_result is not None:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference

                if self.instance_on and mask_pred_result is not None:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)

                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def select_instance_mask_predictions(self, outputs):
        if self.instance_mask_source == "auto":
            mask_pred_results = outputs.get("pred_masks_refined")
            if mask_pred_results is None:
                mask_pred_results = outputs.get("pred_masks")
            return mask_pred_results
        if self.instance_mask_source == "refined":
            return outputs.get("pred_masks_refined")
        if self.instance_mask_source == "coarse":
            return outputs.get("pred_masks")
        raise ValueError(
            f"Unsupported MODEL.MaskDINO.TEST.MASK_INFERENCE_SOURCE: {self.instance_mask_source}"
        )

    def compute_instance_refine_losses(self, outputs, targets):
        outputs_without_aux = {
            key: value
            for key, value in outputs.items()
            if key not in {"aux_outputs", "pred_masks_refined", "pred_mask_embed", "border_logits"}
        }
        indices = self.criterion.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=outputs["pred_masks_refined"].device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        refined_loss_dict = self.criterion.loss_masks(
            {"pred_masks": outputs["pred_masks_refined"]},
            targets,
            indices,
            num_masks,
        )
        return {
            "loss_mask_refine": self.instance_mask_refine_loss_weight * self.instance_mask_refine_mask_weight * refined_loss_dict["loss_mask"],
            "loss_dice_refine": self.instance_mask_refine_loss_weight * self.instance_mask_refine_dice_weight * refined_loss_dict["loss_dice"],
        }

    def compute_border_losses(self, border_logits, batched_inputs):
        losses = {}
        if border_logits is None:
            return losses

        bce_loss = border_logits.new_tensor(0.0)
        dice_loss = border_logits.new_tensor(0.0)
        valid_images = 0
        pos_weight = torch.tensor([self.border_pos_weight], device=border_logits.device)

        for pred_per_image, input_per_image in zip(border_logits, batched_inputs):
            if "sem_seg" not in input_per_image:
                continue
            target = input_per_image["sem_seg"].to(border_logits.device)
            if target.dim() == 3:
                target = target.squeeze(0)
            h, w = target.shape[-2:]
            pred_per_image = pred_per_image[:, :h, :w]
            target = (target > 0).float().unsqueeze(0)
            bce_loss = bce_loss + F.binary_cross_entropy_with_logits(
                pred_per_image,
                target,
                pos_weight=pos_weight,
            )
            dice_loss = dice_loss + self.binary_dice_loss(pred_per_image, target)
            valid_images += 1

        if valid_images == 0:
            return losses

        normalizer = float(valid_images)
        losses["loss_border_bce"] = self.border_loss_weight * self.border_bce_weight * (bce_loss / normalizer)
        losses["loss_border_dice"] = self.border_loss_weight * self.border_dice_weight * (dice_loss / normalizer)
        return losses

    @staticmethod
    def binary_dice_loss(logits, targets, eps=1e-6):
        probs = logits.sigmoid()
        intersection = (probs * targets).sum()
        denominator = probs.sum() + targets.sum()
        return 1.0 - (2.0 * intersection + eps) / (denominator + eps)

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                }
            )
        return new_targets

    def prepare_targets_detr(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        pred_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            pred_box_result = pred_box_result[keep]
        if self.instance_box_source == "mask":
            result.pred_boxes = BitMasks(result.pred_masks > 0.5).get_bounding_boxes()
        elif self.instance_box_source == "pred":
            result.pred_boxes = Boxes(pred_box_result)
        else:
            raise ValueError(
                f"Unsupported MODEL.MaskDINO.TEST.BOX_INFERENCE_SOURCE: {self.instance_box_source}"
            )

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


