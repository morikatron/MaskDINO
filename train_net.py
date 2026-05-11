import os

import cv2
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg, register_coco_instances
from detectron2.data.detection_utils import read_image
DATASET_TRAIN_NAME = 'manga_train'
DATASET_VAL_NAME = 'manga_val'
DATASET_TRAIN_JSON = 'D:/Projects/Manga/data/_datasets/af_240501/train/coco.json'
DATASET_VAL_JSON = 'D:/Projects/Manga/data/_datasets/af_240501/val/coco.json'
DATASET_TRAIN_IMAGE_ROOT = 'D:/Projects/Manga/data/_datasets/af_240501/train'
DATASET_VAL_IMAGE_ROOT = 'D:/Projects/Manga/data/_datasets/af_240501/val'
DATASET_BORDER_TRAIN_NAME = 'manga_border_semantic_train'
DATASET_BORDER_TRAIN_SPLIT_NAME = 'manga_border_semantic_train_split'
DATASET_BORDER_VAL_SPLIT_NAME = 'manga_border_semantic_val_split'
DATASET_BORDER_IMAGE_ROOT = 'D:/Projects/Manga/data/_datasets/semantic_2601/images'
DATASET_BORDER_MASK_ROOT = 'D:/Projects/Manga/data/_datasets/semantic_2601/masks'


def _list_border_dataset_filenames(image_root, mask_root):
    names = []
    for name in sorted(os.listdir(image_root)):
        image_path = os.path.join(image_root, name)
        mask_path = os.path.join(mask_root, name)
        if os.path.isfile(image_path) and os.path.isfile(mask_path):
            names.append(name)
    return names


def _load_border_sem_seg_subset(image_root, mask_root, selected_names):
    selected = set(selected_names)
    dataset = load_sem_seg(mask_root, image_root, gt_ext='png', image_ext='png')
    return [record for record in dataset if os.path.basename(record["file_name"]) in selected]


def _register_border_semantic_dataset(name, image_root, mask_root, selected_names=None):
    if name in DatasetCatalog.list():
        return
    if selected_names is None:
        DatasetCatalog.register(
            name,
            lambda: load_sem_seg(
                mask_root,
                image_root,
                gt_ext='png',
                image_ext='png',
            ),
        )
    else:
        selected_names = tuple(selected_names)
        DatasetCatalog.register(
            name,
            lambda selected_names=selected_names: _load_border_sem_seg_subset(
                image_root,
                mask_root,
                selected_names,
            ),
        )
    MetadataCatalog.get(name).set(
        stuff_classes=['background', 'border'],
        evaluator_type='manga_border_sem_seg',
        ignore_label=255,
        image_root=image_root,
        sem_seg_root=mask_root,
    )

if DATASET_TRAIN_NAME not in DatasetCatalog.list():
    register_coco_instances(DATASET_TRAIN_NAME, {}, DATASET_TRAIN_JSON, DATASET_TRAIN_IMAGE_ROOT)
if DATASET_VAL_NAME not in DatasetCatalog.list():
    register_coco_instances(DATASET_VAL_NAME, {}, DATASET_VAL_JSON, DATASET_VAL_IMAGE_ROOT)
_register_border_semantic_dataset(
    DATASET_BORDER_TRAIN_NAME,
    DATASET_BORDER_IMAGE_ROOT,
    DATASET_BORDER_MASK_ROOT,
)
_border_names = _list_border_dataset_filenames(DATASET_BORDER_IMAGE_ROOT, DATASET_BORDER_MASK_ROOT)
_border_train_split = [name for index, name in enumerate(_border_names) if index % 10 != 0]
_border_val_split = [name for index, name in enumerate(_border_names) if index % 10 == 0]
_register_border_semantic_dataset(
    DATASET_BORDER_TRAIN_SPLIT_NAME,
    DATASET_BORDER_IMAGE_ROOT,
    DATASET_BORDER_MASK_ROOT,
    _border_train_split,
)
_register_border_semantic_dataset(
    DATASET_BORDER_VAL_SPLIT_NAME,
    DATASET_BORDER_IMAGE_ROOT,
    DATASET_BORDER_MASK_ROOT,
    _border_val_split,
)


# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskDINO Training Script based on Mask2Former.
"""
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except BaseException:
    pass

import copy
import itertools
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskDINO
from maskdino import (
    BorderSemSegEvaluator,
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MangaMultiTaskDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    DefaultPredictor,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref

from maskdino.utils.border_inference import predict_border_with_padding


def _is_border_semantic_dataset(dataset_name):
    return dataset_name.startswith("manga_border_semantic")


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        self.freeze_modules_for_border_head(cfg, model)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training

    @staticmethod
    def freeze_modules_for_border_head(cfg, model):
        border_cfg = getattr(cfg.MODEL, "BORDER_HEAD", None)
        if border_cfg is None or not border_cfg.ENABLED:
            return

        if border_cfg.TRAIN_ONLY and not border_cfg.FREEZE_BASE:
            logging.getLogger("detectron2").warning(
                "MODEL.BORDER_HEAD.TRAIN_ONLY=True and FREEZE_BASE=False. "
                "This will update instance-related weights as well."
            )

        if not border_cfg.FREEZE_BASE:
            return

        for param in model.parameters():
            param.requires_grad = False

        border_head = getattr(model.sem_seg_head, "border_head", None)
        if border_head is None:
            raise ValueError("MODEL.BORDER_HEAD.ENABLED is True, but no border head was built.")

        for param in border_head.parameters():
            param.requires_grad = True

        if getattr(border_cfg, "UNFREEZE_PIXEL_DECODER", False):
            pixel_decoder = getattr(model.sem_seg_head, "pixel_decoder", None)
            if pixel_decoder is None:
                raise ValueError(
                    "MODEL.BORDER_HEAD.UNFREEZE_PIXEL_DECODER is True, but no pixel decoder was built."
                )
            for param in pixel_decoder.parameters():
                param.requires_grad = True

        num_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        logger = logging.getLogger("detectron2")
        if getattr(border_cfg, "UNFREEZE_PIXEL_DECODER", False):
            logger.info(
                "Frozen base model except border head and pixel decoder; trainable params: %d",
                num_trainable,
            )
        else:
            logger.info("Frozen base model; trainable border-head params: %d", num_trainable)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "manga_border_sem_seg":
            return BorderSemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "manga_multitask":
            mapper = MangaMultiTaskDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def _colorize_border_prob(prob_map):
    prob_map = np.asarray(prob_map, dtype=np.float32)
    max_value = float(prob_map.max()) if prob_map.size else 0.0
    if max_value <= 0.0:
        heat = np.zeros_like(prob_map, dtype=np.uint8)
    else:
        heat = np.clip((prob_map / max_value) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)


def export_inference_checkpoint(full_checkpoint_path, output_path=None):
    logger = logging.getLogger("detectron2")

    if not os.path.isfile(full_checkpoint_path):
        logger.warning("Full checkpoint not found: %s", full_checkpoint_path)
        return None

    if output_path is None:
        root, ext = os.path.splitext(full_checkpoint_path)
        output_path = root + "_inference" + ext

    checkpoint = torch.load(full_checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        inference_checkpoint = {"model": checkpoint["model"]}
        if "iteration" in checkpoint:
            inference_checkpoint["iteration"] = checkpoint["iteration"]
    else:
        inference_checkpoint = {"model": checkpoint}

    torch.save(inference_checkpoint, output_path)
    logger.info("Saved inference-only checkpoint to %s", output_path)
    return output_path


def export_border_val_predictions(cfg, weights_path):
    if not cfg.MODEL.BORDER_HEAD.ENABLED or len(cfg.DATASETS.TEST) == 0:
        return

    logger = logging.getLogger("detectron2")

    export_cfg = cfg.clone()
    export_cfg.defrost()
    export_cfg.MODEL.WEIGHTS = weights_path
    export_cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    export_cfg.freeze()

    predictor = DefaultPredictor(export_cfg)
    fixed_size = int(getattr(export_cfg.INPUT.BORDER_SEMANTIC, "FIXED_SIZE", 0))
    pad_value = int(getattr(export_cfg.INPUT.BORDER_SEMANTIC, "PAD_VALUE", 255))
    threshold = 0.5

    saved_dirs = []
    for dataset_name in export_cfg.DATASETS.TEST:
        if not _is_border_semantic_dataset(dataset_name):
            continue

        output_dir = os.path.join(cfg.OUTPUT_DIR, "val_predictions", dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        for record in DatasetCatalog.get(dataset_name):
            image = read_image(record["file_name"], format="BGR")
            predictions = predictor(image)

            border_pred = predictions.get("border_sem_seg")
            if fixed_size > 0 and border_pred is not None:
                restored = predict_border_with_padding(predictor, image, fixed_size, pad_value)
                if restored is not None:
                    border_pred = restored

            if border_pred is None:
                continue

            if isinstance(border_pred, torch.Tensor):
                border_pred = border_pred.detach().cpu().numpy()
            if border_pred.ndim == 3:
                border_pred = border_pred[0]
            border_pred = np.clip(border_pred, 0.0, 1.0)

            heatmap = _colorize_border_prob(border_pred)
            heat_overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0.0)
            binary = (border_pred >= threshold).astype(np.uint8) * 255
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            binary_overlay = cv2.addWeighted(image, 0.55, binary_bgr, 0.45, 0.0)
            panel = np.concatenate([image, heat_overlay, binary_overlay], axis=1)

            file_stem = os.path.splitext(os.path.basename(record["file_name"]))[0]
            out_path = os.path.join(output_dir, file_stem + "_panel.png")
            cv2.imwrite(out_path, panel)

        saved_dirs.append(output_dir)

    if saved_dirs:
        logger.info("Saved border validation predictions to %s", ", ".join(saved_dirs))


def run_dual_box_source_eval(cfg, model, args):
    box_sources = list(getattr(cfg.MODEL.MaskDINO.TEST, "BOX_EVAL_SOURCES", ["pred"]))
    if len(box_sources) == 0:
        box_sources = ["pred"]

    if not cfg.MODEL.MaskDINO.TEST.INSTANCE_ON or len(box_sources) == 1:
        results = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            results.update(Trainer.test_with_TTA(cfg, model))
        return results

    box_source_owner = model.module if hasattr(model, "module") else model
    original_box_source = getattr(box_source_owner, "instance_box_source", None)
    merged_results = OrderedDict()
    try:
        for box_source in box_sources:
            eval_cfg = cfg.clone()
            eval_cfg.defrost()
            eval_cfg.MODEL.MaskDINO.TEST.BOX_INFERENCE_SOURCE = box_source
            eval_cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"eval_box_{box_source}")
            eval_cfg.freeze()

            if original_box_source is not None:
                box_source_owner.instance_box_source = box_source

            results = Trainer.test(eval_cfg, model)
            if cfg.TEST.AUG.ENABLED:
                tta_results = Trainer.test_with_TTA(eval_cfg, model)
                results.update(tta_results)

            for dataset_name, dataset_results in results.items():
                if isinstance(dataset_results, dict):
                    merged_results[f"{dataset_name}_box_{box_source}"] = dataset_results
                else:
                    merged_results[f"{dataset_name}_box_{box_source}"] = dataset_results
    finally:
        if original_box_source is not None:
            box_source_owner.instance_box_source = original_box_source

    return merged_results


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = run_dual_box_source_eval(cfg, model, args)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    train_result = trainer.train()
    if comm.is_main_process():
        final_weights = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        inference_weights = export_inference_checkpoint(final_weights)
        export_border_val_predictions(cfg, inference_weights or final_weights)
    return train_result


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
