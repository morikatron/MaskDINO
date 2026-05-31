# 2026-05-30 instance refine frozen-base 作業ログ

## 目的

- 前回の unfrozen 試行を破棄し、MaskDINO 本体を凍結したうえで instance refine head だけを学習する。
- border データは使わず、border head も学習しない。

## 前回試行の整理

- 削除した設定:
  - `configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput.yaml`
- 削除した出力:
  - `output/instance_refine_halfinput_r50`
  - `output/instance_refine_halfinput_smoke`
  - `output/instance_refine_halfinput_smoke2`
- 削除した個別ログ:
  - `plans/2026-05-12_instance_refine`
- 構造台帳では、前回試行を `破棄` として整理した。

## 実装変更

- 設定追加:
  - [maskdino/config.py](../../maskdino/config.py)
  - `MODEL.INSTANCE_MASK_REFINE.FREEZE_BASE`
- 凍結処理追加:
  - [train_net.py](../../train_net.py)
  - `freeze_modules_for_instance_refine`
- 学習設定:
  - [configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml](../../configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml)

## 凍結方針

- `MODEL.INSTANCE_MASK_REFINE.ENABLED: True`
- `MODEL.INSTANCE_MASK_REFINE.FREEZE_BASE: True`
- 学習対象:
  - `instance_mask_refine_head` のみ
- 学習対象パラメータ数:
  - `3,686,848`

## スモーク実行

```powershell
conda run -n maskdino python train_net.py --num-gpus 1 --config-file configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml SOLVER.MAX_ITER 1 SOLVER.STEPS [] SOLVER.CHECKPOINT_PERIOD 1 SOLVER.IMS_PER_BATCH 1 DATALOADER.NUM_WORKERS 0 DATASETS.TEST [] TEST.EVAL_PERIOD 0 OUTPUT_DIR ./output/instance_refine_halfinput_frozen_smoke
```

- 結果:
  - 成功
  - `Frozen base model; trainable instance-refine params: 3686848`
  - `loss_mask_refine` / `loss_dice_refine` を確認

## 本学習

```powershell
conda run -n maskdino python train_net.py --num-gpus 1 --config-file configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml DATALOADER.NUM_WORKERS 0
```

- 出力先:
  - [output/instance_refine_halfinput_r50_frozen](../../output/instance_refine_halfinput_r50_frozen)
- 学習時間:
  - [output/instance_refine_halfinput_r50_frozen/log.txt](../../output/instance_refine_halfinput_r50_frozen/log.txt)
  - `Total training time: 0:22:12`

## 再評価

```powershell
conda run -n maskdino python train_net.py --num-gpus 1 --eval_only --config-file configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml DATALOADER.NUM_WORKERS 0 MODEL.WEIGHTS ./output/instance_refine_halfinput_r50_frozen/model_final_inference.pth
```

- `pred box` 再評価:
  - [output/instance_refine_halfinput_r50_frozen/eval_box_pred/inference/coco_instances_results.json](../../output/instance_refine_halfinput_r50_frozen/eval_box_pred/inference/coco_instances_results.json)
- `mask-derived box` 再評価:
  - [output/instance_refine_halfinput_r50_frozen/eval_box_mask/inference/coco_instances_results.json](../../output/instance_refine_halfinput_r50_frozen/eval_box_mask/inference/coco_instances_results.json)

## 主な成果物

- 学習設定:
  - [configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml](../../configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml)
- 最終 checkpoint:
  - [output/instance_refine_halfinput_r50_frozen/model_final.pth](../../output/instance_refine_halfinput_r50_frozen/model_final.pth)
- inference checkpoint:
  - [output/instance_refine_halfinput_r50_frozen/model_final_inference.pth](../../output/instance_refine_halfinput_r50_frozen/model_final_inference.pth)
- ログ:
  - [output/instance_refine_halfinput_r50_frozen/log.txt](../../output/instance_refine_halfinput_r50_frozen/log.txt)
- metrics:
  - [output/instance_refine_halfinput_r50_frozen/metrics.json](../../output/instance_refine_halfinput_r50_frozen/metrics.json)
- Val 予測:
  - [output/instance_refine_halfinput_r50_frozen/val_predictions/manga_val](../../output/instance_refine_halfinput_r50_frozen/val_predictions/manga_val)

## 最終精度

| 条件 | bbox AP (`pred box`) | bbox AP (`mask-derived box`) | segm AP |
| --- | ---: | ---: | ---: |
| instance refine half-input / frozen-base | 43.1657 | 50.2985 | 49.9335 |

## マスク出力別の再評価

- `coarse` 比較は別ログに追記:
  - [plans/2026-05-31_instance_refine_mask_eval/work_log.md](../2026-05-31_instance_refine_mask_eval/work_log.md)

| 条件 | マスク出力 | bbox AP (`pred box`) | bbox AP (`mask-derived box`) | segm AP |
| --- | --- | ---: | ---: | ---: |
| instance refine half-input / frozen-base | `auto` (`pred_masks_refined` 優先) | 43.1657 | 50.2985 | 49.9335 |
| instance refine half-input / frozen-base | `coarse` (本来のマスクヘッドを原寸へ単純拡大) | 43.2244 | 50.8591 | 48.5173 |

## メモ

- 本学習後の通常評価と `pred box` 再評価は一致した。
- `mask-derived box` は `pred box` より高く、最終 `bbox AP` は `50.2985`。
- 前回の unfrozen 試行は混乱防止のため物理削除し、このログに置き換えた。
