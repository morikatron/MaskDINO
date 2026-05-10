# 2026-05-10 / 2026-05-11 border evaluator・multitask・box ルート再評価ログ

## 概要

- `border_sem_seg` 専用 evaluator を追加した。
- multitask 学習の問題点を調査し、`task_type` に応じて loss を切り替える修正を入れた。
- `border-only` で `pixel decoder` を非凍結にした比較実験を行った。
- bbox 精度の計算ルートを 2 系統に整理し、5 モデルを同一条件で再評価した。
- box ルート切り替え評価に不具合があり、`cfg` だけでなく実モデル側の `instance_box_source` も切り替えるように修正した。

## 関連ファイル

- [train_net.py](../../train_net.py)
- [maskdino/config.py](../../maskdino/config.py)
- [maskdino/maskdino.py](../../maskdino/maskdino.py)
- [maskdino/evaluation/border_sem_seg_evaluation.py](../../maskdino/evaluation/border_sem_seg_evaluation.py)
- [configs/manga/border_head_semantic_2601_ref.yaml](../../configs/manga/border_head_semantic_2601_ref.yaml)
- [configs/manga/border_head_semantic_2601_ref_unfreeze_pixel_decoder.yaml](../../configs/manga/border_head_semantic_2601_ref_unfreeze_pixel_decoder.yaml)
- [configs/manga/border_head_semantic_2601_multitask.yaml](../../configs/manga/border_head_semantic_2601_multitask.yaml)
- [.codex/memory.md](../../.codex/memory.md)

## bbox 精度の計算ルート

### 1. `pred box`

- decoder / box branch が直接出力した box を使う。
- 実装上は `Boxes(pred_box_result)` を `result.pred_boxes` に入れる。

### 2. `mask-derived box`

- 予測マスクを二値化して、その外接矩形を box として使う。
- 実装上は `BitMasks(result.pred_masks > 0.5).get_bounding_boxes()` を使う。

### 修正した不具合

- 当初の `run_dual_box_source_eval()` は `cfg.MODEL.MaskDINO.TEST.BOX_INFERENCE_SOURCE` だけを書き換えていた。
- しかし `eval_only` では model がすでに構築済みのため、実際には model 内の `instance_box_source` が変わらず、`pred` と `mask` の両評価が同じ経路で走っていた。
- [train_net.py](../../train_net.py) を修正し、評価ループごとに model 側の `instance_box_source` も切り替えるようにした。

## 再評価条件

- bbox / segm の再評価はすべて 512 系の推論条件で統一した。
- `pred box` と `mask-derived box` の両方を同じ run で保存した。
- border 指標は `border_sem_seg` evaluator の結果を用いた。

## 評価対象モデル

### 1. baseline instance

- 重み: [output/251210_baseline/model_final.pth](../../output/251210_baseline/model_final.pth)
- 再評価出力: [output/baseline_eval_box_routes_512_fixed](../../output/baseline_eval_box_routes_512_fixed)

### 2. border-only / frozen

- 設定: [configs/manga/border_head_semantic_2601_ref.yaml](../../configs/manga/border_head_semantic_2601_ref.yaml)
- 重み: [output/border_head_semantic_2601_ref_split_frozen_fixedpad/model_final.pth](../../output/border_head_semantic_2601_ref_split_frozen_fixedpad/model_final.pth)
- border 学習出力: [output/border_head_semantic_2601_ref_split_frozen_fixedpad](../../output/border_head_semantic_2601_ref_split_frozen_fixedpad)
- 再評価出力: [output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed](../../output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed)

### 3. border-only / pixel decoder 非凍結

- 設定: [configs/manga/border_head_semantic_2601_ref_unfreeze_pixel_decoder.yaml](../../configs/manga/border_head_semantic_2601_ref_unfreeze_pixel_decoder.yaml)
- 重み: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/model_final.pth](../../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/model_final.pth)
- border 学習出力: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder](../../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder)
- 再評価出力: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed](../../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed)

### 4. multitask / 旧実装

- 設定: [configs/manga/border_head_semantic_2601_multitask.yaml](../../configs/manga/border_head_semantic_2601_multitask.yaml)
- 重み: [output/border_head_semantic_2601_multitask_eval/model_final.pth](../../output/border_head_semantic_2601_multitask_eval/model_final.pth)
- 学習出力: [output/border_head_semantic_2601_multitask_eval](../../output/border_head_semantic_2601_multitask_eval)
- 再評価出力: [output/border_head_semantic_2601_multitask_eval_box_routes_fixed](../../output/border_head_semantic_2601_multitask_eval_box_routes_fixed)

### 5. multitask / task-gated 修正版

- 設定: [configs/manga/border_head_semantic_2601_multitask.yaml](../../configs/manga/border_head_semantic_2601_multitask.yaml)
- 重み: [output/border_head_semantic_2601_multitask_taskgated/model_final.pth](../../output/border_head_semantic_2601_multitask_taskgated/model_final.pth)
- 学習出力: [output/border_head_semantic_2601_multitask_taskgated](../../output/border_head_semantic_2601_multitask_taskgated)
- 再評価出力: [output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed](../../output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed)

## 比較表

| モデル | border F1 | border IoU | bbox AP (`pred box`) | bbox AP (`mask-derived box`) | segm AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline instance | - | - | 40.2311 | 85.6403 | 89.7121 |
| border-only / frozen | 38.0863 | 23.5226 | 40.2311 | 85.6403 | 89.7121 |
| border-only / pixel decoder 非凍結 | 37.4147 | 23.0123 | 0.7914 | 0.7826 | 0.7216 |
| multitask / 旧実装 | 22.8076 | 12.8717 | 39.9480 | 83.0157 | 87.3049 |
| multitask / task-gated 修正版 | 29.8913 | 17.5719 | 41.8026 | 84.0438 | 88.5807 |

## border 指標ファイル

- border-only / frozen:
  - [output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json](../../output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json)
- border-only / pixel decoder 非凍結:
  - [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json](../../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json)
- multitask / 旧実装:
  - [output/border_head_semantic_2601_multitask_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json](../../output/border_head_semantic_2601_multitask_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json)
- multitask / task-gated 修正版:
  - [output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json](../../output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed/eval_box_mask/inference/manga_border_semantic_val_split_border_metrics.json)

## 追加メモ

- `baseline instance` と `border-only / frozen` が両 box ルートで同値になったことで、「frozen なのに instance head が変わった」のではなく、「以前は評価条件が違った」ことが確認できた。
- `mask-derived box` の bbox AP は、`pred box` より大きく高い。つまり現状の大きな劣化は mask そのものよりも box branch 側に強く出ている。
- `pixel decoder` を非凍結にした `border-only` は、border 指標はそこまで落ちない一方で instance が大きく崩れる。
- `task-gated` 修正版は旧 multitask より改善したが、border-only / frozen と baseline にはまだ届いていない。
