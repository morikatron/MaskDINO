# モデル構造概要台帳

## 運用ルール

- このファイルは border head 専用ではなく、追加・変更したモデル構造全般の概要を残す。
- 構造を変更・追加したときは、新しいセクションを追加する。
- 以前の構造説明は削除しない。
- 新しい構造が以前の構造と互換でない場合は、古いセクションのタイトルに `破棄` を付ける。
- 学習や再評価を追加で回したときは、対応する output と log を該当セクションへ追記する。

## 構造セクション一覧

- [2026-05-11 border head 初版](#2026-05-11-border-head-初版)
- [2026-05-12 instance mask refine 初版 破棄](#2026-05-12-instance-mask-refine-初版-破棄)
- [2026-05-30 instance mask refine frozen-base](#2026-05-30-instance-mask-refine-frozen-base)

## 2026-05-11 border head 初版

### 対象

- 追加した dense border prediction head
- 実装本体: [maskdino/modeling/meta_arch/border_head.py](../maskdino/modeling/meta_arch/border_head.py)
- 組み込み箇所: [maskdino/modeling/meta_arch/maskdino_head.py](../maskdino/modeling/meta_arch/maskdino_head.py)
- loss / 推論出力: [maskdino/maskdino.py](../maskdino/maskdino.py)

### 概要

- 役割: 漫画の枠線 border を 1ch の dense map として予測する。
- 入力:
  - `mask_features`: `B x 256 x 128 x 128`
  - `low_level_features (res2)`: `B x 256 x 128 x 128`
  - `images`: `B x 3 x 512 x 512`
- 出力:
  - `border_logits`: `B x 1 x 512 x 512`
  - 推論時は `sigmoid` 後の `border_sem_seg`
- 学習 loss:
  - `loss_border_bce`
  - `loss_border_dice`

### 主な層とサイズ

| 名前 | 主な構造 | 入力サイズ | 出力サイズ |
| --- | --- | --- | --- |
| `mask_proj` | `Conv 3x3 256->256` + `GN` + `ReLU` | `B x 256 x 128 x 128` | `B x 256 x 128 x 128` |
| `low_level_proj` | `Conv 1x1 256->256` + `GN` + `ReLU` | `B x 256 x 128 x 128` | `B x 256 x 128 x 128` |
| `image_s4_stem` | `Conv 3x3 3->64 s2` + `GN` + `ReLU` x2 | `B x 3 x 512 x 512` | `B x 64 x 128 x 128` |
| `fuse_s4` | `Conv 3x3 576->256` + `GN` + `ReLU` x2 | `B x 576 x 128 x 128` | `B x 256 x 128 x 128` |
| `image_s2_stem` | `Conv 3x3 3->128 s2` + `GN` + `ReLU` | `B x 3 x 512 x 512` | `B x 128 x 256 x 256` |
| `refine_s2` | `Conv 3x3 384->128` + `GN` + `ReLU` x2 | `B x 384 x 256 x 256` | `B x 128 x 256 x 256` |
| `image_s1_stem` | `Conv 3x3 3->64` + `GN` + `ReLU` | `B x 3 x 512 x 512` | `B x 64 x 512 x 512` |
| `refine_s1` | `Conv 3x3 192->64` + `GN` + `ReLU` x2 | `B x 192 x 512 x 512` | `B x 64 x 512 x 512` |
| `predictor` | `Conv 1x1 64->1` | `B x 64 x 512 x 512` | `B x 1 x 512 x 512` |

### Mermaid 図

`pixel encoder` 自体は省略し、border head が受ける入力から描く。

```mermaid
flowchart TD
    MF["mask_features\nB x 256 x 128 x 128"] --> MP["mask_proj\nConv 3x3 256->256\nGN + ReLU\nB x 256 x 128 x 128"]
    LL["low_level_features (res2)\nB x 256 x 128 x 128"] --> LP["low_level_proj\nConv 1x1 256->256\nGN + ReLU\nB x 256 x 128 x 128"]
    IMG["images\nB x 3 x 512 x 512"] --> IS4["image_s4_stem\nConv 3x3 3->64 s2\nGN + ReLU\nConv 3x3 64->64 s2\nGN + ReLU\nB x 64 x 128 x 128"]
    IMG --> IS2["image_s2_stem\nConv 3x3 3->128 s2\nGN + ReLU\nB x 128 x 256 x 256"]
    IMG --> IS1["image_s1_stem\nConv 3x3 3->64\nGN + ReLU\nB x 64 x 512 x 512"]

    MP --> CAT4["concat\n256 + 256 + 64 = 576\nB x 576 x 128 x 128"]
    LP --> CAT4
    IS4 --> CAT4

    CAT4 --> F4["fuse_s4\nConv 3x3 576->256\nGN + ReLU\nConv 3x3 256->256\nGN + ReLU\nB x 256 x 128 x 128"]
    F4 --> UP2["upsample\n128 -> 256"]
    UP2 --> CAT2["concat\n256 + 128 = 384\nB x 384 x 256 x 256"]
    IS2 --> CAT2

    CAT2 --> R2["refine_s2\nConv 3x3 384->128\nGN + ReLU\nConv 3x3 128->128\nGN + ReLU\nB x 128 x 256 x 256"]
    R2 --> UP1["upsample\n256 -> 512"]
    UP1 --> CAT1["concat\n128 + 64 = 192\nB x 192 x 512 x 512"]
    IS1 --> CAT1

    CAT1 --> R1["refine_s1\nConv 3x3 192->64\nGN + ReLU\nConv 3x3 64->64\nGN + ReLU\nB x 64 x 512 x 512"]
    R1 --> P["predictor\nConv 1x1 64->1\nB x 1 x 512 x 512"]
    P --> O["border_logits / border_sem_seg"]
```

### 対応する設定

- border-only / frozen:
  - [configs/manga/border_head_semantic_2601_ref.yaml](../configs/manga/border_head_semantic_2601_ref.yaml)
- border-only / pixel decoder 非凍結:
  - [configs/manga/border_head_semantic_2601_ref_unfreeze_pixel_decoder.yaml](../configs/manga/border_head_semantic_2601_ref_unfreeze_pixel_decoder.yaml)
- multitask:
  - [configs/manga/border_head_semantic_2601_multitask.yaml](../configs/manga/border_head_semantic_2601_multitask.yaml)

### 対応する output フォルダとログ

#### 学習 run

- border-only / frozen:
  - 出力: [output/border_head_semantic_2601_ref_split_frozen_fixedpad](../output/border_head_semantic_2601_ref_split_frozen_fixedpad)
  - checkpoint: [output/border_head_semantic_2601_ref_split_frozen_fixedpad/model_final.pth](../output/border_head_semantic_2601_ref_split_frozen_fixedpad/model_final.pth)
  - inference checkpoint: [output/border_head_semantic_2601_ref_split_frozen_fixedpad/model_final_inference.pth](../output/border_head_semantic_2601_ref_split_frozen_fixedpad/model_final_inference.pth)
  - log: [output/border_head_semantic_2601_ref_split_frozen_fixedpad/log.txt](../output/border_head_semantic_2601_ref_split_frozen_fixedpad/log.txt)

- border-only / pixel decoder 非凍結:
  - 出力: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder](../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder)
  - checkpoint: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/model_final.pth](../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/model_final.pth)
  - inference checkpoint: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/model_final_inference.pth](../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/model_final_inference.pth)
  - log: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/log.txt](../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder/log.txt)

- multitask / 旧実装:
  - 出力: [output/border_head_semantic_2601_multitask_eval](../output/border_head_semantic_2601_multitask_eval)
  - checkpoint: [output/border_head_semantic_2601_multitask_eval/model_final.pth](../output/border_head_semantic_2601_multitask_eval/model_final.pth)
  - inference checkpoint: [output/border_head_semantic_2601_multitask_eval/model_final_inference.pth](../output/border_head_semantic_2601_multitask_eval/model_final_inference.pth)
  - log: [output/border_head_semantic_2601_multitask_eval/log.txt](../output/border_head_semantic_2601_multitask_eval/log.txt)

- multitask / task-gated 修正版:
  - 出力: [output/border_head_semantic_2601_multitask_taskgated](../output/border_head_semantic_2601_multitask_taskgated)
  - checkpoint: [output/border_head_semantic_2601_multitask_taskgated/model_final.pth](../output/border_head_semantic_2601_multitask_taskgated/model_final.pth)
  - inference checkpoint: [output/border_head_semantic_2601_multitask_taskgated/model_final_inference.pth](../output/border_head_semantic_2601_multitask_taskgated/model_final_inference.pth)
  - log: [output/border_head_semantic_2601_multitask_taskgated/log.txt](../output/border_head_semantic_2601_multitask_taskgated/log.txt)

#### 再評価 run

- baseline / box 2 ルート再評価:
  - 出力: [output/baseline_eval_box_routes_512_fixed](../output/baseline_eval_box_routes_512_fixed)
  - log: [output/baseline_eval_box_routes_512_fixed/log.txt](../output/baseline_eval_box_routes_512_fixed/log.txt)
  - baseline inference checkpoint: [output/251210_baseline/model_final_inference.pth](../output/251210_baseline/model_final_inference.pth)

- border-only / frozen / box 2 ルート再評価:
  - 出力: [output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed](../output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed)
  - log: [output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed/log.txt](../output/border_head_semantic_2601_ref_split_frozen_fixedpad_eval_box_routes_fixed/log.txt)

- border-only / pixel decoder 非凍結 / box 2 ルート再評価:
  - 出力: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed](../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed)
  - log: [output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed/log.txt](../output/border_head_semantic_2601_ref_unfreeze_pixel_decoder_eval_box_routes_fixed/log.txt)

- multitask / 旧実装 / box 2 ルート再評価:
  - 出力: [output/border_head_semantic_2601_multitask_eval_box_routes_fixed](../output/border_head_semantic_2601_multitask_eval_box_routes_fixed)
  - log: [output/border_head_semantic_2601_multitask_eval_box_routes_fixed/log.txt](../output/border_head_semantic_2601_multitask_eval_box_routes_fixed/log.txt)

- multitask / task-gated 修正版 / box 2 ルート再評価:
  - 出力: [output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed](../output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed)
  - log: [output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed/log.txt](../output/border_head_semantic_2601_multitask_taskgated_eval_box_routes_fixed/log.txt)

### 精度評価表

| モデル | border F1 | border IoU | bbox AP (`pred box`) | bbox AP (`mask-derived box`) | segm AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline instance | - | - | 40.2311 | 85.6403 | 89.7121 |
| border-only / frozen | 38.0863 | 23.5226 | 40.2311 | 85.6403 | 89.7121 |
| border-only / pixel decoder 非凍結 | 37.4147 | 23.0123 | 0.7914 | 0.7826 | 0.7216 |
| multitask / 旧実装 | 22.8076 | 12.8717 | 39.9480 | 83.0157 | 87.3049 |
| multitask / task-gated 修正版 | 29.8913 | 17.5719 | 41.8026 | 84.0438 | 88.5807 |

## 2026-05-12 instance mask refine 初版 破棄

### 対象

- 初回の instance mask refine 試行
- ユーザー指摘により、この試行は「MaskDINO 本体まで更新していた」ため破棄
- 誤設定に紐づく設定ファイル、出力フォルダ、個別ログは削除済み

### 概要

- 初回試行は構造自体ではなく学習条件が不適切だった。
- 正しい isolated 学習結果は後続の `2026-05-30 instance mask refine frozen-base` を参照。

### 精度評価表

| モデル | 状態 |
| --- | --- |
| instance refine half-input | 破棄。誤って MaskDINO 本体も更新していたため削除済み。 |

## 2026-05-30 instance mask refine frozen-base

### 対象

- MaskDINO 本体を凍結し、instance refine head のみを学習する構成
- リファイン head 本体: [maskdino/modeling/meta_arch/instance_mask_refine_head.py](../maskdino/modeling/meta_arch/instance_mask_refine_head.py)
- 組み込み箇所:
  - [maskdino/modeling/meta_arch/maskdino_head.py](../maskdino/modeling/meta_arch/maskdino_head.py)
  - [maskdino/modeling/transformer_decoder/maskdino_decoder.py](../maskdino/modeling/transformer_decoder/maskdino_decoder.py)
  - [maskdino/maskdino.py](../maskdino/maskdino.py)
  - [maskdino/config.py](../maskdino/config.py)
  - [train_net.py](../train_net.py)

### 概要

- 役割: 半分サイズで動かすバックボーンの coarse instance mask を、原寸画像を見ながら原寸マスクへリファインする。
- 学習時に border データは使わず、border head も学習しない。
- 学習時の入力は mapper からは原寸側で入る。現在の設定では LSJ の `IMAGE_SIZE=1024` が基準。
- モデル内部で:
  - 原寸画像 `images_fullres` を保持する。
  - バックボーン入力だけを `BACKBONE_DOWNSCALE_FACTOR=0.5` で縮小し、`512 x 512` 相当で backbone / pixel decoder / transformer decoder を回す。
  - coarse mask は半分入力側の特徴から予測する。
  - 追加の refine head は `mask_features`、`res2`、原寸画像を受け取り、原寸の共有 mask feature を作る。
  - decoder の `pred_mask_embed` と共有 mask feature の積で query ごとの residual mask を作る。
  - 最終出力は「原寸へ拡大した coarse mask + residual mask」。
- 学習対象:
  - `instance_mask_refine_head` のみ
  - trainable params: `3,686,848`

### 主な層とサイズ

以下は今回の学習設定で、原寸入力が `1024 x 1024` の場合の代表サイズ。

| 名前 | 主な構造 | 入力サイズ | 出力サイズ |
| --- | --- | --- | --- |
| `images_fullres` | 原寸 RGB 画像 | `B x 3 x 1024 x 1024` | `B x 3 x 1024 x 1024` |
| `backbone input` | `0.5` 倍へ縮小 | `B x 3 x 1024 x 1024` | `B x 3 x 512 x 512` |
| `mask_features` | coarse 側共有特徴 | `-` | `B x 256 x 128 x 128` |
| `low_level_features (res2)` | 半分入力 backbone の浅い特徴 | `-` | `B x 256 x 128 x 128` |
| `mask_proj` | `Conv 3x3 256->256` + `GN` + `ReLU` | `B x 256 x 128 x 128` | `B x 256 x 128 x 128` |
| `low_level_proj` | `Conv 1x1 256->256` + `GN` + `ReLU` | `B x 256 x 128 x 128` | `B x 256 x 128 x 128` |
| `image_s4_stem` | `Conv 3x3 3->64 s2` + `GN` + `ReLU` x2 | `B x 3 x 1024 x 1024` | `B x 64 x 256 x 256` |
| `fuse_s4` | `Conv 3x3 576->256` + `GN` + `ReLU` x2 | `B x 576 x 128 x 128` | `B x 256 x 128 x 128` |
| `image_s2_stem` | `Conv 3x3 3->128 s2` + `GN` + `ReLU` | `B x 3 x 1024 x 1024` | `B x 128 x 512 x 512` |
| `refine_s2` | `Conv 3x3 384->128` + `GN` + `ReLU` x2 | `B x 384 x 512 x 512` | `B x 128 x 512 x 512` |
| `image_s1_stem` | `Conv 3x3 3->128` + `GN` + `ReLU` | `B x 3 x 1024 x 1024` | `B x 128 x 1024 x 1024` |
| `refine_s1` | `Conv 3x3 256->128` + `GN` + `ReLU` x2 | `B x 256 x 1024 x 1024` | `B x 128 x 1024 x 1024` |
| `predictor` | `Conv 1x1 128->256` | `B x 128 x 1024 x 1024` | `B x 256 x 1024 x 1024` |
| `pred_mask_embed` | decoder query 埋め込み | `-` | `B x Q x 256` |
| `residual masks` | `einsum(pred_mask_embed, refined_features)` | `B x Q x 256`, `B x 256 x 1024 x 1024` | `B x Q x 1024 x 1024` |
| `pred_masks_refined` | `upsampled coarse + residual` | `B x Q x 1024 x 1024` | `B x Q x 1024 x 1024` |

### Mermaid 図

`pixel encoder` 自体は省略し、instance refine が受ける入力から描く。

```mermaid
flowchart TD
    MF["mask_features\nB x 256 x 128 x 128"] --> MP["mask_proj\nConv 3x3 256->256\nGN + ReLU\nB x 256 x 128 x 128"]
    LL["low_level_features (res2)\nB x 256 x 128 x 128"] --> LP["low_level_proj\nConv 1x1 256->256\nGN + ReLU\nB x 256 x 128 x 128"]
    IMG["full-res images\nB x 3 x 1024 x 1024"] --> IS4["image_s4_stem\nConv 3x3 3->64 s2\nGN + ReLU\nConv 3x3 64->64 s2\nGN + ReLU\nB x 64 x 256 x 256"]
    IMG --> IS2["image_s2_stem\nConv 3x3 3->128 s2\nGN + ReLU\nB x 128 x 512 x 512"]
    IMG --> IS1["image_s1_stem\nConv 3x3 3->128\nGN + ReLU\nB x 128 x 1024 x 1024"]

    MP --> CAT4["concat after align\n256 + 256 + 64 = 576\nB x 576 x 128 x 128"]
    LP --> CAT4
    IS4 --> CAT4

    CAT4 --> F4["fuse_s4\nConv 3x3 576->256\nGN + ReLU\nConv 3x3 256->256\nGN + ReLU\nB x 256 x 128 x 128"]
    F4 --> UP2["upsample\n128 -> 512"]
    UP2 --> CAT2["concat\n256 + 128 = 384\nB x 384 x 512 x 512"]
    IS2 --> CAT2

    CAT2 --> R2["refine_s2\nConv 3x3 384->128\nGN + ReLU\nConv 3x3 128->128\nGN + ReLU\nB x 128 x 512 x 512"]
    R2 --> UP1["upsample\n512 -> 1024"]
    UP1 --> CAT1["concat\n128 + 128 = 256\nB x 256 x 1024 x 1024"]
    IS1 --> CAT1

    CAT1 --> R1["refine_s1\nConv 3x3 256->128\nGN + ReLU\nConv 3x3 128->128\nGN + ReLU\nB x 128 x 1024 x 1024"]
    R1 --> P["predictor\nConv 1x1 128->256\nrefined mask features\nB x 256 x 1024 x 1024"]

    EMB["pred_mask_embed\nB x Q x 256"] --> EINSUM["einsum\nquery-wise residual masks\nB x Q x 1024 x 1024"]
    P --> EINSUM
    COARSE["coarse pred_masks\nB x Q x 128 x 128"] --> UPCOARSE["upsample to full-res\nB x Q x 1024 x 1024"]
    UPCOARSE --> SUM["sum"]
    EINSUM --> SUM
    SUM --> OUT["pred_masks_refined\nB x Q x 1024 x 1024"]
```

### 対応する設定

- 学習設定:
  - [configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml](../configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml)
- 初期重み:
  - [output/251210_baseline/model_final_inference.pth](../output/251210_baseline/model_final_inference.pth)

### 対応する output フォルダとログ

- スモーク:
  - 出力: [output/instance_refine_halfinput_frozen_smoke](../output/instance_refine_halfinput_frozen_smoke)
  - log: [output/instance_refine_halfinput_frozen_smoke/log.txt](../output/instance_refine_halfinput_frozen_smoke/log.txt)

- 本学習:
  - 出力: [output/instance_refine_halfinput_r50_frozen](../output/instance_refine_halfinput_r50_frozen)
  - checkpoint: [output/instance_refine_halfinput_r50_frozen/model_final.pth](../output/instance_refine_halfinput_r50_frozen/model_final.pth)
  - inference checkpoint: [output/instance_refine_halfinput_r50_frozen/model_final_inference.pth](../output/instance_refine_halfinput_r50_frozen/model_final_inference.pth)
  - log: [output/instance_refine_halfinput_r50_frozen/log.txt](../output/instance_refine_halfinput_r50_frozen/log.txt)
  - metrics: [output/instance_refine_halfinput_r50_frozen/metrics.json](../output/instance_refine_halfinput_r50_frozen/metrics.json)
  - Val 予測画像: [output/instance_refine_halfinput_r50_frozen/val_predictions/manga_val](../output/instance_refine_halfinput_r50_frozen/val_predictions/manga_val)

- box 2 ルート再評価:
  - `pred box`: [output/instance_refine_halfinput_r50_frozen/eval_box_pred/inference/coco_instances_results.json](../output/instance_refine_halfinput_r50_frozen/eval_box_pred/inference/coco_instances_results.json)
  - `mask-derived box`: [output/instance_refine_halfinput_r50_frozen/eval_box_mask/inference/coco_instances_results.json](../output/instance_refine_halfinput_r50_frozen/eval_box_mask/inference/coco_instances_results.json)

- mask source + box source 再評価:
  - `auto + pred`: [output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_pred/inference/coco_instances_results.json](../output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_pred/inference/coco_instances_results.json)
  - `auto + mask-derived box`: [output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_mask/inference/coco_instances_results.json](../output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_mask/inference/coco_instances_results.json)
  - `coarse-upsample + pred`: [output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_pred/inference/coco_instances_results.json](../output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_pred/inference/coco_instances_results.json)
  - `coarse-upsample + mask-derived box`: [output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_mask/inference/coco_instances_results.json](../output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_mask/inference/coco_instances_results.json)

### 精度評価表

| モデル | マスク出力 | bbox AP (`pred box`) | bbox AP (`mask-derived box`) | segm AP |
| --- | --- | ---: | ---: | ---: |
| instance refine half-input / frozen-base | `auto` (`pred_masks_refined` 優先) | 43.1657 | 50.2985 | 49.9335 |
| instance refine half-input / frozen-base | `coarse` (本来のマスクヘッドを原寸へ単純拡大) | 43.2244 | 50.8591 | 48.5173 |
