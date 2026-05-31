# 2026-05-31 instance refine mask source 再評価ログ

## 目的

- instance refine モデルについて、`pred_masks_refined` だけでなく、本来のマスクヘッド `pred_masks` を原寸へ単純拡大した場合の AP も出す。
- 今後も同様の比較ができるよう、評価ルートを追加する。

## 実装変更

- 設定追加:
  - [maskdino/config.py](../../maskdino/config.py)
  - `MODEL.MaskDINO.TEST.MASK_INFERENCE_SOURCE`
  - `MODEL.MaskDINO.TEST.MASK_EVAL_SOURCES`
- 推論切替:
  - [maskdino/maskdino.py](../../maskdino/maskdino.py)
  - `select_instance_mask_predictions`
- eval_only の複数ルート評価:
  - [train_net.py](../../train_net.py)
  - `run_multi_source_instance_eval`
- 対応設定:
  - [configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml](../../configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml)
  - `MASK_EVAL_SOURCES: ["auto", "coarse"]`

## ルート定義

- `auto`
  - `pred_masks_refined` があればそれを使い、なければ `pred_masks`
- `coarse`
  - `pred_masks` を使い、原寸へ単純拡大して評価

## 実行

```powershell
conda run -n maskdino python train_net.py --num-gpus 1 --eval_only --config-file configs/manga/maskdino_R50_bs16_50ep_3s_instance_refine_halfinput_frozen.yaml DATALOADER.NUM_WORKERS 0 MODEL.WEIGHTS ./output/instance_refine_halfinput_r50_frozen/model_final_inference.pth
```

## 出力先

- `auto + pred`:
  - [output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_pred/inference/coco_instances_results.json](../../output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_pred/inference/coco_instances_results.json)
- `auto + mask-derived box`:
  - [output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_mask/inference/coco_instances_results.json](../../output/instance_refine_halfinput_r50_frozen/eval_mask_auto_box_mask/inference/coco_instances_results.json)
- `coarse-upsample + pred`:
  - [output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_pred/inference/coco_instances_results.json](../../output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_pred/inference/coco_instances_results.json)
- `coarse-upsample + mask-derived box`:
  - [output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_mask/inference/coco_instances_results.json](../../output/instance_refine_halfinput_r50_frozen/eval_mask_coarse_box_mask/inference/coco_instances_results.json)

## 結果

| モデル | マスク出力 | bbox AP (`pred box`) | bbox AP (`mask-derived box`) | segm AP |
| --- | --- | ---: | ---: | ---: |
| instance refine half-input / frozen-base | `auto` (`pred_masks_refined` 優先) | 43.1657 | 50.2985 | 49.9335 |
| instance refine half-input / frozen-base | `coarse` (本来のマスクヘッドを原寸へ単純拡大) | 43.2244 | 50.8591 | 48.5173 |

## メモ

- 今回追加した比較で、`coarse` は `segm AP` が `48.5173`、`auto` は `49.9335`。
- つまり refine を通した方が mask AP は `+1.4162` 高い。
- 一方で bbox AP は `coarse + mask-derived box` が `50.8591` と、`auto + mask-derived box` の `50.2985` よりわずかに高い。
