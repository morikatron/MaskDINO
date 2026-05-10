# 2026-04-30 作業ログ

## 1. border head の初回実装

- 目的: MaskDINO に dense な border semantic head を追加し、漫画の枠線を予測できるようにする。
- 主な変更ファイル:
  - `train_net.py`
  - `maskdino/config.py`
  - `maskdino/maskdino.py`
  - `maskdino/modeling/meta_arch/maskdino_head.py`
  - `maskdino/modeling/meta_arch/border_head.py`
  - `maskdino/data/dataset_mappers/mask_former_semantic_dataset_mapper.py`
  - `configs/manga/border_head_semantic_2601.yaml`
- データ:
  - `semantic_2601` の画像とマスク
- 出力先:
  - `output/border_head_semantic_2601`
- 最終 checkpoint:
  - `output/border_head_semantic_2601/model_final.pth`
- 結果概要:
  - 学習自体は完走
  - border 予測は非常に弱く、ほぼ空予測に近かった

## 2. 画面表示 viewer の実装と修正

- 目的: 画像保存ではなくウィンドウで予測を確認できるようにする。
- 主な変更ファイル:
  - `demo/window_predictor.py`
- 修正内容:
  - viewer 実行時に `INSTANCE_ON=True` を強制
  - border heatmap を、その画像内の最大応答で正規化
  - 応答が弱い場合の fallback threshold を追加
- 結果概要:
  - インスタンス出力と border 応答を画面上で確認できるようになった

## 3. 参考実装を踏まえた semantic 学習条件の改善

- 参考:
  - 外部の semantic segmentation 学習スクリプト
- 取り入れた主な考え:
  - `512x512` の固定キャンバス
  - 縦横比維持リサイズ
  - 画像側は白 padding
  - リサイズ後のマスクを二値化
- 主な変更ファイル:
  - `maskdino/config.py`
  - `maskdino/data/dataset_mappers/mask_former_semantic_dataset_mapper.py`
  - `configs/manga/border_head_semantic_2601_ref.yaml`
- 出力先:
  - `output/border_head_semantic_2601_ref`
- 最終 checkpoint:
  - `output/border_head_semantic_2601_ref/model_final.pth`
- 結果概要:
  - 初回より border 応答はかなり改善
  - ただし `FREEZE_BASE=False` により instance 関連重みまで更新され、実運用用の run としては不採用

## 4. base freeze 修正を入れた semantic 学習

- 目的: semantic 学習条件の改善は残しつつ、instance 予測を壊さないようにする。
- 主な変更ファイル:
  - `train_net.py`
  - `configs/manga/border_head_semantic_2601_ref.yaml`
- 修正内容:
  - `MODEL.BORDER_HEAD.FREEZE_BASE=True` に変更
  - 出力先を新しい run に変更
  - `TRAIN_ONLY=True` かつ `FREEZE_BASE=False` のとき警告を出すようにした
- 実行コマンド:
  - `conda run -n maskdino python train_net.py --config-file configs/manga/border_head_semantic_2601_ref.yaml --num-gpus 1 DATALOADER.NUM_WORKERS 0`
- 出力先:
  - `output/border_head_semantic_2601_ref_frozen`
- 最終 checkpoint:
  - `output/border_head_semantic_2601_ref_frozen/model_final.pth`
- 確認:
  - `00006.jpg.png` で baseline と比較し、`scores/classes/boxes/masks` は一致
  - border 応答は sample 上では強かった
- 学習メモ:
  - 最終ログ付近では sparse mask 特有の不安定さが残った
  - sample 推論は良好でも、loss の見え方だけでは品質を判断しづらかった

## 5. Val 出力の異常と padding バグ調査

- ユーザー報告:
  - border 予測が不自然な縦線ばかりに見える
- 調査結果:
  - 縦横比崩れ自体は起きていなかった
  - 決定的な原因は `MaskFormerSemanticDatasetMapper` の `size_divisibility` padding 計算バグ
  - `32 - width`, `32 - height` のような値になっており、`512x512` 学習サンプルが実質 `32x32` まで切り詰められていた
- 主な修正ファイル:
  - `maskdino/data/dataset_mappers/mask_former_semantic_dataset_mapper.py`
- 確認:
  - 修正後の semantic sample size は `(3, 512, 512)` になった

## 6. padding 修正後の高解像度 border head 再学習

- 使用 config:
  - `configs/manga/border_head_semantic_2601_ref.yaml`
- 出力先:
  - `output/border_head_semantic_2601_ref_split_frozen_fixedpad`
- 最終 checkpoint:
  - `output/border_head_semantic_2601_ref_split_frozen_fixedpad/model_final.pth`
- Val 予測保存先:
  - `output/border_head_semantic_2601_ref_split_frozen_fixedpad/val_predictions`
- 学習内容:
  - 高解像度 refine 型 border head
  - border target dilation を有効化
  - 3000 iter 実行
- 最終 loss 付近:
  - `loss_border_bce=0.001187`
  - `loss_border_dice=0.006871`
  - `total_loss=0.007961`
- 重要メモ:
  - 当時の `metrics.json` にある `sem_seg/*` は `border_sem_seg` の品質指標ではなく、border head の評価には使えなかった
  - border の確認には `val_predictions` を用いた
- 定性的結果:
  - 以前の「縦辺だけ」から改善し、上辺・下辺も含む矩形らしい予測が出るようになった
- instance 確認:
  - 比較元: `output/251210_baseline/model_final.pth`
  - `scores_equal=True`
  - `classes_equal=True`
  - `boxes_equal=True`
  - `masks_equal=True`

## 7. multitask 学習の足場

- 追加したファイル:
  - `maskdino/data/dataset_mappers/manga_multitask_dataset_mapper.py`
  - `configs/manga/border_head_semantic_2601_multitask.yaml`
- 目的:
  - instance と border を同じ base model 上で同時学習できるようにする
- 備考:
  - この時点では足場だけ用意し、最終的に有効だった run はまだ border-only 側
