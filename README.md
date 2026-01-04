
モリカコミックでコマのセグメンテーション用に、MASKDINOを学習して使っている。

## DOCS

- [Original README](README_ORIGINAL.md)
- [installation instructions](INSTALL.md)


## 学習手順概要

- COCO 形式のデータセット作成、任意の場所に置く。
- `train_net.py` 冒頭で、データセットの情報を設定。
- `configs/manga/`にて、config ファイルを追加。
- `train_net.py` を実行して学習。
```
python train_net.py --num-gpus 1 --config-file configs\manga\boundary_finetune.yaml
```
yaml でもベース重みを指定できますが、コマンドラインで指定するのも可。その場合は上記コマンドの最後に　`MODEL.WEIGHTS D:\model.pth` みたいな設定を追加。


### config について

#### 履歴

- 現在使われているモデルは　`maskdino_R50_bs16_50ep_3s.yaml`（`base.yaml`を継承）で学習されている。
- `backup_maskdino_231120.yaml` は学習回したときに出力された既定値を含めた完全版 configで、念の為置いた、学習には使わない。
- `boundaryXXX.yaml`: boundary loss で学習用の config。
  - `boundary_finetune2.yaml`: 現在モデルをベースに、boundary loss 追加した形でファイチューニングする。

#### よく使う設定

- データセットの設定：`train_net.py` 冒頭で定義した名前で設定。
　`base.yaml` では以下のように指定しているが、継承する場合は自ファイルで別に設定してオーバーライト。
```
DATASETS:
  TRAIN: ("manga_train",)
  TEST: ("manga_val",)
```

- 追加学習の場合は、以下のように設定。
```
MODEL:
  WEIGHTS: "output/251210_baseline/model_final.pth"
```

- 出力場所。任意だが以下が一例。
```
OUTPUT_DIR: ./output/boundary_finetune2
```

- 学習回数、学習率
```
SOLVER:
  BASE_LR: 1.0e-05
  MAX_ITER: 10000
```
