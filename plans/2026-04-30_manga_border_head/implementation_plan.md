# Manga Border Head 実装計画

## 目標

既存のインスタンスセグメンテーション学習済み checkpoint `output/251210_baseline/model_final.pth` を土台にして、現在の MaskDINO fork に dense な border 予測 head を追加して学習する。

今回のタスク範囲:

- 新しい head を実装する
- まずは新しい head を中心に学習する
- `semantic_2601` の border マスクだけを使う
- インスタンスマスク後処理はまだ実装しない

## 現時点の前提

- インスタンス学習用データセット登録は `train_net.py` 冒頭にある。
- 既存の学習済み checkpoint は `output/251210_baseline/` にある。
- `semantic_2601` には `images/` と `masks/` がある。
- border マスクは主に `0` と `255` の PNG で表現されている。
- ローカル学習環境は conda 環境 `maskdino` である。

## 問題の再整理

狙いは次の 3 点を同時に満たすこと。

1. 現在のインスタンスマスクよりも pixel 単位で border 精度を上げる
2. インスタンスモデルを縮小入力で動かしても、元解像度側へ戻すための構造情報を持てるようにする
3. 枠線欠けのような曖昧領域では柔軟さを残しつつ、下流では規則的な形状を作りやすくする

## 代替案メモ

### A. インスタンスマスクのみを後処理で整形する

モルフォロジー、輪郭スナップ、グラフ最適化などで既存マスクだけを整形する案。

長所:

- モデル変更が小さい
- 試行は速い

短所:

- 欠けた枠線などの構造情報を新たに持てない
- 曖昧領域に弱い
- 不確実性を後処理ヒューリスティクスに押し込みすぎる

結論:

- 後で役立つが、最初の一手としては不足

### B. 距離変換回帰 head

border 二値マスクではなく、距離場を予測する案。

長所:

- 細線に対して勾配が安定しやすい
- 後段の輪郭正則化と相性が良い

短所:

- target 生成が増える
- dense branch を動かす前段としては複雑

結論:

- 将来拡張として有力だが、初回実装には重い

### C. 低解像度 trunk + 高解像度 crop refiner

MaskDINO 本体は縮小入力で動かし、候補領域だけ元画像 crop から別ネットで refine する案。

長所:

- 元解像度に近い出力へ最も伸びやすい
- 計算を必要領域に集中できる

短所:

- 候補抽出と refiner の統合が必要
- 実装面積が大きい

結論:

- 本命になりうるが、今回のスコープには広すぎる

### D. pixel decoder から分岐する dense border head

MaskDINO の pixel decoder から別分岐の dense head を出し、border semantic map を予測する案。

長所:

- ユーザー提案の「別 head を足す」に近い
- query ベース instance branch に押し込まず構造 supervision を入れられる
- 既存 checkpoint から比較的導入しやすい
- 後段のインスタンス整形にもつなげやすい

短所:

- 初版は学習パイプライン解像度の制約を受ける
- 真の元解像度 refine は別段必要

結論:

- 今回の第一段として最適

## 採用した実装方針

### 今回のフェーズ

border 専用の dense semantic head を追加する。

- 入力:
  - pixel decoder の mask features
  - backbone の `res2`
  - モデルに入った RGB 画像 tensor
- 出力:
  - 1 channel の border logit map
- 学習:
  - BCE-with-logits
  - soft Dice loss
- 最適化:
  - 既存の backbone / pixel decoder / transformer decoder / instance branch は必要に応じて固定
  - 新しい border head を重点的に学習

### この形にした理由

pixel decoder には多段の構造情報があり、`res2` と入力画像は局所的な edge 情報を持つ。query ベース branch でぼやけやすい輪郭を、この組み合わせで補う狙い。

## 解像度方針

### 当面の実装

まずは semantic mapper が作る学習解像度に揃った dense map を予測し、loss 計算時にそのサイズで比較する。

小画像への対策として:

- `semantic_2601` に対して疑似 upscale を mapper 側で扱えるようにする
- その上で通常 transform に通す

### 将来拡張

次の段階では以下が自然。

1. trunk は縮小解像度のまま維持
2. 元画像に近い高解像度参照を保持
3. border feature を upsample して image-guided refiner で整える
4. refined border map を panel-mask 正則化の境界場として使う

## データ解釈

`semantic_2601` では:

- `0` = background / non-border
- `255` = border

学習時に `255 -> 1` へ変換し、binary target として loss に与える。

## 実装項目

1. border head 用 config を追加
2. `train_net.py` で border semantic dataset を登録
3. semantic mapper を拡張し、以下に対応
   - `255 -> 1` 変換
   - 任意の疑似 upscale
4. 新しい border head module を追加
5. `MaskDINOHead` に接続
6. `MaskDINO` 本体で border loss を計算
7. 新 head のみを更新できる freeze 経路を追加
8. 学習 config を追加

## 学習計画

ベース重み:

- `output/251210_baseline/model_final.pth`

学習モード:

- border head を有効化
- まずは border head を中心に学習
- この段階では後処理未実装

## このタスクの期待成果

- 新しい border head のコード
- dataset 登録と学習 config
- baseline checkpoint から開始した学習 run
- `output/` 配下に保存された成果物

## 先送りにした項目

- border 誘導による panel instance 後処理
- 余白 branch
- border + 余白 + キャラを含む full panoptic branch
- 低解像度 trunk 後に元画像参照で refine する高解像度 branch
