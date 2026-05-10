# プロジェクトメモ

## ドキュメント運用ルール

- ドキュメントは日本語で記述する。
- 実装、修正、学習、再評価を行ったら、`plans/` 配下に Markdown を残す。
- 構造を変更・追加したときは、モデル概要ドキュメントにも追記する。
- `plans/model_summary.md` は border head 専用ではなく、追加・変更したモデル構造全般の概要台帳として使う。
- 構造変更を記録するときは、既存内容を上書きせず、新しいセクションを追加する。
- 新しい構造が旧構造と互換でない場合は、旧セクションのタイトルに `破棄` を明記する。
- `plans/model_summary.md` の各構造セクションには、その構造に対応する主な精度評価表も載せる。
- 学習や再評価を行ったときは、モデル概要ドキュメントの対応する出力・ログ一覧にも追記する。
- 学習や再評価を行った場合は、少なくとも次を記録する。
  - 使用した設定ファイル
  - 使用した重み
  - 出力先フォルダ
  - 最終 checkpoint
  - Val 予測や評価結果の保存先
  - 試行の要点と採否
- 後から不採用になった試行も削除せず、何が問題だったかを明記して残す。

## Markdown 内のパス表記ルール

- 設定、モデル、出力フォルダ、ログ、画像などのパスは、Markdown では相対リンクで記載する。
- 表示テキストは「プロジェクトルート基準の相対パス」にする。
- 実際のリンク先は、その Markdown ファイルの場所から辿れる相対リンクにする。
- 例:
  - `[configs/manga/example.yaml](../configs/manga/example.yaml)`
  - `[output/example_run/model_final.pth](../output/example_run/model_final.pth)`

## 現在の主な記録

- border head 初回設計メモ: [plans/2026-04-30_manga_border_head/implementation_plan.md](../plans/2026-04-30_manga_border_head/implementation_plan.md)
- 2026-04-30 作業ログ: [plans/2026-04-30_work_log/work_log.md](../plans/2026-04-30_work_log/work_log.md)
- 2026-05-10 / 2026-05-11 border evaluator と multitask 再評価ログ: [plans/2026-05-10_border_eval_multitask/work_log.md](../plans/2026-05-10_border_eval_multitask/work_log.md)
- モデル構造と対応出力の概要台帳: [plans/model_summary.md](../plans/model_summary.md)
