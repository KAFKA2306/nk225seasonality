# CLAUDE.md

> **AIエージェントおよび開発者向けガイド**

## 🧠 コンテキスト
**プロジェクト**: 日経225 季節性分析 & バリュエーションシステム (Nikkei 225 Seasonality Analysis & Valuation System)
**目標**: 厳密な統計的季節性検定と、市場バリュエーションモデル（イールドギャップ/PER）を統合する。
**技術スタック**: Python 3.12+, `uv` (依存関係管理), `ruff` (linter), `pytest`.

## 🛠️ 開発ワークフロー
本プロジェクトでは `uv` と `Taskfile` を使用してワークフローを合理化しています。

### 主要コマンド
- **環境セットアップ**: `task setup` (または `uv sync`)
- **テスト実行**: `task test` (または `uv run pytest`)
- **Lint/フォーマット**: `task validate` (または `uv run ruff check .`)
- **バリュエーション分析**: `task valuation`
- **時系列バリュエーション**: `task valuation-ts YEARS=5 YIELD=3.5`
- **季節性分析**: `uv run python main.py seasonality --years 5`

## 🏗️ アーキテクチャ
- **Root**: 最小限の設定ファイルのみ配置 (`pyproject.toml`, `Taskfile.yml`, `main.py`)。
- **Src**: 全てのロジックは `src/` 配下に配置。
    - `src/analysis/valuation.py`: イールドギャップ/適正PERのコアロジック。
    - `src/analysis/seasonality.py`: 市場パターンの統計的検定。
- **Data**: 入出力データは `data/` および `outputs/` に配置。

## 📝 コーディングガイドライン
1. **ルートディレクトリの最小化**: ルートにファイルを追加しないこと。クリーンに保つ。
2. **型安全性**: 全ての関数シグネチャで `typing` (List, Dict, Optional 等) を使用する。
3. **クリーンコード**: 未使用のインポート、デッドコード、冗長なコメントを削除する。
4. **エラーハンドリング**: CLIのエントリーポイントでは `try/except` ブロックを使用し、ユーザーフレンドリーなエラーを表示する。

## 🚀 最近の変更点
- **コードベースのクリーンアップ**: 全てのPythonファイルから冗長なコメントとdocstringを削除し、可読性を向上。
- **検証プロセスの強化**: `scripts/generate_report.py` の修正と `task validate` による厳格なチェックを通過。
- **ドキュメントの最適化**: `README.md` にステータスバッジを追加し、アーキテクチャ図を更新。
- **時系列バリュエーション分析** (`valuation-ts`) を追加: 過去の月次PER推移と乖離率を表示。
- `ingestion.py` に yfinance インポートを追加。
- "バリュエーションダッシュボード" (イールドギャップ分析) を統合。