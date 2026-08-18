# Nikkei 225 Seasonality Analysis

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pages](https://img.shields.io/badge/Dashboard-GitHub%20Pages-blue)](https://kafka2306.github.io/nk225seasonality/)

日経225指数の季節性検定とバリュエーション分析を行うPythonプロジェクトです。過去の評価では、実行日の金利を全期間へ適用せず、各価格観測日以前に取得可能だった10年JGB利回りを使用します。

## 主な機能

### 市場バリュエーション

- イールドギャップ分析: 株式益利回りと10年JGB利回りを比較
- 適正PER: `100 / (JGB利回り + 株式リスクプレミアム)`
- point-in-time評価: 各価格日以前の最新JGB観測値をbackward as-of join
- 証跡: 使用したJGB観測日、欠損件数、ticker、計算方式を保存
- 現在金利シナリオ: `current_rate_revaluation_*` の別系列として出力

対象日より後のJGB値しか存在しない場合は未来値をbackfillせず、`jgb_yield`、`fair_per`、`divergence`を欠損として扱います。

### 季節性分析

- 月次・四半期季節性
- 年度末効果
- 曜日効果とローリング分析
- メカニズム分析

## セットアップ

前提: Python 3.10+、`uv`

```bash
uv sync
```

## 実行

現在時点のバリュエーション:

```bash
uv run python main.py valuation
uv run python main.py valuation --current-per 19.75
```

時系列バリュエーション:

```bash
task valuation-ts YEARS=5
```

季節性分析:

```bash
task seasonality YEARS=5
```

`main.py` では `valuation`、`valuation-ts`、`seasonality` の3 subcommandを提供します。

## バリュエーション出力

```text
price
estimated_eps
estimated_per
jgb_yield
jgb_observed_at
fair_per
divergence
valuation_status
valuation_method
current_rate_revaluation_jgb_yield
current_rate_revaluation_fair_per
current_rate_revaluation_divergence
```

`fair_per` と `divergence` がpoint-in-time評価です。`current_rate_revaluation_*` は現在の金利条件を過去価格へ当てる別シナリオです。

季節性パイプラインもCLIと同じ `apply_point_in_time_valuation()` を使用します。

## 検証

```bash
uv run pytest
uv run ruff check .
```

回帰テストでは、直前のJGB観測値の選択、未来値のbackfill禁止、現在金利シナリオとの分離、UTC/JST境界を検証します。

## 構成

```text
.
├── src/
├── tests/
├── scripts/
├── docs/
├── main.py
├── pyproject.toml
└── Taskfile.yml
```
