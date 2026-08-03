# Nikkei 225 Seasonality Analysis System

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pages](https://img.shields.io/badge/Dashboard-GitHub%20Pages-blue)](https://kafka2306.github.io/nk225seasonality/)

> **日本株市場の季節性パターン検出と、時点整合的なバリュエーション分析のための定量プラットフォーム**

## 概要

本システムは、日経225指数の季節性検定とバリュエーション分析を統合します。過去の割高・割安判定では、実行日の金利を全期間へ適用せず、各価格観測日以前に取得可能だった10年JGB利回りを使用します。

## 主な機能

### 市場バリュエーション

- **イールドギャップ分析:** 株式益利回りと10年JGB利回りを比較
- **適正PER:** `100 / (JGB利回り + 株式リスクプレミアム)`
- **point-in-time評価:** 各価格日以前の最新JGB観測値をbackward as-of join
- **証跡:** 使用したJGB観測日、欠損件数、ticker、計算方式を保存
- **現在金利シナリオ:** 過去時点評価を上書きせず、`current_rate_revaluation_*` の別系列として出力

対象日より後のJGB値しか存在しない場合は未来値をbackfillせず、`jgb_yield`、`fair_per`、`divergence`を欠損として扱います。

### 季節性分析

- 月次・四半期季節性
- 年度末効果
- 曜日効果とローリング分析
- メカニズム分析

## クイックスタート

### 前提条件

- Python 3.10+
- uv

### インストール

```bash
task setup
```

### 現在時点のバリュエーション

```bash
task valuation
task valuation PER=19.75
```

### 時系列バリュエーション

```bash
task valuation-ts YEARS=5
```

出力列:

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

`fair_per` と `divergence` が正準の時点評価です。`current_rate_revaluation_*` は、現在の金利条件を過去価格へ当てるシナリオであり、正準系列とは分離されます。

### 季節性分析

```bash
uv run python main.py seasonality --years 5
```

季節性パイプラインもCLIと同じ `apply_point_in_time_valuation()` を使用するため、チャートと時系列レポートの計算境界は一致します。

## 検証

```bash
uv run pytest
uv run ruff check .
```

回帰テストでは以下を固定します。

- 月ごとに直前のJGB観測値を選ぶ
- 未来のJGB値を過去へbackfillしない
- 現在金利シナリオを変えても過去のpoint-in-time系列が変わらない
- UTCとJSTの時刻差で観測日の対応がずれない

## アーキテクチャ

```text
.
├── src/
│   ├── analysis/
│   ├── data/
│   ├── options/
│   ├── risk/
│   └── visualization/
├── tests/
├── scripts/
├── main.py
├── pyproject.toml
└── Taskfile.yml
```

---
**Built for Quantitative Excellence.**
