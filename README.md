# Nikkei 225 Seasonality Analysis System

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pages](https://img.shields.io/badge/Dashboard-GitHub%20Pages-blue)](https://kafka2306.github.io/nk225seasonality/)

> **日本株市場の季節性パターン検出とバリュエーション分析のためのプロフェッショナル向け定量的金融プラットフォーム**

## 🚀 概要

本システムは、日経225指数の高度な統計的検定と堅牢なバリュエーションモデルを組み合わせた、機関投資家レベルの分析ツールです。市場の歪みや適正価格を精密に分析し、データに基づいた意思決定を行うトレーダーや研究者向けに設計されています。

## ✨ 主な機能

### 📊 市場バリュエーションダッシュボード
金利動向に基づいた市場の割安・割高をリアルタイムで評価します。
- **イールドギャップ分析**: 株式益利回りとJGB（日本国債）利回りを瞬時に比較。
- **適正PERモデリング**: リスクプレミアムに基づいた「適正PER」を算出。
- **バリュエーション判定**: 「割高/割安」のシグナルと乖離率を可視化。

### 📅 高度な季節性分析エンジン
統計的に有意な繰り返しパターンを検出・検証します。
- **月次・四半期季節性**: 「セル・イン・メイ」や「掉尾の一振（年末ラリー）」などの強力なトレンドを特定。
- **年度末効果**: 3月末の機関投資家のリバランスによる影響を定量化。
- **メカニズム分析**: 観測された市場アノマリーの要因を分析。

### 🛡️ クリーンでモダンなアーキテクチャ
信頼性と使いやすさを追求した最新のPython標準で構築されています。
- **クリーンアーキテクチャ**: ルートディレクトリを最小限に抑え、関心事を分離。
- **最新のツール群**: `uv` による高速な依存関係管理、`ruff` によるコード品質維持。
- **自動化ワークフロー**: `Taskfile` によるワンコマンド操作。

## 🛠️ クイックスタート

### 前提条件
- **Python 3.10+**
- **uv** (最新のPythonパッケージインストーラー)

### インストール
```bash
task setup
```

### 📉 バリュエーション分析
現在の市場が割安か割高かを分析します。
```bash
# デフォルト設定で実行（JGB利回りはYahoo Financeから自動取得）
task valuation

# シナリオ分析（PERのみ指定）
task valuation PER=19.75
```

### 📈 時系列バリュエーション分析
過去の市場バリュエーションの変化を時系列で分析します。
```bash
# 過去5年間の月次バリュエーション推移
task valuation-ts YEARS=5

# 出力例: 月次PER、適正PER、乖離率、割安/割高判定
# （JGB利回りは取得時点の最新値を使用）
```

### 🗓️ 季節性分析
過去のデータに基づいて統計的検定を実行します。
```bash
# 過去5年分の分析を実行
uv run python main.py seasonality --years 5
```

## 🏗️ アーキテクチャ

```
.
├── src/
│   ├── analysis/       # バリュエーションおよび統計モデル
│   ├── data/           # データ収集・検証パイプライン
│   ├── options/        # オプション価格計算・戦略ロジック
│   ├── risk/           # モンテカルロ・VaRエンジン
│   └── visualization/  # 描画・レポーティング
├── tests/              # 網羅的なテストスイート
├── scripts/            # ユーティリティスクリプト
├── main.py             # 統合CLIエントリーポイント
├── pyproject.toml      # プロジェクト設定・依存関係
└── Taskfile.yml        # 自動化スクリプト
```

---
**Built for Quantitative Excellence.**
