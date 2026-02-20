# CLAUDE.md

This file provides guidance to Claude Code ([claude.ai/code](http://claude.ai/code)) when working with code in this repository.

---

## ⚠️ 必須ルール（常に確認）

### ✏️ 作業時のチェックリスト

ファイル解析や移動などの作業をした場合は、**必ず**以下を実行すること：

- [ ] `Experimental_record/{yyyymmdd}.md` に作業内容を記録

  - 概要 (Overview)
  - 作業を行った理由
  - 作業の結果
  - 作業で嵌った部分と解決方法（次からスムーズに参照して行えるように）


### 📝 コード変更時のチェックリスト

コードを変更・作成した場合は、**必ず**以下を実行すること：

- [ ] `Experimental_record/{yyyymmdd}.md` に変更内容を記録

  - 概要 (Overview)
  - 変更元ファイル / 新規作成ファイル
  - Before / After 比較（表形式）
  - 変更理由 (Reason for change)

### 📊 データ解析時のチェックリスト

統計解析やデータ分析を行った場合は、**必ず**以下を実行すること：

- [ ] `Experimental_record/{yyyymmdd}.md` に解析内容を記録

  - 解析目的 (Purpose)
  - 解析手法 (Methods)
  - 主要な結果 (Results)
  - 解釈・考察 (Interpretation)

### 記録テンプレート

```markdown

# {yyyy-mm-dd} 実験記録


## {変更内容のタイトル}


### 概要 (Overview)

{簡潔な説明}


### 変更元ファイル (Source file)

`{ファイルパス}`


### Before / After 比較

| 項目 | Before | After |

|------|--------|-------|

| ... | ... | ... |


### 変更理由 (Reason for change)

{理由}

```

### 解析記録テンプレート

```markdown

# {yyyy-mm-dd} 解析記録


## {解析内容のタイトル}


### 解析目的 (Purpose)

{なぜこの解析を行ったか}


### 解析手法 (Methods)

- 対象データ: {データソース、サンプルサイズ}
- 統計手法: {使用した検定・モデル}
- 使用ツール: {Python, R など}


### 主要な結果 (Results)

{数値結果、有意性など}


### 解釈・考察 (Interpretation)

{臨床的・科学的意義}

```