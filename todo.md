# TODOリスト

## フェーズ1: プロジェクト構造の準備

- [x] `ROP_project`ディレクトリを作成する
- [x] `ROP_project`内に、以下のサブディレクトリを作成する:
    - `data/`
    - `notebooks/`
    - `scripts/`
    - `src/`
        - `utils/`
- [x] 既存のJupyter Notebook (`.ipynb`) を `ROP_project/notebooks/` に移動する
- [x] 画像や動画などのデータファイルを `ROP_project/data/` に移動またはコピーする

## フェーズ2: 機能のモジュール化

- [x] `utils`ディレクトリに`__init__.py`を作成する
- [x] `utils/file_utils.py`を作成し、ファイル操作関連の関数を実装する
    - [x] ファイル検索 (`glob`)
    - [x] ファイルコピー/移動
    - [x] ディレクトリ作成
    - [x] データセット分割
- [x] `utils/image_processing.py`を作成し、画像処理関連の関数を実装する
    - [x] 動画からフレームを抽出
    - [x] レンズ領域の検出
    - [x] レンズの切り出し
    - [x] 背景の塗りつぶし
    - [x] 画像のリサイズ
    - [x] 画像のオーグメンテーション
- [x] `utils/yaml_utils.py`を作成し、YAML操作関連の関数を実装する
    - [x] YAMLファイルの読み込み
    - [x] YAMLファイルの書き込み
    - [x] データセット用YAMLファイルの生成

## フェーズ3: CLIの作成

- [x] `ROP_project`ディレクトリ直下に`main.py`を作成する
- [x] `main.py`に`argparse`を導入し、CLIの骨格を実装する
- [x] `prepare_data`サブコマンドを実装する
    - [x] 設定ファイルの読み込み
    - [x] フレーム抽出
    - [x] レンズ検出と切り出し
    - [x] 背景塗りつぶし
    - [x] データセット分割
    - [x] データセットYAML生成
- [x] `train`サブコマンドを実装する
    - [x] `detection`モデルのトレーニング機能を呼び出す
    - [x] `segmentation`モデルのトレーニング機能を呼び出す
- [x] `inference`サブコマンドを実装する
    - [x] `detection`モデルでの推論機能を呼び出す

## フェーズ4: ドキュメントと依存関係の整理

- [x] `README.md`を作成し、プロジェクトの概要、セットアップ方法、使い方を記述する
- [x] `requirements.txt`を作成し、プロジェクトの依存関係をリストアップする
- [x] 各スクリプトと関数にdocstringを追加する