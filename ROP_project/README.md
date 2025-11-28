# ROP AI Project

This project is a CLI-based tool for preparing data, training, and running inference for ROP (Retinopathy of Prematurity) detection and segmentation models.

## セットアップ (Setup)

1.  **リポジトリをクローンします。**
    ```bash
    git clone <repository_url>
    cd ROP_AI_project
    ```

2.  **仮想環境を有効化し、依存関係をインストールします。**
    ```bash
    # Windows
    .\ropenv\Scripts\activate
    
    # pipで依存関係をインストール
    pip install -r ROP_project/requirements.txt
    ```

## ディレクトリ構造 (Directory Structure)
```
ROP_project/
├── data/                # データセット、動画、設定ファイルなど
├── models/              # 事前学習済みモデルの重みファイル (.pt)
├── notebooks/           # 実験用のJupyter Notebook
├── outputs/             # 推論結果の出力先
├── src/                 # ソースコード
│   └── utils/           # 各種ユーティリティモジュール
├── main.py              # CLIのエントリーポイント
├── requirements.txt     # 依存ライブラリ
└── README.md            # このファイル
```

## 使い方 (Usage)

このプロジェクトは、以下の3つのステップで利用します。

### ステップ1: データ準備の設定

まず、データの前処理方法を定義した設定ファイル（YAML形式）を作成します。
`ROP_project`ディレクトリに、`data_prep_config.yaml`のような名前でファイルを作成します。

**`data_prep_config.yaml` の例:**
```yaml
# データ準備のための設定ファイル
base_dir: 'C:/Users/ykita/ROP_AI_project/ROP_project/data'
video_path: 'C:/Users/ykita/ROP_AI_project/ROP_project/data/ROP_video/video.mp4'
image_size: [640, 640]
train_ratio: 0.8
class_names: ['class1', 'class2']
```

### ステップ2: データの前処理 (`prepare_data`)

次に、作成した設定ファイルを使ってデータの前処理を実行します。
このコマンドは、動画からのフレーム抽出、画像のリサイズ、データセットの分割、学習用の`dataset.yaml`の生成を自動で行います。

**コマンド例:**
```bash
# ROP_project ディレクトリで実行
python main.py prepare_data --config data_prep_config.yaml
```
実行後、`base_dir`で指定したディレクトリに`dataset.yaml`が生成されます。

### ステップ3: モデルのトレーニング (`train`)

データセットの準備ができたら、モデルをトレーニングします。
使用する事前学習済みモデル、データセットのYAMLファイル、エポック数を指定します。

**コマンド例:**
```bash
# `yolov8n.pt`をベースに50エポックでトレーニングする場合
# `yolov8n.pt`は事前に `ROP_project/models` などに配置してください。
python main.py train --model models/yolov8n.pt --data data/dataset.yaml --epochs 50
```
トレーニングが完了すると、結果が`ROP_project/runs/detect/train/`のようなディレクトリに保存されます。

### ステップ4: 推論の実行 (`inference`)

トレーニング済みのモデルを使って、新しい画像や動画に対して推論を実行します。
使用するモデルの重みファイル（通常は`best.pt`）と、推論対象の画像/ディレクトリを指定します。

**コマンド例:**
```bash
# 学習済みモデルを使い、検証用画像に対して推論を実行する場合
python main.py inference --model runs/detect/train/weights/best.pt --source data/val/images
```
推論結果は、`ROP_project/runs/detect/predict/`のようなディレクトリに保存されます。
