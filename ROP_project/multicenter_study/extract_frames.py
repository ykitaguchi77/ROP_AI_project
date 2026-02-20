"""
動画からフレームを抽出するモジュール
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import List


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 5,
    image_prefix: str = None
) -> List[str]:
    """
    動画から指定間隔でフレームを抽出してPNGとして保存
    
    Args:
        video_path: 動画ファイルのパス
        output_dir: 出力ディレクトリ
        frame_interval: 抽出間隔（デフォルト: 5フレーム毎）
        image_prefix: 画像ファイル名のプレフィックス（Noneの場合は動画のbasenameを使用）
    
    Returns:
        抽出された画像ファイルのパスのリスト
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル名のプレフィックスを決定
    if image_prefix is None:
        image_prefix = Path(video_path).stem
    
    # 動画の読み込み
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けませんでした: {video_path}")
    
    # 総フレーム数の取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    extracted_images = []
    frame_count = 0
    saved_count = 0
    
    # 進捗バーを表示しながらフレーム抽出
    with tqdm(total=total_frames, desc=f"フレーム抽出: {Path(video_path).name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 指定間隔ごとに保存
            if frame_count % frame_interval == 0:
                image_filename = f"{image_prefix}_{saved_count:04d}.png"
                image_path = output_dir / image_filename
                cv2.imwrite(str(image_path), frame)
                extracted_images.append(str(image_path))
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"合計 {saved_count} フレームを抽出しました")
    return extracted_images


def find_video_files(
    root_dir: str,
    extensions: tuple = ('.mov', '.mp4', '.MOV', '.MP4'),
    exclude_folders: List[str] = None
) -> List[str]:
    """
    指定ディレクトリ内の動画ファイルを検索（除外フォルダを除く）
    
    Args:
        root_dir: 検索対象のルートディレクトリ
        extensions: 対象となる拡張子のタプル
        exclude_folders: 除外するフォルダ名のリスト（パスに含まれる場合除外）
    
    Returns:
        動画ファイルのパスのリスト
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"ディレクトリが存在しません: {root_dir}")
    
    if exclude_folders is None:
        exclude_folders = []
    
    video_files = []
    
    for ext in extensions:
        for video_path in root_path.rglob(f"*{ext}"):
            # 除外フォルダチェック
            should_exclude = False
            for exclude_folder in exclude_folders:
                if exclude_folder in str(video_path):
                    should_exclude = True
                    break
            
            if not should_exclude:
                video_files.append(str(video_path))
    
    return sorted(video_files)


