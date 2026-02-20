"""
マルチセンター研究用動画処理と画像選定パイプライン
メインスクリプト
"""
import os
import sys
from pathlib import Path
from typing import List
import pandas as pd
from datetime import datetime

# モジュールのインポート
from extract_frames import find_video_files, extract_frames_from_video
from quality_assessment import load_models, assess_image_quality
from select_best_images import select_best_images, copy_best_images


# ==================== 設定 ====================

# パス設定
INPUT_VIDEO_DIR = r"E:\Multicenter_ROP_study\Multicenter_movies"
OUTPUT_IMAGES_DIR = r"E:\Multicenter_ROP_study\Multicenter_images\all_images"
OUTPUT_SELECTED_DIR = r"E:\Multicenter_ROP_study\Multicenter_images\selected_images"
OUTPUT_EXCEL_PATH = r"E:\Multicenter_ROP_study\Multicenter_images\selected_images.xlsx"

# モデルパス
MODELS_DIR = r"C:\Users\ykita\ROP_AI_project\ROP_project\models"
RTDETR_MODEL_PATH = os.path.join(MODELS_DIR, "rtdetr-l-1697_1703.pt")
YOLO_SEG_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n-seg_19movies.pt")

# 処理設定
FRAME_INTERVAL = 5  # 5フレーム毎に抽出
TOP_K = 10  # ベスト10を選出
NEED_K = 5  # 最低必要数

# 除外フォルダ
EXCLUDE_FOLDERS = ["do not use"]


def check_paths():
    """必要なパスとモデルファイルの存在確認"""
    errors = []
    
    # 入力ディレクトリ
    if not os.path.exists(INPUT_VIDEO_DIR):
        errors.append(f"入力動画ディレクトリが存在しません: {INPUT_VIDEO_DIR}")
    
    # モデルファイル
    if not os.path.exists(RTDETR_MODEL_PATH):
        errors.append(f"RT-DETRモデルが見つかりません: {RTDETR_MODEL_PATH}")
    
    if not os.path.exists(YOLO_SEG_MODEL_PATH):
        errors.append(f"YOLO-segモデルが見つかりません: {YOLO_SEG_MODEL_PATH}")
    
    # 出力ディレクトリは自動作成するので、親ディレクトリのみ確認
    output_images_parent = os.path.dirname(OUTPUT_IMAGES_DIR)
    if not os.path.exists(output_images_parent):
        try:
            os.makedirs(output_images_parent, exist_ok=True)
        except Exception as e:
            errors.append(f"出力ディレクトリを作成できません: {output_images_parent}: {e}")
    
    if errors:
        print("エラー: 以下の問題が見つかりました:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def output_to_excel(all_best_results: List[pd.DataFrame], output_path: str):
    """
    全動画のベスト10結果を1つのExcelファイルにまとめて出力（新アルゴリズム対応）

    Args:
        all_best_results: 各動画のベスト10結果のDataFrameのリスト
        output_path: 出力Excelファイルのパス
    """
    if not all_best_results:
        print("警告: 出力するデータがありません")
        return

    # 全結果を結合
    all_df = pd.concat(all_best_results, ignore_index=True)

    # Excel出力用のカラムを選択・整理（新アルゴリズム対応）
    output_columns = [
        'image_id', 'rank', 'image_name', 'selection_stage',
        'retina_ratio', 'retina_area',
        'disc_detected', 'disc_edge_coverage_ratio', 'disc_edge_covered',
        'mbss_Grad_p90', 'mbss_score', 'S_mean',
        'score'
    ]

    # 存在するカラムのみを選択
    available_columns = [col for col in output_columns if col in all_df.columns]
    output_df = all_df[available_columns].copy()

    # 出力ディレクトリを作成
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        output_df.to_excel(output_path, index=False)
        print(f"\n保存しました: {output_path}")
        print(f"総行数: {len(output_df)}")
        print(f"動画数: {output_df['image_id'].nunique()}")

        # 統計情報
        if 'selection_stage' in output_df.columns:
            stage1_count = len(output_df[output_df['selection_stage'] == 'Stage1_edge_cov>=0.80'])
            stage2_count = len(output_df[output_df['selection_stage'] == 'Stage2_補完'])
            print(f"  Stage1 (edge_cov>=0.80): {stage1_count}件")
            print(f"  Stage2 (補完): {stage2_count}件")

    except PermissionError as e:
        # ファイルが使用中の場合はタイムスタンプ付きで保存
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(output_path)
        alt_path = f"{root}_{ts}{ext}"
        output_df.to_excel(alt_path, index=False)
        print(f"\n[WARN] 出力先ファイルが使用中のため上書きできませんでした: {output_path}")
        print(f"       代替ファイルに保存しました: {alt_path}")
        print(f"       総行数: {len(output_df)}")
        print(f"       動画数: {output_df['image_id'].nunique()}")
    except Exception as e:
        print(f"\nExcel出力に失敗しました: {e}")
        # 代替: CSV
        alt_csv = output_path.replace('.xlsx', '.csv')
        output_df.to_csv(alt_csv, index=False, encoding='utf-8-sig')
        print(f"代替でCSV保存しました: {alt_csv}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("マルチセンター研究用動画処理と画像選定パイプライン")
    print("=" * 60)
    
    # パス確認
    print("\n[1/5] パスとモデルファイルの確認...")
    if not check_paths():
        print("エラー: パス確認に失敗しました。処理を中断します。")
        sys.exit(1)
    print("✓ パス確認完了")
    
    # 動画ファイル検索
    print(f"\n[2/5] 動画ファイルを検索中: {INPUT_VIDEO_DIR}")
    video_files = find_video_files(
        INPUT_VIDEO_DIR,
        extensions=('.mov', '.mp4', '.MOV', '.MP4'),
        exclude_folders=EXCLUDE_FOLDERS
    )
    
    if not video_files:
        print("エラー: 処理対象の動画ファイルが見つかりませんでした")
        sys.exit(1)
    
    print(f"✓ {len(video_files)}個の動画ファイルが見つかりました")

    # モデルを1回だけ読み込み
    print("\n[2.5/5] モデルを読み込み中...")
    detection_model, segmentation_model = load_models(
        RTDETR_MODEL_PATH, YOLO_SEG_MODEL_PATH
    )
    print("✓ モデル読み込み完了")

    # 全動画のベスト10結果を保存するリスト
    all_best_results = []

    # 各動画を処理
    print(f"\n[3/5] 各動画を処理中...")
    for idx, video_path in enumerate(video_files, 1):
        video_basename = Path(video_path).stem
        print(f"\n--- [{idx}/{len(video_files)}] {video_basename} ---")
        
        try:
            # 2-1. フレーム抽出
            print("  [1/4] フレーム抽出中...")
            extracted_images = extract_frames_from_video(
                video_path=video_path,
                output_dir=OUTPUT_IMAGES_DIR,
                frame_interval=FRAME_INTERVAL,
                image_prefix=video_basename
            )
            
            if not extracted_images:
                print(f"  警告: フレームが抽出できませんでした（スキップ）")
                continue
            
            print(f"  ✓ {len(extracted_images)}フレーム抽出完了")
            
            # 2-2. 品質評価
            print("  [2/4] 品質評価中...")
            results_df = assess_image_quality(
                image_paths=extracted_images,
                detection_model=detection_model,
                segmentation_model=segmentation_model,
                image_id=video_basename
            )
            print(f"  ✓ 品質評価完了: {len(results_df)}枚の画像を評価")
            
            # 2-3. ベスト10選出
            print("  [3/4] ベスト10選出中...")
            best10_df = select_best_images(
                df=results_df,
                top_k=TOP_K,
                need_k=NEED_K
            )
            print(f"  ✓ ベスト{len(best10_df)}枚を選出")
            
            # 2-4. selected_imagesにコピー
            print("  [4/4] ベスト画像をコピー中...")
            copy_best_images(
                best_df=best10_df,
                output_dir=OUTPUT_SELECTED_DIR,
                source_column='image_path'
            )
            print(f"  ✓ コピー完了")
            
            # 2-5. 結果をリストに追加（後でExcel出力用）
            all_best_results.append(best10_df)
            
            print(f"  ✓ {video_basename} の処理完了")
            
        except Exception as e:
            print(f"  エラー: {video_basename} の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            print(f"  → この動画をスキップして続行します")
            continue
    
    # Excelにまとめて出力
    print(f"\n[4/5] Excelファイルに出力中...")
    if all_best_results:
        output_to_excel(all_best_results, OUTPUT_EXCEL_PATH)
    else:
        print("警告: 出力する結果がありませんでした")
    
    # 完了
    print(f"\n[5/5] 処理完了!")
    print(f"=" * 60)
    print(f"処理した動画数: {len(all_best_results)}")
    print(f"抽出画像保存先: {OUTPUT_IMAGES_DIR}")
    print(f"ベスト画像保存先: {OUTPUT_SELECTED_DIR}")
    print(f"Excel出力先: {OUTPUT_EXCEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

