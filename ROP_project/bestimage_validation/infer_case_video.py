"""
動画 inference（validate_images.ipynb のモデル/前処理に合わせた汎用版）

フロー:
  動画をフレームに分解 →
  RT-DETRでLens bbox検出 →
  bboxでクロップ →
  レンズ外を灰色(114,114,114)で塗りつぶす円形マスク →
  YOLO-seg（retina_masks=True） →
  retina/disc/macula を元フレームに重畳 →
  動画にレンダリング

使い方:
  - 単一 case_id を処理:
      .\\ropenv\\Scripts\\python.exe ROP_project\\bestimage_validation\\infer_case_video.py --case_id 1363
  - 複数 case_id を一括処理:
      .\\ropenv\\Scripts\\python.exe ROP_project\\bestimage_validation\\infer_case_video.py --case_ids 1227,1376,1601
  - 動画パスを直接指定（case_id推定: .../<id>/movie/<file> を想定）:
      .\\ropenv\\Scripts\\python.exe ROP_project\\bestimage_validation\\infer_case_video.py --video_path \"...\\1363\\movie\\IMG_1363.MOV\"

入出力:
  - 入力動画（既定）: bestimage_validation/<case_id>/movie/*.MOV 等
  - 出力動画（既定）: bestimage_validation/inference/<case_id>/<stem>_inferred.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import RTDETR, YOLO


BASE_DIR = Path(__file__).resolve().parent  # .../ROP_project/bestimage_validation

# YOLO-seg のクラスID（このプロジェクトの前提）
CLASS_ID_TO_NAME = {
    0: "Fundus",  # retina
    1: "Disc",
    2: "Macula",
}

# 描画色（BGR）
NAME_TO_COLOR = {
    "Lens_BBox": (255, 0, 0),  # 青
    "Fundus": (255, 0, 0),     # 青
    "Disc": (0, 255, 0),       # 緑
    "Macula": (0, 0, 255),     # 赤
}


def _get_windows_short_path(path: str) -> str:
    """Windowsで日本語パス等によりVideoCaptureが開けない場合があるため8.3短縮パスで再試行する。"""
    if os.name != "nt":
        return path
    try:
        import ctypes

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint

        buf = ctypes.create_unicode_buffer(32768)
        rc = GetShortPathNameW(path, buf, len(buf))
        if rc == 0:
            return path
        short_path = buf.value
        return short_path if short_path else path
    except Exception:
        return path


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    """OpenCV VideoCapture を堅牢に開く（.MOVや日本語パスで失敗しがちなためフォールバック付き）。"""
    candidates = [("default", video_path, None), ("ffmpeg", video_path, cv2.CAP_FFMPEG)]
    short_path = _get_windows_short_path(video_path)
    if short_path != video_path:
        candidates += [("default_short", short_path, None), ("ffmpeg_short", short_path, cv2.CAP_FFMPEG)]

    last = None
    for tag, p, api in candidates:
        cap = cv2.VideoCapture(p) if api is None else cv2.VideoCapture(p, api)
        if cap.isOpened():
            if p != video_path:
                print(f"[INFO] VideoCapture opened via {tag}: {p}")
            return cap
        last = cap

    if last is not None:
        last.release()
    raise RuntimeError(
        f"動画を開けません: {video_path}\n"
        f"- 対策: 一度ファイルを英数字のみのパス（例: C:\\\\temp\\\\IMG_1363.MOV）にコピーして再実行\n"
        f"- または: .MOV を .mp4(H.264) に変換して再実行（OpenCVのコーデック相性回避）"
    )


def pick_best_lens_bbox(det_results) -> Optional[np.ndarray]:
    """RT-DETR結果から Lens(cls=0) bbox(xyxy) を1つ選ぶ（conf最大を優先）。"""
    best = None
    best_conf = -1.0
    for r in det_results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
        xyxy = boxes.xyxy.detach().cpu().numpy()
        for i in range(len(cls_ids)):
            if int(cls_ids[i]) != 0:
                continue
            conf = float(confs[i]) if confs is not None else 0.0
            if conf > best_conf:
                best_conf = conf
                best = xyxy[i]
    return best


def clamp_xyxy(xyxy: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(c) for c in xyxy]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def apply_circular_mask_to_crop(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """validate_images.ipynb と同様の円形マスク（クロップ内中心・直径=(w+h)/2）。"""
    h, w = crop_bgr.shape[:2]
    center_x, center_y = w // 2, h // 2
    diameter = (w + h) / 2.0
    radius = int(diameter / 2.0)

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)

    masked = crop_bgr.copy()
    masked[circle_mask == 0] = (114, 114, 114)
    return masked, circle_mask


def overlay_masks_on_frame(
    frame_bgr: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    circle_mask_crop: np.ndarray,
    yolo_masks: np.ndarray,
    yolo_cls_ids: np.ndarray,
    alpha_fundus: float = 0.45,
) -> np.ndarray:
    """bbox領域に、Fundusは半透明・Disc/Maculaは不透明で重畳する。"""
    x1, y1, x2, y2 = bbox_xyxy
    crop_h, crop_w = (y2 - y1), (x2 - x1)
    if crop_h <= 0 or crop_w <= 0:
        return frame_bgr

    out = frame_bgr.copy()
    roi = out[y1:y2, x1:x2].copy()
    overlay = roi.copy()

    # Fundus半透明合成の対象マスク（Disc/Macula領域は除外して色が薄まらないようにする）
    fundus_alpha_mask = np.zeros((crop_h, crop_w), dtype=bool)

    for mask_data, cls_id in zip(yolo_masks, yolo_cls_ids):
        name = CLASS_ID_TO_NAME.get(int(cls_id))
        if not name:
            continue

        color = NAME_TO_COLOR.get(name, (255, 255, 255))
        mask_resized = cv2.resize(mask_data.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_resized > 0.5) & (circle_mask_crop > 0)
        if not mask_bin.any():
            continue

        if name == "Fundus":
            overlay[mask_bin] = color
            fundus_alpha_mask |= mask_bin
        else:
            # Disc/Macula は不透明（純色で上書き）
            roi[mask_bin] = color
            # Fundus半透明合成がDisc/Maculaにかからないよう除外
            fundus_alpha_mask[mask_bin] = False

    # Fundusのみ半透明で合成（Disc/Maculaは純色のまま残す）
    if fundus_alpha_mask.any():
        blended = cv2.addWeighted(overlay, alpha_fundus, roi, 1.0 - alpha_fundus, 0.0)
        roi[fundus_alpha_mask] = blended[fundus_alpha_mask]
    out[y1:y2, x1:x2] = roi
    return out


def find_video_for_case(case_id: str) -> Path:
    """bestimage_validation/<case_id>/movie/ 配下から動画を1つ選ぶ（拡張子優先順あり）。"""
    movie_dir = BASE_DIR / case_id / "movie"
    if not movie_dir.is_dir():
        raise FileNotFoundError(f"movieフォルダが見つかりません: {movie_dir}")

    exts = [".MOV", ".mov", ".MP4", ".mp4", ".AVI", ".avi"]
    for ext in exts:
        cand = list(movie_dir.glob(f"*{ext}"))
        if cand:
            return sorted(cand, key=lambda p: p.name.lower())[0]
    raise FileNotFoundError(f"動画ファイルが見つかりません: {movie_dir} (expected one of {exts})")


def default_output_path(case_id: str, video_path: Path) -> Path:
    out_dir = BASE_DIR / "inference" / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{video_path.stem}_inferred.mp4"


def parse_case_ids(s: str) -> List[str]:
    # "1227, 1376 1601" どちらでもOK
    s = s.replace(",", " ")
    items = [x.strip() for x in s.split() if x.strip()]
    # 重複除去（順序維持）
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def infer_video(
    video_path: str,
    output_video_path: str,
    rtdetr_model_path: str,
    yolo_seg_model_path: str,
    yolo_input_width: int = 640,
    conf_det: float = 0.25,
    conf_seg: float = 0.25,
    iou_seg: float = 0.45,
    max_frames: int = 0,
    every_n: int = 1,
    device: str = "auto",
    save_frames_dir: str = "",
    detection_model=None,
    segmentation_model=None,
) -> None:
    video_path = os.path.abspath(str(video_path))
    output_video_path = str(output_video_path)

    # 8.3短縮パス（取れれば）を優先してデコードへ渡す
    video_path_for_decode = _get_windows_short_path(video_path)
    if video_path_for_decode != video_path:
        print(f"[INFO] using short path for decode: {video_path_for_decode}")
    if not os.path.exists(video_path_for_decode):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
        video_path_for_decode = video_path

    # --- 動画入力のオープン ---
    cap = None
    use_imageio = False
    try:
        cap = open_video_capture(video_path_for_decode)
    except Exception as e:
        print(f"[WARN] OpenCVで動画を開けませんでした。imageioで再試行します。\n  reason: {e}")
        use_imageio = True

    fps = 30.0
    W = 0
    H = 0

    if not use_imageio:
        assert cap is not None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        import imageio.v2 as imageio  # type: ignore

        reader = imageio.get_reader(video_path_for_decode)
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 30.0) or 30.0)
        first_rgb = reader.get_data(0)
        first_bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
        H, W = first_bgr.shape[:2]

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriterの初期化に失敗しました: {output_video_path}")

    # --- モデルロード（外部注入も可） ---
    if detection_model is None:
        detection_model = RTDETR(rtdetr_model_path)
    if segmentation_model is None:
        segmentation_model = YOLO(yolo_seg_model_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda指定ですがCUDAが利用できません。cpuで続行します。")
        device = "cpu"
    if device != "cpu":
        detection_model.to(device)
        segmentation_model.to(device)

    if save_frames_dir:
        Path(save_frames_dir).mkdir(parents=True, exist_ok=True)

    total_frames = 0
    n_to_process = None
    if not use_imageio and cap is not None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
        n_to_process = total_frames if total_frames > 0 else None

    idx = 0
    processed = 0
    pbar_total = n_to_process if n_to_process is not None else 0
    pbar = tqdm(total=pbar_total, desc="動画フレーム推論", ncols=100) if pbar_total else tqdm(desc="動画フレーム推論", ncols=100)

    try:
        if use_imageio:
            import imageio.v2 as imageio  # type: ignore

            reader = imageio.get_reader(video_path_for_decode)
            frame_iter = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in reader)
        else:
            assert cap is not None

            def _opencv_frames(c):
                while True:
                    ret, fr = c.read()
                    if not ret:
                        break
                    yield fr

            frame_iter = _opencv_frames(cap)

        for frame in frame_iter:
            if frame is None:
                break

            idx += 1
            if every_n > 1 and (idx % every_n) != 0:
                writer.write(frame)
                if pbar_total:
                    pbar.update(1)
                continue

            det_results = detection_model(frame, verbose=False, conf=conf_det)
            bbox = pick_best_lens_bbox(det_results)
            if bbox is None:
                writer.write(frame)
                if save_frames_dir:
                    cv2.imwrite(str(Path(save_frames_dir) / f"frame_{idx:06d}_no_lens.jpg"), frame)
                if pbar_total:
                    pbar.update(1)
                processed += 1
                if max_frames and processed >= max_frames:
                    break
                continue

            x1, y1, x2, y2 = clamp_xyxy(bbox, W, H)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                writer.write(frame)
                if pbar_total:
                    pbar.update(1)
                processed += 1
                if max_frames and processed >= max_frames:
                    break
                continue

            masked_crop, circle_mask = apply_circular_mask_to_crop(crop)
            crop_h, crop_w = masked_crop.shape[:2]
            aspect = crop_h / max(crop_w, 1)
            yolo_h = int(yolo_input_width * aspect)
            yolo_input = cv2.resize(masked_crop, (yolo_input_width, yolo_h), interpolation=cv2.INTER_AREA)

            seg_results = segmentation_model(
                yolo_input,
                verbose=False,
                retina_masks=True,
                conf=conf_seg,
                iou=iou_seg,
            )

            out_frame = frame.copy()
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), NAME_TO_COLOR["Lens_BBox"], 2)

            if seg_results and seg_results[0].masks is not None and seg_results[0].boxes is not None:
                r0 = seg_results[0]
                masks = r0.masks.data.detach().cpu().numpy()
                cls_ids = r0.boxes.cls.detach().cpu().numpy().astype(int)
                out_frame = overlay_masks_on_frame(
                    out_frame,
                    (x1, y1, x2, y2),
                    circle_mask_crop=circle_mask,
                    yolo_masks=masks,
                    yolo_cls_ids=cls_ids,
                    alpha_fundus=0.45,
                )

            writer.write(out_frame)

            if save_frames_dir:
                cv2.imwrite(str(Path(save_frames_dir) / f"frame_{idx:06d}.jpg"), out_frame)

            if pbar_total:
                pbar.update(1)

            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        pbar.close()
        if cap is not None:
            cap.release()
        writer.release()

    print(f"\n[DONE] output_video_path: {output_video_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", default="", help="単一case_id（例: 1363）")
    parser.add_argument("--case_ids", default="", help="複数case_id（例: 1227,1376,1601）。指定すると順次実行します")
    parser.add_argument("--video_path", default="", help="入力動画パス（--case_id(s) より優先）")
    parser.add_argument("--output_video_path", default="", help="単一実行時の出力動画パス（未指定なら inference/<case_id>/ に保存）")
    parser.add_argument(
        "--rtdetr_model_path",
        default=r"C:\Users\ykita\ROP_AI_project\ROP_project\models\rtdetr-l-1697_1703.pt",
        help="RT-DETRモデルパス",
    )
    parser.add_argument(
        "--yolo_seg_model_path",
        default=r"C:\Users\ykita\ROP_AI_project\ROP_project\models\yolo11n-seg_19movies.pt",
        help="YOLO-segモデルパス",
    )
    parser.add_argument("--yolo_input_width", type=int, default=640)
    parser.add_argument("--conf_det", type=float, default=0.25)
    parser.add_argument("--conf_seg", type=float, default=0.25)
    parser.add_argument("--iou_seg", type=float, default=0.45)
    parser.add_argument("--max_frames", type=int, default=0, help="0なら全フレーム。デバッグ用に上限指定可能。")
    parser.add_argument("--every_n", type=int, default=1, help="nフレームに1回推論（スピード調整）。1なら全フレーム。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_frames_dir", default="", help="デバッグ用: 生成フレームを画像保存するフォルダ（未指定なら保存しない）")
    args = parser.parse_args()

    # --- case list ---
    case_list: List[str] = []
    if args.case_ids:
        case_list = parse_case_ids(args.case_ids)
    elif args.case_id:
        case_list = [str(args.case_id)]

    # --- single by video_path ---
    if args.video_path:
        video_path = Path(args.video_path)
        case_id = case_list[0] if case_list else video_path.parent.parent.name
        if args.output_video_path:
            out_path = Path(args.output_video_path)
        else:
            out_path = default_output_path(case_id, video_path)
        infer_video(
            video_path=str(video_path),
            output_video_path=str(out_path),
            rtdetr_model_path=args.rtdetr_model_path,
            yolo_seg_model_path=args.yolo_seg_model_path,
            yolo_input_width=args.yolo_input_width,
            conf_det=args.conf_det,
            conf_seg=args.conf_seg,
            iou_seg=args.iou_seg,
            max_frames=args.max_frames,
            every_n=args.every_n,
            device=args.device,
            save_frames_dir=args.save_frames_dir,
        )
        return

    # --- batch by case_id(s) ---
    if not case_list:
        raise SystemExit("エラー: --case_id / --case_ids / --video_path のいずれかを指定してください")
    if args.output_video_path and len(case_list) > 1:
        raise SystemExit("エラー: --case_ids（複数指定）の場合は --output_video_path は指定しないでください（出力は inference/<case_id>/ に保存されます）")

    # モデルは一度だけロードして使い回す（高速化）
    detection_model = RTDETR(args.rtdetr_model_path)
    segmentation_model = YOLO(args.yolo_seg_model_path)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device != "cpu" and torch.cuda.is_available():
        detection_model.to(device)
        segmentation_model.to(device)

    failed = []
    for cid in case_list:
        try:
            print(f"\n=== case_id={cid} ===")
            vpath = find_video_for_case(cid)
            out_path = default_output_path(cid, vpath)
            infer_video(
                video_path=str(vpath),
                output_video_path=str(out_path),
                rtdetr_model_path=args.rtdetr_model_path,
                yolo_seg_model_path=args.yolo_seg_model_path,
                yolo_input_width=args.yolo_input_width,
                conf_det=args.conf_det,
                conf_seg=args.conf_seg,
                iou_seg=args.iou_seg,
                max_frames=args.max_frames,
                every_n=args.every_n,
                device=device,
                save_frames_dir=args.save_frames_dir,
                detection_model=detection_model,
                segmentation_model=segmentation_model,
            )
        except Exception as e:
            print(f"[ERROR] case_id={cid}: {e}")
            failed.append(cid)

    if failed:
        raise SystemExit(f"一部失敗: {failed}")


if __name__ == "__main__":
    main()


