# ==== fast_mosaic_detector_hybrid.py ====

from __future__ import annotations
from dataclasses import dataclass, replace
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import os
import cv2
import numpy as np
import torch

# ---- 相対インポート（ComfyUI custom_nodes パッケージ配下である前提） ----
# D（高精度／NumPy & 単発Torch）
from .fast_mosaic_detector_accurate import (
    MosaicDetectionConfig as DConfig,
    _detect_mosaic_mask_numpy as detect_numpy_D,
)

# B（高速／Torchバッチ）
try:
    from .fast_mosaic_detector_fast import (
        _detect_mosaic_mask_torch_batch as detect_torch_B_batch,
    )
    HAS_B = True
except Exception:
    detect_torch_B_batch = None
    HAS_B = False

# ----------------- ユーティリティ -----------------

def _ensure_nhwc(image: torch.Tensor) -> torch.Tensor:
    """
    ComfyUIのIMAGE入力を (N,H,W,3) float32 [0,1] に正規化する。
    許容:
      (H,W,3), (3,H,W), (N,H,W,3), (N,3,H,W)
    """
    t = image
    if t.ndim == 3:
        if t.shape[-1] == 3:
            t = t.unsqueeze(0)  # (1,H,W,3)
        elif t.shape[0] == 3:
            t = t.permute(1, 2, 0).unsqueeze(0)  # (1,H,W,3)
        else:
            raise ValueError(f"Expected (H,W,3) or (3,H,W), got {tuple(t.shape)}")
    elif t.ndim == 4:
        if t.shape[-1] == 3:
            pass  # (N,H,W,3)
        elif t.shape[1] == 3:
            t = t.permute(0, 2, 3, 1)  # (N,H,W,3)
        else:
            raise ValueError(f"Expected (N,H,W,3) or (N,3,H,W), got {tuple(t.shape)}")
    else:
        raise ValueError(f"Unexpected rank for IMAGE: {t.ndim}")

    t = t.to(dtype=torch.float32).clamp(0.0, 1.0)
    return t


def _mask_bboxes(mask_u8: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    2値(0/255)マスクから外接矩形群 [(x0,y0,x1,y1), ...] を返す。
    """
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    _, binmask = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binmask, connectivity=8)
    boxes: List[Tuple[int,int,int,int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area > 0 and w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def _expand_box(box: Tuple[int,int,int,int], w: int, h: int, margin: int) -> Tuple[int,int,int,int]:
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(w, x1 + margin),
        min(h, y1 + margin),
    )


def _apply_refine_logic(mask_fast: np.ndarray, mask_acc: np.ndarray, logic: str) -> np.ndarray:
    logic = (logic or "replace").lower()
    if logic == "replace":
        return mask_acc
    if logic == "union":
        return np.where((mask_fast > 0) | (mask_acc > 0), 255, 0).astype(np.uint8)
    if logic == "intersect":
        return np.where((mask_fast > 0) & (mask_acc > 0), 255, 0).astype(np.uint8)
    # デフォルトは置換
    return mask_acc

# ==== mosaic_detector_hybridBD.py ====

class MosaicDetectionHybridNode:
    """
    FAST:     Torchバッチ (_detect_mosaic_mask_torch_batch)
    ACCURATE: NumPy (_detect_mosaic_mask_numpy)
    HYBRID:   Part 3で実装（バッチ → ROIだけAccurateで再検出）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 画像入力
                "image": ("IMAGE",),

                # D（高精度）の基本パラメータ
                "hsv_skin_h_low": ("INT", {"default": 35, "min": 0, "max": 180}),
                "hsv_skin_h_high": ("INT", {"default": 160, "min": 0, "max": 180}),
                "hsv_skin_s_threshold": ("INT", {"default": 25, "min": 0, "max": 255}),
                "gradient_threshold": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0}),
                "ratio_threshold": ("FLOAT", {"default": 2.74, "min": 0.0, "max": 20.0, "step":0.01}),
                "histogram_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step":0.01}),
                "mosaic_length_min": ("INT", {"default": 4, "min": 2, "max": 64}),
                "mosaic_length_max": ("INT", {"default": 25, "min": 3, "max": 128}),
                "intersection_margin": ("INT", {"default": 5, "min": 0, "max": 50}),
                "gradient_band_height": ("INT", {"default": 1, "min": 1, "max": 8}),
                "gradient_band_half_width": ("INT", {"default": 15, "min": 1, "max": 64}),

                # 実行モード
                "mode": (
                    [
                        'FAST',
                        'ACCURATE',
                        'HYBRID'
                    ], {
                        "default": 'HYBRID'
                    }),

                # 実行系
                "processing_backend": (
                    [
                        'AUTO',
                        'CPU',
                        'TORCH'
                    ], {
                        "default": 'AUTO'
                    }),

                "max_workers": ("INT", {"default": 8, "min": 0, "max": 32}),

                # --- 以下はHYBRID用（Part 3で使用・ここでは未使用） ---
                "fast_recall_boost": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0}),
                "roi_margin_px": ("INT", {"default": 24, "min": 0, "max": 128}),
                "refine_logic": (
                    [
                        'replace',
                        'union',
                        'intersect'
                    ], {
                        "default": 'replace'
                    }),
                "refine_frame_stride": ("INT", {"default": 3, "min": 1, "max": 12}),
                "roi_merge_dilate_px": ("INT", {"default": 8, "min": 0, "max": 64}),
                "roi_max_count": ("INT", {"default": 5, "min": 1, "max": 32}),
                "min_mask_pixels": ("INT", {"default": 200, "min": 0, "max": 10000}),
                "frame_cover_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.95, "step":0.01}),

                # --- 以下はAdaptiveROI用 ---
                "adaptive_roi_area_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step":0.01}),
                # ROIがフレーム面積の何%以上なら縮小OK
                "adaptive_roi_min_side": ("INT", {"default": 48, "min": 4, "max": 512}),  # ROIの短辺がこのpx未満なら等倍で精査
                "roi_downscale_large": ("FLOAT", {"default": 0.75, "min": 0.4, "max": 1.0, "step":0.01}),  # 大きいROIにだけ適用する縮小率

                "roi_aspect_ratio_max": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}), # recommend range 3-5
                "roi_min_short_side": ("INT", {"default": 12, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("MASK", "INT")
    FUNCTION = "execute"
    CATEGORY = "fast-mosaic-detector"

    # ---- 内部ヘルパ ----

    def _make_config(self, **kw) -> DConfig:
        return DConfig(**kw)

    def _accurate_detect_one_bgr(self, frame_bgr_u8: np.ndarray, cfg_acc: DConfig) -> Tuple[np.ndarray, int]:
        # DのNumPy（高精度）を1フレーム分実行
        return detect_numpy_D(frame_bgr_u8, cfg_acc)

    # ---- 本体 ----

    def execute(
        self,
        image: torch.Tensor,
        hsv_skin_h_low: int,
        hsv_skin_h_high: int,
        hsv_skin_s_threshold: int,
        gradient_threshold: float,
        ratio_threshold: float,
        histogram_threshold: float,
        mosaic_length_min: int,
        mosaic_length_max: int,
        intersection_margin: int,
        gradient_band_height: int,
        gradient_band_half_width: int,
        mode: str,
        processing_backend: str,
        max_workers: int,
        fast_recall_boost: float,
        roi_margin_px: int,
        refine_logic: str,
        refine_frame_stride: int,
        roi_merge_dilate_px: int,
        roi_max_count: int,
        min_mask_pixels: int,
        frame_cover_threshold: float,
        adaptive_roi_area_ratio: float,
        adaptive_roi_min_side: int,
        roi_downscale_large: float,
        roi_aspect_ratio_max: float,
        roi_min_short_side: int,
    ) -> Tuple[torch.Tensor, int]:

        # --- 設定生成（D準拠）
        base_cfg = self._make_config(
            hsv_skin_h_low=hsv_skin_h_low,
            hsv_skin_h_high=hsv_skin_h_high,
            hsv_skin_s_threshold=hsv_skin_s_threshold,
            gradient_threshold=gradient_threshold,
            ratio_threshold=ratio_threshold,
            histogram_threshold=histogram_threshold,
            mosaic_length_min=mosaic_length_min,
            mosaic_length_max=mosaic_length_max,
            intersection_margin=intersection_margin,
            gradient_band_height=gradient_band_height,
            gradient_band_half_width=gradient_band_half_width,
        )

        # --- 画像正規化 (N,H,W,3) float32 [0,1]
        batch_rgb = _ensure_nhwc(image)
        N, H, W, _ = batch_rgb.shape

        # --- backend選択（GPUへ移すだけ・各検出器の内部はそれぞれで最適化）
        backend_choice = (processing_backend or "AUTO").upper()
        if backend_choice == "AUTO":
            use_cuda = torch.cuda.is_available()
        elif backend_choice == "TORCH":
            use_cuda = True
        else:
            use_cuda = False

        if use_cuda and batch_rgb.device.type == "cpu":
            batch_rgb = batch_rgb.cuda(non_blocking=True)

        masks_out: List[np.ndarray] = []
        sizes_out: List[int] = []

        mode_up = (mode or "HYBRID").upper()

        # ---------------- ACCURATE（D＝高精度・遅い） ----------------
        if mode_up == "ACCURATE":
            frames_bgr = (batch_rgb.detach().cpu().numpy()[..., ::-1] * 255.0).astype(np.uint8)  # NHWC BGR
            if N == 1 or max_workers <= 1:
                results = [self._accurate_detect_one_bgr(frames_bgr[0], base_cfg)]
            else:
                with ThreadPoolExecutor(max_workers=min(max_workers, os.cpu_count() or 1)) as ex:
                    results = list(ex.map(
                        lambda f: self._accurate_detect_one_bgr(f, base_cfg),
                        list(frames_bgr)
                    ))
            for m, s in results:
                masks_out.append(m.astype(np.float32) / 255.0)
                sizes_out.append(int(s))

        # ---------------- FAST（B＝高速バッチ） ----------------
        elif mode_up == "FAST":
            if not HAS_B or detect_torch_B_batch is None:
                raise RuntimeError("FAST mode requires B's _detect_mosaic_mask_torch_batch.")
            # Bは“そのままの設定”でOK（boost等はHYBRIDで使用）
            mask_list, size_list = detect_torch_B_batch(batch_rgb, base_cfg, max_workers)
            for m, s in zip(mask_list, size_list):
                masks_out.append(m.astype(np.float32) / 255.0)
                sizes_out.append(int(s))

        # ---------------- HYBRID（Part 3で実装） ----------------
        # ---------------- HYBRID（Bバッチ→Dで部分精査） ----------------
        else:
            if not HAS_B or detect_torch_B_batch is None:
                raise RuntimeError("HYBRID mode requires B's _detect_mosaic_mask_torch_batch.")

            # 一次検出はB（高速）。Dに回すROIを増やしすぎないよう、やや厳しめの設定に寄せる
            cfg_fast = replace(
                base_cfg,
                gradient_threshold=base_cfg.gradient_threshold * fast_recall_boost,
                ratio_threshold=max(0.1, base_cfg.ratio_threshold * fast_recall_boost),
            )

            # Bを全フレームで一括実行（GPU可）
            fast_masks_u8, fast_sizes = detect_torch_B_batch(batch_rgb, cfg_fast, max_workers)
            frames_bgr = (batch_rgb.detach().cpu().numpy()[..., ::-1] * 255.0).astype(np.uint8)

            last_refined_mask: np.ndarray | None = None

            for i in range(N):
                fast_mask_u8 = fast_masks_u8[i]
                fast_size = int(fast_sizes[i])
                H0, W0 = fast_mask_u8.shape[:2]
                nz = int((fast_mask_u8 > 0).sum())
                cover = nz / float(H0 * W0) if H0 * W0 > 0 else 0.0

                # A) フレーム間引き：kフレーム毎にDを実行（間は前回結果 or FASTを使う）
                if (i % refine_frame_stride) != 0:
                    use_mask = last_refined_mask if last_refined_mask is not None else fast_mask_u8
                    masks_out.append(use_mask.astype(np.float32) / 255.0)
                    sizes_out.append(int(fast_size))
                    continue

                # B) ごく小さいノイズはDに送らずFAST採用
                if nz < int(min_mask_pixels):
                    masks_out.append(fast_mask_u8.astype(np.float32) / 255.0)
                    sizes_out.append(int(fast_size))
                    last_refined_mask = fast_mask_u8
                    continue

                # C) 画面の大半がマスク → フレーム全体を1回だけD（大きいので縮小可）
                if cover >= float(frame_cover_threshold):
                    crop = frames_bgr[i]

                    # フルフレームは基本「大きい」とみなす：roi_downscale_large を適用
                    use_down = float(roi_downscale_large)
                    if use_down < 0.999:
                        ds_w = max(4, int(W0 * use_down))
                        ds_h = max(4, int(H0 * use_down))
                        small = cv2.resize(crop, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

                        acc_mask_small, acc_size = self._accurate_detect_one_bgr(small, base_cfg)

                        acc_mask_full = cv2.resize(acc_mask_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
                    else:
                        acc_mask_full, acc_size = self._accurate_detect_one_bgr(crop, base_cfg)

                    acc_sizes = [int(acc_size)] if (acc_size and acc_size > 0) else []

                else:
                    # D) ROI抽出：過検出ぎみのFASTマスクをモルフォロジーでまとめて、上位K個だけ精査
                    merged = fast_mask_u8
                    k = int(roi_merge_dilate_px)
                    if k > 0:
                        kernel = np.ones((k, k), np.uint8)
                        merged = cv2.dilate(merged, kernel, iterations=1)
                        merged = cv2.erode(merged, kernel, iterations=1)

                    contours, _ = cv2.findContours((merged > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    rects = []
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w * h
                        if area > 0:
                            rects.append((area, (x, y, x + w, y + h)))

                    filtered_rects = []
                    for area, (x0, y0, x1, y1) in rects:
                        w_roi = x1 - x0
                        h_roi = y1 - y0
                        short_side = min(w_roi, h_roi)
                        long_side = max(w_roi, h_roi)
                        aspect_ratio = long_side / max(1, short_side)

                        # ---- 細長いものは除外 ----
                        if aspect_ratio > float(roi_aspect_ratio_max):
                            continue
                        # ---- 非常に細いライン（短辺が小さすぎる）を除外 ----
                        if short_side < int(roi_min_short_side):
                            continue

                        filtered_rects.append((area, (x0, y0, x1, y1)))

                    rects = filtered_rects

                    rects.sort(reverse=True, key=lambda t: t[0])
                    rects = [r[1] for r in rects[:int(roi_max_count)]]

                    candidates = [_expand_box(b, W0, H0, int(roi_margin_px)) for b in rects]
                    acc_mask_full = np.zeros((H0, W0), dtype=np.uint8)
                    acc_sizes: List[int] = []

                    # ROIが複数なら並列（DはGIL解放が多くThreadPoolが効くことが多い）
                    if len(candidates) >= 2 and max_workers > 1:
                        def run_one(bx: tuple[int, int, int, int]):
                            x0, y0, x1, y1 = bx
                            crop = frames_bgr[i, y0:y1, x0:x1].copy()
                            if crop.size == 0:
                                return (bx, None, 0)

                            # ---- Adaptive ROI Downscale ----
                            w_roi = max(1, x1 - x0)
                            h_roi = max(1, y1 - y0)
                            area_roi = w_roi * h_roi
                            area_thr = float(adaptive_roi_area_ratio) * float(W0 * H0)
                            min_side = min(w_roi, h_roi)

                            # 小さいROIは等倍（=精度重視）、大きいROIのみ縮小OK
                            if (area_roi > area_thr) and (min_side >= int(adaptive_roi_min_side)):
                                use_down = float(roi_downscale_large)
                            else:
                                use_down = 1.0

                            if use_down < 0.999:
                                ds_w = max(4, int(w_roi * use_down))
                                ds_h = max(4, int(h_roi * use_down))
                                small = cv2.resize(crop, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

                                m, sz = self._accurate_detect_one_bgr(small, base_cfg)

                                if m is not None and m.size > 0:
                                    m = cv2.resize(m, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                            else:
                                m, sz = self._accurate_detect_one_bgr(crop, base_cfg)

                            return (bx, m, int(sz) if sz else 0)

                        with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as ex:
                            results = list(ex.map(run_one, candidates))

                        for (bx, m, sz) in results:
                            if m is not None and m.size > 0:
                                x0, y0, x1, y1 = bx
                                acc_mask_full[y0:y1, x0:x1] = np.maximum(acc_mask_full[y0:y1, x0:x1], m)
                            if sz and sz > 0:
                                acc_sizes.append(int(sz))

                    else:
                        # ROIが1個以下ならシリアルで十分
                        for (x0, y0, x1, y1) in candidates:
                            crop = frames_bgr[i, y0:y1, x0:x1].copy()
                            if crop.size == 0:
                                continue

                            # ---- Adaptive ROI Downscale ----
                            w_roi = max(1, x1 - x0)
                            h_roi = max(1, y1 - y0)
                            area_roi = w_roi * h_roi
                            area_thr = float(adaptive_roi_area_ratio) * float(W0 * H0)
                            min_side = min(w_roi, h_roi)

                            if (area_roi > area_thr) and (min_side >= int(adaptive_roi_min_side)):
                                use_down = float(roi_downscale_large)
                            else:
                                use_down = 1.0

                            if use_down < 0.999:
                                ds_w = max(4, int(w_roi * use_down))
                                ds_h = max(4, int(h_roi * use_down))
                                small = cv2.resize(crop, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

                                m, sz = self._accurate_detect_one_bgr(small, base_cfg)

                                if m is not None and m.size > 0:
                                    m = cv2.resize(m, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                            else:
                                m, sz = self._accurate_detect_one_bgr(crop, base_cfg)

                            if m is not None and m.size > 0:
                                acc_mask_full[y0:y1, x0:x1] = np.maximum(acc_mask_full[y0:y1, x0:x1], m)
                            if sz and sz > 0:
                                acc_sizes.append(int(sz))

                # E) 統合＆保持
                acc_size_final = int(np.median(acc_sizes)) if acc_sizes else int(fast_size)
                merged_mask = _apply_refine_logic(fast_mask_u8, acc_mask_full, refine_logic)

                masks_out.append(merged_mask.astype(np.float32) / 255.0)
                sizes_out.append(int(acc_size_final))
                last_refined_mask = merged_mask

        # ---- 出力整形
        mask_batch = np.stack(masks_out, axis=0)  # (N,H,W)
        nonzero = [s for s in sizes_out if s > 0]
        size_value = int(np.median(nonzero)) if nonzero else 0

        mask_tensor = torch.from_numpy(mask_batch)
        if use_cuda and batch_rgb.device.type != "cpu":
            mask_tensor = mask_tensor.to(batch_rgb.device)

        return (mask_tensor, size_value)


NODE_CLASS_MAPPINGS = {"MosaicDetectionNode": MosaicDetectionHybridNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MosaicDetectionNode": "Fast Mosaic Detector"}

# （任意）簡易デバッグトグル：必要なときだけ True に
DEBUG_HYBRID = False

def _dbg(*args, **kwargs):
    if DEBUG_HYBRID:
        print("[HYBRIDDBG]", *args, **kwargs)
