"""ComfyUI node that ports the C++ mosaic detector to Python."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MosaicDetectionConfig:
    """Parameters controlling the heuristic thresholds."""

    hsv_skin_h_low: int = 35
    hsv_skin_h_high: int = 160
    hsv_skin_s_threshold: int = 25
    gradient_threshold: float = 2.5
    ratio_threshold: float = 2.74
    mosaic_length_min: int = 4
    mosaic_length_max: int = 25
    histogram_threshold: float = 0.1
    intersection_margin: int = 5
    gradient_band_height: int = 1
    gradient_band_half_width: int = 15

    def __post_init__(self) -> None:
        if not (0 <= self.hsv_skin_h_low <= 180 and 0 <= self.hsv_skin_h_high <= 180):
            raise ValueError("HSV hue thresholds must be within [0, 180]")
        if not (0 <= self.hsv_skin_s_threshold <= 255):
            raise ValueError("Saturation threshold must be within [0, 255]")
        if self.mosaic_length_min < 2:
            raise ValueError("mosaic_length_min must be >= 2")
        if self.mosaic_length_max <= self.mosaic_length_min:
            raise ValueError("mosaic_length_max must be greater than mosaic_length_min")
        if self.intersection_margin < 0:
            raise ValueError("intersection_margin must be non-negative")
        if self.gradient_band_height < 1:
            raise ValueError("gradient_band_height must be >= 1")
        if self.gradient_band_half_width < 1:
            raise ValueError("gradient_band_half_width must be >= 1")


def _rect_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Tuple[float, int]:
    """Return (sum, area) for the rectangle [x0, x1) x [y0, y1)."""

    height = integral.shape[0] - 1
    width = integral.shape[1] - 1

    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))

    if x1 <= x0 or y1 <= y0:
        return 0.0, 0

    total = (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )
    area = (x1 - x0) * (y1 - y0)
    return float(total), int(area)


def _get_skin_mask_from_hsv(hsv: np.ndarray, config: MosaicDetectionConfig) -> np.ndarray:
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]

    mask_h_low = (h_channel <= config.hsv_skin_h_low).astype(np.uint8) * 255
    mask_h_high = (h_channel >= config.hsv_skin_h_high).astype(np.uint8) * 255
    mask_s = (s_channel <= config.hsv_skin_s_threshold).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask_h_low, mask_h_high)
    mask = cv2.bitwise_or(mask, mask_s)
    return mask


def _rect_sum_grid(
    integral: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized integral lookup for broadcast-friendly slices."""

    sum_val = (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )
    area = (x1 - x0) * (y1 - y0)
    return sum_val, area


def _get_mosaic_feature_map(v_channel: np.ndarray, config: MosaicDetectionConfig) -> np.ndarray:
    v_float = v_channel.astype(np.float32)
    height, width = v_float.shape

    grad_h = np.zeros_like(v_float)
    grad_v = np.zeros_like(v_float)
    grad_h[:-1, :] = np.abs(v_float[:-1, :] - v_float[1:, :])
    grad_v[:, :-1] = np.abs(v_float[:, :-1] - v_float[:, 1:])

    integral_h = cv2.integral(grad_h).astype(np.float32)
    integral_v = cv2.integral(grad_v).astype(np.float32)

    pw1 = config.gradient_band_height
    pw2 = config.gradient_band_half_width

    output = np.zeros_like(v_channel, dtype=np.uint8)

    x_range = np.arange(2 + pw1, width - 3 - pw1, dtype=np.int32)
    y_range = np.arange(2 + pw1, height - 3 - pw1, dtype=np.int32)

    if x_range.size == 0 or y_range.size == 0:
        return output

    X = x_range[None, :]
    Y = y_range[:, None]

    def _clipped_bounds(offset_x0: int, offset_x1: int, offset_y0: int, offset_y1: int):
        x0 = np.clip(X + offset_x0, 0, width)
        x1 = np.clip(X + offset_x1, 0, width)
        y0 = np.clip(Y + offset_y0, 0, height)
        y1 = np.clip(Y + offset_y1, 0, height)
        return x0.astype(np.int32), x1.astype(np.int32), y0.astype(np.int32), y1.astype(np.int32)

    x0, x1, y0, y1 = _clipped_bounds(-pw2, pw2 + 1, -(2 + pw1), -2)
    sum1h, area1h = _rect_sum_grid(integral_h, y0, y1, x0, x1)

    x0, x1, y0, y1 = _clipped_bounds(-pw2, pw2 + 1, 3, 3 + pw1)
    sum2h, area2h = _rect_sum_grid(integral_h, y0, y1, x0, x1)

    x0, x1, y0, y1 = _clipped_bounds(-pw2, pw2 + 1, 0, 1)
    sum3h, area3h = _rect_sum_grid(integral_h, y0, y1, x0, x1)

    x0, x1, y0, y1 = _clipped_bounds(-(2 + pw1), -2, -pw2, pw2 + 1)
    sum1v, area1v = _rect_sum_grid(integral_v, y0, y1, x0, x1)

    x0, x1, y0, y1 = _clipped_bounds(3, 3 + pw1, -pw2, pw2 + 1)
    sum2v, area2v = _rect_sum_grid(integral_v, y0, y1, x0, x1)

    x0, x1, y0, y1 = _clipped_bounds(0, 1, -pw2, pw2 + 1)
    sum3v, area3v = _rect_sum_grid(integral_v, y0, y1, x0, x1)

    valid_area = (
        (area1h > 0)
        & (area2h > 0)
        & (area3h > 0)
        & (area1v > 0)
        & (area2v > 0)
        & (area3v > 0)
    )

    if not np.any(valid_area):
        return output

    val1h = np.zeros_like(sum1h, dtype=np.float32)
    val2h = np.zeros_like(sum2h, dtype=np.float32)
    val3h = np.zeros_like(sum3h, dtype=np.float32)
    val1v = np.zeros_like(sum1v, dtype=np.float32)
    val2v = np.zeros_like(sum2v, dtype=np.float32)
    val3v = np.zeros_like(sum3v, dtype=np.float32)

    np.divide(sum1h, area1h, out=val1h, where=area1h > 0)
    np.divide(sum2h, area2h, out=val2h, where=area2h > 0)
    np.divide(sum3h, area3h, out=val3h, where=area3h > 0)
    np.divide(sum1v, area1v, out=val1v, where=area1v > 0)
    np.divide(sum2v, area2v, out=val2v, where=area2v > 0)
    np.divide(sum3v, area3v, out=val3v, where=area3v > 0)

    val12 = val1h + val2h + val1v + val2v + 1.0
    val3hv = val3h * val3v

    ratio_term = np.zeros_like(val12, dtype=np.float32)
    np.divide(16.0 * val3hv, val12 * val12 + 1e-6, out=ratio_term)

    detected = (
        (val3hv > config.gradient_threshold**2)
        & (ratio_term > config.ratio_threshold**2)
        & valid_area
    )

    output[np.ix_(y_range, x_range)] = detected.astype(np.uint8) * 255
    return output


# def _get_mosaic_feature_map2(v_channel: np.ndarray, config: MosaicDetectionConfig) -> np.ndarray:
#     """
#     元の _get_mosaic_feature_map と同等の判定式を維持しつつ、
#     行列演算の重複を減らして高速化。出力は 0/255 の uint8 マップ。
#     ※ 完全同一ピクセル一致は保証しませんが、判定式は同一です。
#     """
#     v_float = v_channel.astype(np.float32)
#     H, W = v_float.shape
#
#     # 1階差分（Sobelではなく隣接差分の絶対値）— 元式と同じ
#     grad_h = np.zeros_like(v_float, dtype=np.float32)
#     grad_v = np.zeros_like(v_float, dtype=np.float32)
#     grad_h[:-1, :] = np.abs(v_float[:-1, :] - v_float[1:, :])
#     grad_v[:, :-1] = np.abs(v_float[:, :-1] - v_float[:, 1:])
#
#     # 積分画像（H+1, W+1, CV_32F）
#     ih = cv2.integral(grad_h).astype(np.float32)
#     iv = cv2.integral(grad_v).astype(np.float32)
#
#     pw1 = int(config.gradient_band_height)
#     pw2 = int(config.gradient_band_half_width)
#
#     out = np.zeros((H, W), dtype=np.uint8)
#
#     # 元式と同じ「安全域」
#     x0 = 2 + pw1
#     x1 = W - 3 - pw1
#     y0 = 2 + pw1
#     y1 = H - 3 - pw1
#     if x1 <= x0 or y1 <= y0:
#         return out
#
#     # 評価格子
#     X = np.arange(x0, x1, dtype=np.int32)[None, :]  # (1, NX)
#     Y = np.arange(y0, y1, dtype=np.int32)[:, None]  # (NY, 1)
#
#     # クリップ境界生成（ベクトル化）
#     def _bounds(dx0, dx1, dy0, dy1):
#         x0b = np.clip(X + dx0, 0, W).astype(np.int32)
#         x1b = np.clip(X + dx1, 0, W).astype(np.int32)
#         y0b = np.clip(Y + dy0, 0, H).astype(np.int32)
#         y1b = np.clip(Y + dy1, 0, H).astype(np.int32)
#         return x0b, x1b, y0b, y1b
#
#     # 矩形和（積分画像差分）— ベクトル化
#     def _rect_sum_grid(ii, y0b, y1b, x0b, x1b):
#         s11 = ii[y0b, x0b]
#         s12 = ii[y0b, x1b]
#         s21 = ii[y1b, x0b]
#         s22 = ii[y1b, x1b]
#         rect = (s22 - s12) - (s21 - s11)
#         area = (y1b - y0b) * (x1b - x0b)
#         return rect.astype(np.float32), area.astype(np.float32)
#
#     # 元式の6帯域（水平3 + 垂直3）
#     xb0, xb1, yb0, yb1 = _bounds(-pw2, pw2 + 1, -(2 + pw1), -2)
#     sum1h, area1h = _rect_sum_grid(ih, yb0, yb1, xb0, xb1)
#     xb0, xb1, yb0, yb1 = _bounds(-pw2, pw2 + 1, 3, 3 + pw1)
#     sum2h, area2h = _rect_sum_grid(ih, yb0, yb1, xb0, xb1)
#     xb0, xb1, yb0, yb1 = _bounds(-pw2, pw2 + 1, 0, 1)
#     sum3h, area3h = _rect_sum_grid(ih, yb0, yb1, xb0, xb1)
#
#     xb0, xb1, yb0, yb1 = _bounds(-(2 + pw1), -2, -pw2, pw2 + 1)
#     sum1v, area1v = _rect_sum_grid(iv, yb0, yb1, xb0, xb1)
#     xb0, xb1, yb0, yb1 = _bounds(3, 3 + pw1, -pw2, pw2 + 1)
#     sum2v, area2v = _rect_sum_grid(iv, yb0, yb1, xb0, xb1)
#     xb0, xb1, yb0, yb1 = _bounds(0, 1, -pw2, pw2 + 1)
#     sum3v, area3v = _rect_sum_grid(iv, yb0, yb1, xb0, xb1)
#
#     valid = (area1h > 0) & (area2h > 0) & (area3h > 0) & (area1v > 0) & (area2v > 0) & (area3v > 0)
#     if not np.any(valid):
#         return out
#
#     # 平均化（where 指定でゼロ割回避）
#     def _safe_div(a, b):
#         dst = np.zeros_like(a, dtype=np.float32)
#         np.divide(a, b, out=dst, where=b > 0)
#         return dst
#
#     v1h = _safe_div(sum1h, area1h)
#     v2h = _safe_div(sum2h, area2h)
#     v3h = _safe_div(sum3h, area3h)
#     v1v = _safe_div(sum1v, area1v)
#     v2v = _safe_div(sum2v, area2v)
#     v3v = _safe_div(sum3v, area3v)
#
#     val12  = v1h + v2h + v1v + v2v + 1.0
#     val3hv = v3h * v3v
#     ratio  = (16.0 * val3hv) / (val12 * val12 + 1e-6)
#
#     det = (val3hv > (config.gradient_threshold ** 2)) & (ratio > (config.ratio_threshold ** 2)) & valid
#
#     # 評価領域にだけ反映
#     out[y0:y1, x0:x1] = det.astype(np.uint8) * 255
#     return out


def _checklabel(
    srclines_x: List[int],
    srclines_y: List[int],
    srclines_w: List[int],
    srclines_l: List[int],
    dstlines_x: List[int],
    dstlines_y: List[int],
    dstlines_w: List[int],
    dstlines_l: List[int],
    index: int,
    is_horizontal: bool,
    label: int,
    margin: int,
) -> None:
    if is_horizontal:
        h_x = srclines_x[index]
        h_y = srclines_y[index]
        h_w = srclines_w[index]
    else:
        v_x = srclines_x[index]
        v_y = srclines_y[index]
        v_w = srclines_w[index]

    srclines_l[index] = label

    for k in range(len(dstlines_x)):
        if dstlines_l[k] != 0:
            continue

        if is_horizontal:
            v_x = dstlines_x[k]
            v_y = dstlines_y[k]
            v_w = dstlines_w[k]
        else:
            h_x = dstlines_x[k]
            h_y = dstlines_y[k]
            h_w = dstlines_w[k]

        if (
            h_x - margin <= v_x
            and v_x <= h_x + h_w + margin
            and v_y - margin <= h_y
            and h_y <= v_y + v_w + margin
        ):
            dstlines_l[k] = label
            _checklabel(
                dstlines_x,
                dstlines_y,
                dstlines_w,
                dstlines_l,
                srclines_x,
                srclines_y,
                srclines_w,
                srclines_l,
                k,
                not is_horizontal,
                label,
                margin,
            )


def _get_mosaic_mask(
    binary_map: np.ndarray, config: MosaicDetectionConfig
) -> Tuple[np.ndarray, bool, int]:
    height, width = binary_map.shape
    mask = np.zeros_like(binary_map)
    detected_size = 0

    mosaic_hist_x = np.zeros(config.mosaic_length_max + 1, dtype=np.float32)
    mosaic_hist_y = np.zeros(config.mosaic_length_max + 1, dtype=np.float32)

    limit_y = max(0, height - config.mosaic_length_max)
    limit_x = max(0, width - config.mosaic_length_max)

    for y in range(limit_y):
        for x in range(limit_x):
            if binary_map[y, x] <= 128:
                continue

            x_short = (
                binary_map[y, x + 1 : x + config.mosaic_length_min] > 128
            ).any()
            y_short = (
                binary_map[y + 1 : y + config.mosaic_length_min, x] > 128
            ).any()

            if not x_short:
                for k in range(config.mosaic_length_min, config.mosaic_length_max + 1):
                    if x + k < width and binary_map[y, x + k] > 128:
                        mosaic_hist_x[k] += 1.0
                        break

            if not y_short:
                for k in range(config.mosaic_length_min, config.mosaic_length_max + 1):
                    if y + k < height and binary_map[y + k, x] > 128:
                        mosaic_hist_y[k] += 1.0
                        break

    hist_sum_x = mosaic_hist_x[config.mosaic_length_min :].sum()
    hist_sum_y = mosaic_hist_y[config.mosaic_length_min :].sum()
    hist_max_x = mosaic_hist_x[config.mosaic_length_min :].max(initial=0.0)
    hist_max_y = mosaic_hist_y[config.mosaic_length_min :].max(initial=0.0)

    if (
        hist_sum_x <= 2.0
        or hist_sum_y <= 2.0
        or hist_max_x <= 3.0
        or hist_max_y <= 3.0
    ):
        return mask, False, detected_size

    if hist_sum_x > 0:
        mosaic_hist_x /= hist_sum_x
    if hist_sum_y > 0:
        mosaic_hist_y /= hist_sum_y

    max_index_x = int(np.argmax(mosaic_hist_x[config.mosaic_length_min :])) + config.mosaic_length_min
    max_index_y = int(np.argmax(mosaic_hist_y[config.mosaic_length_min :])) + config.mosaic_length_min

    if (
        mosaic_hist_x[max_index_x] <= config.histogram_threshold
        or mosaic_hist_y[max_index_y] <= config.histogram_threshold
    ):
        return mask, False, detected_size

    cyc_x = [max_index_x, 0, 0]
    cyc_x_num = 1

    if max_index_x == config.mosaic_length_min:
        if mosaic_hist_x[max_index_x + 1] > config.histogram_threshold:
            cyc_x[1] = max_index_x + 1
            cyc_x_num = 2
    elif max_index_x == config.mosaic_length_max:
        if mosaic_hist_x[max_index_x - 1] > config.histogram_threshold:
            cyc_x[0] = max_index_x - 1
            cyc_x[1] = max_index_x
            cyc_x_num = 2
    elif config.mosaic_length_min + 1 <= max_index_x <= config.mosaic_length_max - 1:
        if mosaic_hist_x[max_index_x - 1] > config.histogram_threshold:
            cyc_x[0] = max_index_x - 1
            cyc_x[1] = max_index_x
            cyc_x_num = 2
            if mosaic_hist_x[max_index_x + 1] > config.histogram_threshold:
                cyc_x[2] = max_index_x + 1
                cyc_x_num = 3
        elif mosaic_hist_x[max_index_x + 1] > config.histogram_threshold:
            cyc_x[1] = max_index_x + 1
            cyc_x_num = 2
    else:
        return mask, False, detected_size

    cyc_y = [max_index_y, 0, 0]
    cyc_y_num = 1

    if max_index_y == config.mosaic_length_min:
        if mosaic_hist_y[max_index_y + 1] > config.histogram_threshold:
            cyc_y[1] = max_index_y + 1
            cyc_y_num = 2
    elif max_index_y == config.mosaic_length_max:
        if mosaic_hist_y[max_index_y - 1] > config.histogram_threshold:
            cyc_y[0] = max_index_y - 1
            cyc_y[1] = max_index_y
            cyc_y_num = 2
    elif config.mosaic_length_min + 1 <= max_index_y <= config.mosaic_length_max - 1:
        if mosaic_hist_y[max_index_y - 1] > config.histogram_threshold:
            cyc_y[0] = max_index_y - 1
            cyc_y[1] = max_index_y
            cyc_y_num = 2
            if mosaic_hist_y[max_index_y + 1] > config.histogram_threshold:
                cyc_y[2] = max_index_y + 1
                cyc_y_num = 3
        elif mosaic_hist_y[max_index_y + 1] > config.histogram_threshold:
            cyc_y[1] = max_index_y + 1
            cyc_y_num = 2
    else:
        return mask, False, detected_size

    bin_x = np.zeros_like(binary_map)
    for y in range(height):
        for x in range(width - cyc_x[cyc_x_num - 1] * 2):
            if binary_map[y, x] <= 128:
                continue

            pos_tmp = -1
            for k in range(cyc_x[cyc_x_num - 1] * 2, cyc_x[0] * 2 - 1, -1):
                if x + k < width and binary_map[y, x + k] > 128:
                    pos_tmp = k
                    break

            if pos_tmp < 0:
                for k in range(cyc_x[cyc_x_num - 1], cyc_x[0] - 1, -1):
                    if x + k < width and binary_map[y, x + k] > 128:
                        pos_tmp = k
                        break

            if pos_tmp >= cyc_x[0]:
                for k in range(pos_tmp + 1):
                    if x + k < width:
                        bin_x[y, x + k] = 255

    bin_y = np.zeros_like(binary_map)
    for x in range(width):
        for y in range(height - cyc_y[cyc_y_num - 1] * 2):
            if binary_map[y, x] <= 128:
                continue

            pos_tmp = -1
            for k in range(cyc_y[cyc_y_num - 1] * 2, cyc_y[0] * 2 - 1, -1):
                if y + k < height and binary_map[y + k, x] > 128:
                    pos_tmp = k
                    break

            if pos_tmp < 0:
                for k in range(cyc_y[cyc_y_num - 1], cyc_y[0] - 1, -1):
                    if y + k < height and binary_map[y + k, x] > 128:
                        pos_tmp = k
                        break

            if pos_tmp >= cyc_y[0]:
                for k in range(pos_tmp + 1):
                    if y + k < height:
                        bin_y[y + k, x] = 255

    xlines_x: List[int] = []
    xlines_y: List[int] = []
    xlines_w: List[int] = []
    xlines_l: List[int] = []

    for y in range(height):
        line_start_flag = False
        for x in range(1, width):
            if (
                not line_start_flag
                and bin_x[y, x] > 128
                and bin_x[y, x - 1] <= 128
            ):
                xlines_x.append(x)
                xlines_y.append(y)
                xlines_w.append(0)
                xlines_l.append(0)
                line_start_pos = x
                line_start_flag = True

            if (
                line_start_flag
                and bin_x[y, x] <= 128
                and bin_x[y, x - 1] > 128
            ):
                xlines_w[-1] = x - line_start_pos
                line_start_flag = False

        if line_start_flag and xlines_w:
            xlines_w[-1] = max(1, width - line_start_pos)

    ylines_x: List[int] = []
    ylines_y: List[int] = []
    ylines_w: List[int] = []
    ylines_l: List[int] = []

    for x in range(width):
        line_start_flag = False
        for y in range(1, height):
            if (
                not line_start_flag
                and bin_y[y, x] > 128
                and bin_y[y - 1, x] <= 128
            ):
                ylines_x.append(x)
                ylines_y.append(y)
                ylines_w.append(0)
                ylines_l.append(0)
                line_start_pos = y
                line_start_flag = True

            if (
                line_start_flag
                and bin_y[y, x] <= 128
                and bin_y[y - 1, x] > 128
            ):
                ylines_w[-1] = y - line_start_pos
                line_start_flag = False

        if line_start_flag and ylines_w:
            ylines_w[-1] = max(1, height - line_start_pos)

    if len(xlines_x) <= 1 or len(ylines_x) <= 1:
        return mask, False, detected_size

    label = 0
    for idx in range(len(xlines_l)):
        if xlines_l[idx] == 0:
            label += 1
            _checklabel(
                xlines_x,
                xlines_y,
                xlines_w,
                xlines_l,
                ylines_x,
                ylines_y,
                ylines_w,
                ylines_l,
                idx,
                True,
                label,
                config.intersection_margin,
            )

    if label == 0:
        return mask, False, detected_size

    hist_label = [0] * (label + 1)
    for val in xlines_l:
        if 0 < val <= label:
            hist_label[val] += 1
    for val in ylines_l:
        if 0 < val <= label:
            hist_label[val] += 1

    hist_index = [0, 0]
    maxval = [-1, -1]
    for idx in range(1, label + 1):
        if hist_label[idx] > 10:
            if hist_label[idx] > maxval[0]:
                hist_index[1] = hist_index[0]
                hist_index[0] = idx
                maxval[1] = maxval[0]
                maxval[0] = hist_label[idx]
            elif hist_label[idx] > maxval[1]:
                hist_index[1] = idx
                maxval[1] = hist_label[idx]

    if maxval[0] < 0:
        return mask, False, detected_size

    mosaic_count = 1
    y_min = [math.inf, math.inf]
    y_max = [-math.inf, -math.inf]

    for L in range(mosaic_count):
        idx = hist_index[L]
        for k in range(len(ylines_l)):
            if ylines_l[k] == idx:
                y_max[L] = max(y_max[L], ylines_y[k] + ylines_w[k])
                y_min[L] = min(y_min[L], ylines_y[k])

        if not np.isfinite(y_min[L]) or not np.isfinite(y_max[L]):
            return mask, False, detected_size

        if y_max[L] - y_min[L] < cyc_y[0]:
            return mask, False, detected_size

        minlines = [0] * (int(y_max[L]) - int(y_min[L]) + 1)
        maxlines = [0] * (int(y_max[L]) - int(y_min[L]) + 1)

        for y in range(int(y_min[L]), int(y_max[L]) + 1):
            minval_tmp = math.inf
            maxval_tmp = -math.inf
            for k in range(len(ylines_l)):
                if ylines_l[k] == idx and ylines_y[k] <= y <= ylines_y[k] + ylines_w[k]:
                    if ylines_x[k] > maxval_tmp:
                        maxval_tmp = ylines_x[k]
                    if ylines_x[k] < minval_tmp:
                        minval_tmp = ylines_x[k]

            if maxval_tmp == -math.inf and y > int(y_min[L]):
                maxlines[y - int(y_min[L])] = maxlines[y - int(y_min[L]) - 1]
            else:
                maxlines[y - int(y_min[L])] = int(maxval_tmp if maxval_tmp != -math.inf else 0)

            if minval_tmp == math.inf and y > int(y_min[L]):
                minlines[y - int(y_min[L])] = minlines[y - int(y_min[L]) - 1]
            else:
                minlines[y - int(y_min[L])] = int(minval_tmp if minval_tmp != math.inf else 0)

        cyc_y_half = cyc_y[0] / 2.0
        y_min_tmp = int(y_min[L])
        y_max_tmp = int(y_max[L])

        while (
            y_min_tmp <= y_max_tmp
            and (maxlines[y_min_tmp - int(y_min[L])] - minlines[y_min_tmp - int(y_min[L])]) < 4
        ):
            y_min_tmp += 1

        while (
            y_max_tmp >= y_min_tmp
            and (maxlines[y_max_tmp - int(y_min[L])] - minlines[y_max_tmp - int(y_min[L])]) < cyc_y_half
        ):
            y_max_tmp -= 1

        y_min_tmp = int(math.ceil(y_min_tmp / 8.0) * 8)
        y_max_tmp = int(math.floor(y_max_tmp / 8.0) * 8)

        if y_max_tmp - y_min_tmp + 1 < 10:
            continue

        minlines2 = [0] * (y_max_tmp - y_min_tmp + 1)
        maxlines2 = [0] * (y_max_tmp - y_min_tmp + 1)

        for y in range(y_min_tmp, y_max_tmp + 1):
            minlines2[y - y_min_tmp] = minlines[y - int(y_min[L])]
            maxlines2[y - y_min_tmp] = maxlines[y - int(y_min[L])]

        for idx_y in range(len(minlines2)):
            maxlines2[idx_y] = int(math.ceil((maxlines2[idx_y] + 0.5) / 8.0) * 8)
            minlines2[idx_y] = int(math.floor((minlines2[idx_y] - 0.5) / 8.0) * 8)

        for y in range(y_min_tmp, y_max_tmp + 1):
            x_min = max(0, minlines2[y - y_min_tmp])
            x_max = min(width - 1, maxlines2[y - y_min_tmp])
            if x_max >= x_min:
                mask[y, x_min : x_max + 1] = 255

    detected_size = max(1, int(round((cyc_x[0] + cyc_y[0]) / 2)))

    return mask, bool(mask.any()), detected_size


def _rgb_to_hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB image in [0, 1] to OpenCV-style HSV (H:[0,180], S/V:[0,255])."""

    r, g, b = rgb.unbind(dim=-1)
    maxc, _ = torch.max(rgb, dim=-1)
    minc, _ = torch.min(rgb, dim=-1)
    delta = maxc - minc

    # Hue calculation
    hue = torch.zeros_like(maxc)
    mask = delta > 1e-6
    delta_safe = torch.where(mask, delta, torch.ones_like(delta))

    hue_r = ((g - b) / delta_safe) % 6.0
    hue_g = ((b - r) / delta_safe) + 2.0
    hue_b = ((r - g) / delta_safe) + 4.0

    hue = torch.where((maxc == r) & mask, hue_r, hue)
    hue = torch.where((maxc == g) & mask, hue_g, hue)
    hue = torch.where((maxc == b) & mask, hue_b, hue)
    hue = (hue * 60.0) % 360.0

    # Saturation and value
    saturation = torch.where(maxc > 1e-6, delta / torch.clamp(maxc, min=1e-6), torch.zeros_like(maxc))
    value = maxc

    hsv = torch.stack((hue / 2.0, saturation * 255.0, value * 255.0), dim=-1)
    return hsv


def _get_mosaic_feature_map_torch(
    v_channel: torch.Tensor, config: MosaicDetectionConfig
) -> torch.Tensor:
    v_float = v_channel.to(dtype=torch.float32)
    height, width = v_float.shape

    grad_h = torch.zeros_like(v_float)
    grad_v = torch.zeros_like(v_float)
    grad_h[:-1, :] = torch.abs(v_float[:-1, :] - v_float[1:, :])
    grad_v[:, :-1] = torch.abs(v_float[:, :-1] - v_float[:, 1:])

    integral_h = torch.zeros((height + 1, width + 1), dtype=torch.float32, device=v_float.device)
    integral_v = torch.zeros((height + 1, width + 1), dtype=torch.float32, device=v_float.device)
    integral_h[1:, 1:] = grad_h
    integral_v[1:, 1:] = grad_v
    integral_h = integral_h.cumsum(0).cumsum(1)
    integral_v = integral_v.cumsum(0).cumsum(1)

    pw1 = config.gradient_band_height
    pw2 = config.gradient_band_half_width

    output = torch.zeros_like(v_channel, dtype=torch.uint8)

    x_range = torch.arange(2 + pw1, width - 3 - pw1, device=v_float.device)
    y_range = torch.arange(2 + pw1, height - 3 - pw1, device=v_float.device)

    if x_range.numel() == 0 or y_range.numel() == 0:
        return output

    X = x_range.unsqueeze(0)
    Y = y_range.unsqueeze(1)

    def _clipped_bounds(offset_x0: int, offset_x1: int, offset_y0: int, offset_y1: int):
        x0 = torch.clamp(X + offset_x0, 0, width)
        x1 = torch.clamp(X + offset_x1, 0, width)
        y0 = torch.clamp(Y + offset_y0, 0, height)
        y1 = torch.clamp(Y + offset_y1, 0, height)
        return x0.to(torch.int64), x1.to(torch.int64), y0.to(torch.int64), y1.to(torch.int64)

    def _rect_sum_torch(integral: torch.Tensor, x0, x1, y0, y1):
        sum_val = (
            integral[y1, x1]
            - integral[y0, x1]
            - integral[y1, x0]
            + integral[y0, x0]
        )
        area = (x1 - x0) * (y1 - y0)
        return sum_val, area

    x0, x1, y0, y1 = _clipped_bounds(-pw2, pw2 + 1, -(2 + pw1), -2)
    sum1h, area1h = _rect_sum_torch(integral_h, x0, x1, y0, y1)

    x0, x1, y0, y1 = _clipped_bounds(-pw2, pw2 + 1, 3, 3 + pw1)
    sum2h, area2h = _rect_sum_torch(integral_h, x0, x1, y0, y1)

    x0, x1, y0, y1 = _clipped_bounds(-pw2, pw2 + 1, 0, 1)
    sum3h, area3h = _rect_sum_torch(integral_h, x0, x1, y0, y1)

    x0, x1, y0, y1 = _clipped_bounds(-(2 + pw1), -2, -pw2, pw2 + 1)
    sum1v, area1v = _rect_sum_torch(integral_v, x0, x1, y0, y1)

    x0, x1, y0, y1 = _clipped_bounds(3, 3 + pw1, -pw2, pw2 + 1)
    sum2v, area2v = _rect_sum_torch(integral_v, x0, x1, y0, y1)

    x0, x1, y0, y1 = _clipped_bounds(0, 1, -pw2, pw2 + 1)
    sum3v, area3v = _rect_sum_torch(integral_v, x0, x1, y0, y1)

    valid_area = (
        (area1h > 0)
        & (area2h > 0)
        & (area3h > 0)
        & (area1v > 0)
        & (area2v > 0)
        & (area3v > 0)
    )

    if not torch.any(valid_area):
        return output

    eps = torch.finfo(torch.float32).eps

    val1h = torch.zeros_like(sum1h, dtype=torch.float32)
    val2h = torch.zeros_like(sum2h, dtype=torch.float32)
    val3h = torch.zeros_like(sum3h, dtype=torch.float32)
    val1v = torch.zeros_like(sum1v, dtype=torch.float32)
    val2v = torch.zeros_like(sum2v, dtype=torch.float32)
    val3v = torch.zeros_like(sum3v, dtype=torch.float32)

    val1h = torch.where(area1h > 0, sum1h / torch.clamp(area1h, min=eps), val1h)
    val2h = torch.where(area2h > 0, sum2h / torch.clamp(area2h, min=eps), val2h)
    val3h = torch.where(area3h > 0, sum3h / torch.clamp(area3h, min=eps), val3h)
    val1v = torch.where(area1v > 0, sum1v / torch.clamp(area1v, min=eps), val1v)
    val2v = torch.where(area2v > 0, sum2v / torch.clamp(area2v, min=eps), val2v)
    val3v = torch.where(area3v > 0, sum3v / torch.clamp(area3v, min=eps), val3v)

    val12 = val1h + val2h + val1v + val2v + 1.0
    val3hv = val3h * val3v

    ratio_term = (16.0 * val3hv) / (val12 * val12 + eps)

    detected = (
        (val3hv > config.gradient_threshold**2)
        & (ratio_term > config.ratio_threshold**2)
        & valid_area
    )

    output[y_range.unsqueeze(1), x_range] = detected.to(torch.uint8) * 255
    return output


def _detect_mosaic_mask_numpy(
    image_bgr: np.ndarray, config: MosaicDetectionConfig
) -> Tuple[np.ndarray, int]:
    """Detect mosaic regions and return a uint8 mask and detected tile size."""

    if image_bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 image in BGR order")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    small_size = (max(1, hsv.shape[1] // 2), max(1, hsv.shape[0] // 2))
    hsv_small = cv2.resize(hsv, small_size, interpolation=cv2.INTER_NEAREST)

    skin_mask = _get_skin_mask_from_hsv(hsv_small, config)
    v_channel = hsv_small[:, :, 2]

    mosaic_feature = _get_mosaic_feature_map(v_channel, config)
    mosaic_feature = cv2.bitwise_and(mosaic_feature, skin_mask)

    mask_small, has_mosaic, mosaic_size = _get_mosaic_mask(mosaic_feature, config)

    if not has_mosaic:
        return (
            np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8),
            0,
        )

    full_mask = cv2.resize(
        mask_small,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return full_mask, mosaic_size * 2


def _detect_mosaic_mask_torch(
    image_rgb: torch.Tensor, config: MosaicDetectionConfig
) -> Tuple[np.ndarray, int]:
    if image_rgb.dtype != torch.float32:
        image_rgb = image_rgb.to(dtype=torch.float32)

    image_rgb = torch.clamp(image_rgb, 0.0, 1.0)

    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError("Expected RGB tensor with shape (H, W, 3)")

    height, width, _ = image_rgb.shape
    small_height = max(1, height // 2)
    small_width = max(1, width // 2)

    rgb_chw = image_rgb.permute(2, 0, 1).unsqueeze(0)
    rgb_small = F.interpolate(
        rgb_chw,
        size=(small_height, small_width),
        mode="nearest",
    ).squeeze(0).permute(1, 2, 0)

    hsv_small = _rgb_to_hsv_torch(rgb_small)

    h_channel = hsv_small[..., 0]
    s_channel = hsv_small[..., 1]

    mask = (
        (h_channel <= config.hsv_skin_h_low)
        | (h_channel >= config.hsv_skin_h_high)
        | (s_channel <= config.hsv_skin_s_threshold)
    ).to(torch.uint8)
    mask = mask * 255

    v_channel = hsv_small[..., 2]
    feature_map = _get_mosaic_feature_map_torch(v_channel, config)
    feature_map = torch.minimum(feature_map, mask)

    mask_small, has_mosaic, mosaic_size = _get_mosaic_mask(
        feature_map.cpu().numpy(), config
    )

    if not has_mosaic:
        return (
            np.zeros((height, width), dtype=np.uint8),
            0,
        )

    full_mask = cv2.resize(
        mask_small,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return full_mask, mosaic_size * 2


class MosaicDetectionNode:
    """ComfyUI node that exposes :func:`detect_mosaic_mask`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hsv_skin_h_low": ("INT", {"default": 35, "min": 0, "max": 180}),
                "hsv_skin_h_high": ("INT", {"default": 160, "min": 0, "max": 180}),
                "hsv_skin_s_threshold": ("INT", {"default": 25, "min": 0, "max": 255}),
                "gradient_threshold": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0}),
                "ratio_threshold": ("FLOAT", {"default": 2.74, "min": 0.0, "max": 20.0}),
                "histogram_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "mosaic_length_min": ("INT", {"default": 4, "min": 2, "max": 64}),
                "mosaic_length_max": ("INT", {"default": 25, "min": 3, "max": 128}),
                "intersection_margin": ("INT", {"default": 5, "min": 0, "max": 50}),
                "gradient_band_height": ("INT", {"default": 1, "min": 1, "max": 8}),
                "gradient_band_half_width": ("INT", {"default": 15, "min": 1, "max": 64}),
                "processing_backend": (
                    "STRING",
                    {
                        "default": "AUTO",
                        "choices": ["AUTO", "CPU", "TORCH"],
                    },
                ),
                "max_workers": ("INT", {"default": 0, "min": 0, "max": 32}),
            }
        }

    RETURN_TYPES = ("MASK", "INT")
    FUNCTION = "execute"
    CATEGORY = "fast-mosaic-detector"

    def execute(
        self,
        image: np.ndarray,
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
        processing_backend: str,
        max_workers: int,
    ) -> Tuple[torch.Tensor, int]:
        config = MosaicDetectionConfig(
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

        if isinstance(image, torch.Tensor):
            batch_tensor = image.detach()
        else:
            batch_tensor = torch.from_numpy(np.asarray(image))

        if batch_tensor.ndim == 3:
            if batch_tensor.shape[0] in (1, 3) and batch_tensor.shape[-1] != 3:
                batch_tensor = batch_tensor.permute(1, 2, 0).unsqueeze(0)
            else:
                batch_tensor = batch_tensor.unsqueeze(0)
        elif batch_tensor.ndim == 4:
            if batch_tensor.shape[-1] != 3 and batch_tensor.shape[1] in (1, 3):
                batch_tensor = batch_tensor.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                "Expected image tensor with 3 (HWC/CHW) or 4 (NHWC/NCHW) dimensions"
            )

        if batch_tensor.shape[-1] != 3:
            raise ValueError("Image input must have 3 channels")

        batch_tensor = batch_tensor.to(dtype=torch.float32)
        device = batch_tensor.device

        backend_choice = processing_backend.upper()
        if backend_choice == "AUTO":
            use_torch_backend = batch_tensor.is_cuda
        elif backend_choice == "TORCH":
            use_torch_backend = True
        else:
            use_torch_backend = False

        masks: List[np.ndarray] = []
        sizes: List[int] = []

        if use_torch_backend:
            for frame in batch_tensor:
                mask, mosaic_size = _detect_mosaic_mask_torch(frame, config)
                masks.append(mask.astype(np.float32) / 255.0)
                sizes.append(int(mosaic_size))
        else:
            frames_np = batch_tensor.cpu().numpy()
            frames_np = np.clip(frames_np, 0.0, 1.0)
            frames_bgr = (frames_np[..., ::-1] * 255.0).astype(np.uint8)

            worker = partial(_detect_mosaic_mask_numpy, config=config)
            batch_size = frames_bgr.shape[0]

            if batch_size == 1:
                results = [worker(frames_bgr[0])]
            else:
                worker_count = max_workers if max_workers > 0 else min(
                    os.cpu_count() or 1, batch_size
                )
                if worker_count <= 1:
                    results = [worker(frame) for frame in frames_bgr]
                else:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        results = list(executor.map(worker, list(frames_bgr)))

            for mask, mosaic_size in results:
                masks.append(mask.astype(np.float32) / 255.0)
                sizes.append(int(mosaic_size))

        mask_batch = np.stack(masks, axis=0)
        nonzero_sizes = [size for size in sizes if size > 0]
        if nonzero_sizes:
            size_value = int(round(float(np.median(nonzero_sizes))))
        else:
            size_value = 0

        mask_tensor = torch.from_numpy(mask_batch)
        if use_torch_backend and device.type != "cpu":
            mask_tensor = mask_tensor.to(device)

        return (mask_tensor, size_value)


NODE_CLASS_MAPPINGS = {"MosaicDetectionNode": MosaicDetectionNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MosaicDetectionNode": "Mosaic Detection"}