# ComfyUI Fast Mosaic Detector

📗 **日本語** | 📘 [English](README.md)

ComfyUI 用の高精度・高速モザイク検出ノードです。  
FAST / ACCURATE / HYBRID の 3 モードを搭載し、HYBRID では  
**ACCURATE に近い精度を維持したまま最大 7 倍高速化** を実現します。

---

## 🔍 概要

本ノードは画像／動画フレームからモザイク（ブロックノイズ・ぼかし）を検出し、

- **0/255 のバイナリマスク**
- **推定モザイクサイズ**

を出力します。
[README.md](README.md)
内部的には以下の2系統の検出器を組み合わせています：

- **Bモード（FAST）**：CUDA対応の高速検出器  
- **Dモード（ACCURATE）**：勾配・ヒストグラム・格子パターン解析による精密検出  
- **HYBRID**：FASTで候補抽出 → Dで精査する最適化版

HYBRIDは  
✅ ACCURATE の 80–95% 程度の精度  
✅ 約 1/7 の時間  
という実践的に最も優れたパフォーマンスです。

---

## Workflow Screenshot
![Workflow Screenshot](assets/examples/ScreenShot.jpg)

## Input Video

https://github.com/user-attachments/assets/17b95590-77cc-435e-8df1-6cf04604c46b

[example_input.mp4](assets/examples/example_input.mp4)

## Output Video (Image&Mask Blend)

https://github.com/user-attachments/assets/28731ee8-8a11-4971-af86-5be63b44a870

[example_output.mp4](assets/examples/example_output.mp4)


---

## Example Workflow

You can find the example workflow here:

💾 [FastMosaicDetectorExample.json](example_workflow/FastMosaicDetectorExample.json)

---


# ✅ 特徴

### ✅ FAST モード
- 約3秒（80フレームの場合）  
- 誤検出気味になることも  
- とにかく素早く結果を見たい場合に

### ✅ ACCURATE モード
- 約420秒  
- 最高の精度  
- 全画面を詳細解析するため重い

### ✅ HYBRID モード（推奨）
- 約40〜60秒  
- ACCURATE に近い品質  
- 実用上最もバランスが良い

---

# ✅ 入力パラメータ

以下、ノードの各パラメータを詳しく説明します。

---

# 🎛 ACCURATE（D）モードの主パラメータ

## ✅ 肌色領域抽出（HSV）
| パラメータ | 説明 |
|-----------|------|
| `hsv_skin_h_low` | Hue下限 |
| `hsv_skin_h_high` | Hue上限 |
| `hsv_skin_s_threshold` | Saturation下限 |

主に肌色領域での精度向上を狙います。

---

## ✅ 勾配解析
| パラメータ | 説明 |
|-----------|------|
| `gradient_threshold` | 勾配強度しきい値 |
| `ratio_threshold` | モザイクのパターン比しきい値 |
| `gradient_band_height` | 勾配抽出帯の高さ |
| `gradient_band_half_width` | 勾配抽出帯の横幅 |

---

## ✅ ヒストグラム・格子パターン解析
| パラメータ | 説明 |
|-----------|------|
| `histogram_threshold` | 局所ヒストグラムのピーク比 |
| `mosaic_length_min` | モザイク最小ブロックサイズ |
| `mosaic_length_max` | モザイク最大ブロックサイズ |
| `intersection_margin` | 格子の重なり許可幅 |

---

# 🚀 実行モード
mode = FAST / ACCURATE / HYBRID

用途に応じて選択します。HYBRIDがおすすめ。

---

# ⚙ バックエンド
processing_backend = AUTO / CPU / TORCH

- AUTO：最適設定  
- TORCH：CUDA必須（FASTで利用）  

---

# 🔧 実行系設定
### `max_workers`
ACCURATE の CPU スレッド数。

---

# 🟦 HYBRIDモード専用パラメータ

### ✅ `fast_recall_boost`
FAST の検出率を引き上げる係数。

### ✅ `roi_margin_px`
ROI を広げるピクセル数。

### ✅ `refine_logic`
| 値 | 意味 |
|----|------|
| replace | FAST マスクを置き換える |
| union | FAST と精査結果を統合 |
| intersect | 重複部分のみ採用 |

### ✅ その他
| パラメータ | 説明 |
|-----------|------|
| `refine_frame_stride` | 動画で何フレームおきに精査するか |
| `roi_merge_dilate_px` | ROI を膨張して統合 |
| `roi_max_count` | ROI の最大数 |
| `min_mask_pixels` | FAST マスク最低数 |
| `frame_cover_threshold` | 過検出扱いの閾値 |

---

# 🟩 アダプティブ ROI（自動縮小）

| パラメータ | 説明 |
|-----------|------|
| `adaptive_roi_area_ratio` | ROI がフレームの何%超なら縮小するか |
| `adaptive_roi_min_side` | この短辺未満なら縮小しない |
| `roi_downscale_large` | 縮小倍率（0.75 推奨） |

---

# 🟨 アスペクト比フィルタ

| パラメータ | 説明                   |
|-----------|----------------------|
| `roi_aspect_ratio_max` | 細長すぎる ROI を除外（3～5推奨） |
| `roi_min_short_side` | 小さすぎる ROI を除外        |

格子や窓枠などの誤検出を防ぎます。

---

# ✅ 出力

- `mask` — 0 または 255 のバイナリマスク  
- `size` — 推定モザイクサイズ  

---

# ✅ 推奨プリセット

## ✅ 実用最適
mode = HYBRID  
roi_downscale_large = 0.75  
roi_aspect_ratio_max = 3.0  
fast_recall_boost = 0.9  

## ✅ 最高精度
mode = ACCURATE

## ✅ クイックプレビュー
mode = FAST

---

# ✅ 備考

- 小さなモザイクは縮小禁止（roi_downscale_large=1.0）  
- アニメ系はHSVのしきい値を広く  
- HYBRID の検出漏れがある場合 → fast_recall_boost を上げる  
- このノードは、ビデオフレームの読み込みと処理済みビデオ出力の保存において、ComfyUI-VideoHelperSuiteと完全に互換性があります。ビデオワークフローでこのノードを使用する場合は、VideoHelperSuiteのインストールを強く推奨します。


---

# ✅ ライセンス
MIT

# ✅ 作者
Takahiro Yano  
Mosaic Detector Hybrid BD (ComfyUI)
