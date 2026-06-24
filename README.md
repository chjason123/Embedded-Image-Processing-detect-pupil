# 雙瞳孔偵測與批次處理系統 (Dual Pupil Detection & Batch Processing)

本專案基於 OpenCV 與 NumPy 開發，專門用於自動化識別與標記影像中的**雙眼瞳孔（左眼與右眼）**。程式支援自動化批次處理，能一鍵讀取指定資料夾內的所有眼部影像，進行去噪、影像二值化與輪廓分析，最終將偵測結果標記並自動儲存。

---

## 📌 功能特點

* **雙瞳孔偵測**：打破傳統單眼偵測限制，利用幾何中心（Centroid）與距離排序演算法，精準鎖定影像中的雙眼瞳孔。
* **影像前處理流程**：結合非局部均值去噪（Non-Local Means Denoising）、高斯模糊（Gaussian Blur）與形態學侵蝕（Erosion），有效過濾睫毛、反光等雜訊。
* **自動化批次處理**：自動遍歷指定資料夾，支援常見格式（`.jpg`, `.png`, `.bmp` 等），偵測後自動加上 `processed_` 前綴並匯出。
* **靈活防呆機制**：
* 自動建立不存在的輸出資料夾。
* 批次處理時自動關閉視窗彈出，避免程式中斷卡死。



---

## 🛠️ 程式架構與核心邏輯說明

程式主要由 `pupil_detection` 類別與主程式批次執行區域組成：

### 1. 影像重心計算 (`centroid`)

在進行前處理前，程式會先將原圖轉為灰階並進行基礎二值化，計算整張圖的**影像矩（Moments）**，藉此取得影像的幾何中心點 $C(x, y)$。這個中心點將作為後續篩選雙眼位置的重要基準。

### 2. 瞳孔核心偵測演算法 (`detect_pupil`)

此為本程式的核心處理管道（Pipeline），步驟如下：

1. **去噪與模糊**：使用 `fastNlMeansDenoisingColored` 與 `GaussianBlur` 消除影像細微雜訊與毛邊。
2. **影像反相與二值化**：將影像反相（黑白對調），讓原本暗色系的瞳孔區域變成高亮度的白色區塊，並透過門檻值（Threshold = 210）過濾掉多餘背景。
3. **輪廓擷取**：利用 `findContours` 找出所有白色區塊的輪廓。
4. **雜訊篩選與距離排序**：
* 排除半徑小於 3 像素或大於 100 像素的異常輪廓。
* 計算每個輪廓中心與步驟 1 所得之**影像重心的絕對距離（Manhattan Distance）**。
* 將所有輪廓依距離**由小到大排序**，並精準擷取前 2 個最接近中心的輪廓（即為左眼與右眼）。


5. **繪製與紀錄**：在原圖上繪製藍色圓圈與 `Pupil 1` / `Pupil 2` 標籤，並將座標與半徑資料存入 `self._pupils`。

---

## 💻 程式碼方法（Methods）清單

| 方法名稱 | 參數 | 功能描述 |
| --- | --- | --- |
| `__init__` | `image_path` | 初始化類別，設定影像路徑並配置資料儲存變數。 |
| `load_image` | 無 | 讀取指定路徑的影像，並回傳是否讀取成功（布林值）。 |
| `show_image` | `img` | 彈出視窗顯示影像結果，按任意鍵後關閉。 |
| `centroid` | 無 | 計算影像的幾何中心點（Centroid），存於 `self._centroid`。 |
| `detect_pupil` | `save_path=None` | 執行核心前處理、輪廓分析、雙瞳篩選與標記繪製。 |
| `start_detection` | `save_path=None` | 串聯 `load_image` ➔ `centroid` ➔ `detect_pupil` 的高階驅動方法。 |

---

## 🚀 使用指南

### 1. 環境需求

請確保您的 Python 環境已安裝以下套件：

```bash
pip install opencv-python numpy

```

### 2. 資料夾結構配置

請在您的程式碼相同目錄下，建立一個名為 `Eye Images` 的資料夾，並將欲處理的眼部圖片放入其中：

```text
.
├── pupil_detection.py      # 本程式碼
├── Eye Images/             # 放入原始眼部影像 (輸入)
│   ├── eye_01.jpg
│   └── eye_02.png
└── Eye Images Output/      # 執行後自動生成 (輸出)
    ├── processed_eye_01.jpg
    └── processed_eye_02.png

```

### 3. 執行程式

直接執行 Python 檔案即可開始批次處理：

```bash
python pupil_detection.py

```

### 4. 終端機輸出範例

執行時，終端機（Terminal）會即時回報每張圖片的偵測數據：

```text
已建立輸出資料夾：Eye Images Output
[eye_01.jpg] 成功偵測到 2 個瞳孔： [(142, 210, 25), (385, 208, 24)]
[eye_02.png] 成功偵測到 2 個瞳孔： [(150, 198, 22), (390, 201, 23)]

批次處理完成！共處理了 2 張圖片，結果已儲存至：Eye Images Output

```

---
