# PigPredict — 智慧配色工作台

**[English](README.md) · [開啟互動展示](https://yellowelephanthive.github.io/PigPredict/)**

PigPredict 是將染整製程知識與深度學習結合的紡織配色專案。輸入目標色的 CIELAB 數值與布種，系統從歷史配方挑選染料組合，再透過已訓練的神經網路反推濃度，讓使用者比較不同的候選配方。

原始專案開發於 2018–2019 年，使用 Keras、TensorFlow 與 Django。2026 年的更新保留原始模型權重，將表單與表格介面整理成可互動、可展示的配色工作台，並補上模型轉換、測試與工程文件。

## 互動展示

- **目標色設定：**輸入 L*、a*、b* 與布種，或載入四組實測樣本。
- **候選配方比較：**搜尋最多八組歷史染料組合，依模型預測色差排序。
- **即時濃度調整：**使用滑桿或數值欄位調整濃度，立即查看預測色與 ΔL*、Δa*、Δb*。
- **配方匯出：**CSV 包含染料、濃度、目標 Lab、預測 Lab、預測色差及是否經過手動調整。
- **雙語與行動版：**每次載入預設英文，右上角提供醒目的 English／中文切換。

網頁直接執行原始訓練模型，配方搜尋在 Web Worker 中運算。GitHub Pages 展示不需要額外的推論伺服器。

## 專案中的機器學習

| 項目 | 內容 |
| --- | --- |
| 模型輸入 | 44 個染料濃度欄位與 3 個布種 one-hot 編碼 |
| 模型輸出 | 正規化後的 L*、a*、b*，再還原為 Lab |
| 網路架構 | `47 → 128 × 7 → 32 → 16 → 3` |
| 啟動函數 | LeakyReLU，α = 0.5 |
| 訓練參數 | 109,923 |
| 支援布種 | FVF2184、FVF2429、FVF2666 |
| 展示資料 | 1,417 筆經篩選、去重的歷史配方紀錄 |
| 配方排序 | 目標色與模型預測色的歐氏距離 ΔE₇₆ |

原始方法固定模型權重，使用 TensorFlow 1.x 的 Adam 從多組初始濃度進行反向搜尋。新版固定相同權重，使用具濃度上下界的確定性阻尼最小平方搜尋，讓配方探索可以直接在瀏覽器執行。這次更新沒有重新訓練模型。

## 本機執行

只需要展示頁時，不必安裝 Django：

```bash
git clone https://github.com/YellowElephantHive/PigPredict.git
cd PigPredict
python3 -m http.server 8771 --bind 127.0.0.1 --directory docs
```

瀏覽 `http://127.0.0.1:8771/`。請透過 HTTP 開啟，讓瀏覽器能載入模型與 Web Worker。

執行 Django 與 Python 推論 API，使用 Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8771
```

Windows 啟用環境時改用 `.venv\Scripts\activate`。目前工作台不會將每次輸入寫入資料庫；需要 Django 管理功能時，資料庫使用忽略追蹤的 `.runtime/` 目錄。

## 工程驗證

```bash
python -m pip install -r requirements-dev.txt
python manage.py check
python -m pytest tests -q
node --test tests/engine.test.mjs
```

14 項 Python 測試涵蓋原始 HDF5 權重、NumPy 與瀏覽器推論比對、資料輸出欄位、Django 路由、API 輸入驗證與 CSRF。4 項 Node 測試涵蓋導數驗證、搜尋的可重現性、布種隔離、濃度邊界及色彩轉換。

所有 1,417 筆展示配方都以原 HDF5 權重獨立計算，與 NumPy、JavaScript 結果比對，Lab 絕對誤差容許值為 `1e-8`。這項檢查驗證模型移植的一致性。

## 文件

- [架構與演算法](docs/ARCHITECTURE.md)：模型來源、正規化、逆向搜尋與資料流程。
- [展示步驟](docs/DEMO.md)：面試時可以使用的操作順序與說明重點。
- [更新紀錄](docs/MODERNIZATION.md)：舊版與新版的對照、保留內容與驗證項目。

修改 `templates/PigPredict_app/studio.html` 或 `static/studio/` 後，執行 `python scripts/build_demo.py` 更新 GitHub Pages 靜態版。完整目錄與 API 用法請見 [English README](README.md)。
