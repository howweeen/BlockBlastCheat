# Block Blast 自動遊玩 AI — 架構規劃書

## Context（背景）

這是一個全新（greenfield）專案，目標是打造一個能自動遊玩 **Block Blast**（8x8 棋盤、每回合給 3 個方塊、填滿整行/整列即消除）的 AI，運行於電腦端的 Android 模擬器上。

技術限制與決策：
- **語言/環境**：Python 3.10+
- **影像辨識**：OpenCV
- **裝置操作**：ADB（本地連線）
- **矩陣運算**：NumPy
- **演算法**：貪婪演算法 + 啟發式評估函數（Heuristic Search），**不使用強化學習**

本文件為架構藍圖與開發步驟，不含實作程式碼。

---

## 1. 目錄與檔案結構

```text
BlockBlastCheat/
├── README.md                      # 專案說明、安裝與執行方式
├── requirements.txt               # 相依套件清單 (opencv-python, numpy, 等)
├── pyproject.toml                 # 專案 metadata 與工具設定 (ruff/pytest)
├── config/
│   ├── settings.yaml              # 全域設定：ADB 裝置 ID、執行頻率、除錯開關
│   └── calibration.yaml           # 棋盤/方塊區的螢幕座標校準資料 (像素邊界)
├── assets/
│   └── templates/                 # 方塊顏色/形狀的樣板圖，供影像比對使用
├── src/
│   └── block_blast/
│       ├── __init__.py
│       ├── main.py                # 程式進入點：串接四大模組的主迴圈
│       ├── vision/
│       │   ├── __init__.py
│       │   ├── capture.py         # 透過 ADB 擷取模擬器螢幕截圖 (回傳 ndarray)
│       │   ├── board_reader.py    # 將截圖辨識為 8x8 棋盤矩陣
│       │   └── piece_reader.py    # 辨識當前 3 個待放置方塊的形狀
│       ├── core/
│       │   ├── __init__.py
│       │   ├── state.py           # 定義 GameState / Piece / Move 等資料結構
│       │   ├── rules.py           # 遊戲邏輯：合法放置、消除行列、模擬落子後盤面
│       │   └── board.py           # 棋盤的純函式操作 (放置、檢查、計算消除)
│       ├── ai/
│       │   ├── __init__.py
│       │   ├── search.py          # 貪婪搜尋：枚舉 3 方塊所有放法並挑最佳序列
│       │   └── heuristics.py      # 啟發式評估函數：對盤面打分
│       ├── control/
│       │   ├── __init__.py
│       │   ├── adb_client.py      # 封裝 ADB 指令 (screencap, input swipe/tap)
│       │   └── actuator.py        # 將 Move 轉換為螢幕拖曳座標並執行
│       └── utils/
│           ├── __init__.py
│           ├── logger.py          # 統一日誌輸出
│           └── visualize.py       # 除錯用：把辨識結果疊圖輸出
├── scripts/
│   ├── calibrate.py               # 互動式校準工具，產生 calibration.yaml
│   └── debug_vision.py            # 單獨測試影像辨識結果的小工具
└── tests/
    ├── test_board.py              # 棋盤放置/消除邏輯單元測試
    ├── test_rules.py              # 遊戲規則正確性測試
    ├── test_heuristics.py         # 評估函數行為測試
    ├── test_search.py             # 搜尋演算法 (以固定盤面驗證最佳解) 測試
    └── fixtures/                  # 測試用的固定盤面與截圖樣本
```

---

## 2. 核心模組介面設計（資料傳遞）

四大模組以**單向資料流**串接，由 `main.py` 的主迴圈協調。各模組之間只傳遞明確定義的資料結構（定義於 `core/state.py`），彼此不直接耦合。

### 共用資料結構（`core/state.py`）
- `Board`：`np.ndarray` shape `(8, 8)`，`0` 表示空格、`1` 表示已填滿。
- `Piece`：以小型布林矩陣（如 `np.ndarray`）表示方塊形狀；一回合有 `list[Piece]` 共 3 個。
- `GameState`：包含 `board: Board` 與 `pieces: list[Piece]`。
- `Move`：`(piece_index, row, col)`，表示「把第幾個方塊放在棋盤的哪個錨點」。
- `Plan`：`list[Move]`，本回合 3 個方塊的放置順序與位置。

### 資料流（一個回合的循環）

```text
┌─────────────┐   螢幕截圖 (ndarray)    ┌──────────────────┐
│  Control     │ ──────────────────────▶ │  Vision          │
│ adb_client   │   capture.grab()        │ board_reader     │
└─────────────┘                          │ piece_reader     │
       ▲                                 └────────┬─────────┘
       │ 執行拖曳指令                              │ GameState
       │ (螢幕座標)                                │ (Board + 3 Pieces)
       │                                          ▼
┌─────────────┐      Plan (list[Move])   ┌──────────────────┐
│  Control     │ ◀────────────────────── │  AI              │
│  actuator    │                         │ search +         │
└─────────────┘                         │ heuristics       │
                                         └────────┬─────────┘
                                                  │ 呼叫驗證/模擬
                                                  ▼
                                         ┌──────────────────┐
                                         │  Core (rules)    │
                                         │ 合法性 / 消除模擬 │
                                         └──────────────────┘
```

### 各介面契約（輸入 → 輸出）

1. **Vision（視覺辨識）**
   - `capture.grab() -> np.ndarray`：呼叫 `control.adb_client` 取得 RGB 截圖。
   - `board_reader.read(img) -> Board`：截圖 → 8x8 矩陣。
   - `piece_reader.read(img) -> list[Piece]`：截圖 → 3 個方塊形狀。
   - 輸出組合為 `GameState`，交給 AI。

2. **Core（環境邏輯，純函式、無 I/O）**
   - `board.place(board, piece, row, col) -> Board | None`：回傳放置後盤面，非法則 `None`。
   - `rules.clear_lines(board) -> (Board, cleared_count)`：消除滿行/滿列並回傳。
   - `rules.legal_moves(board, piece) -> list[(row, col)]`：列舉某方塊所有合法錨點。
   - 此模組同時被 AI（模擬未來盤面）與測試使用，不依賴任何外部模組。

3. **AI（決策）**
   - `search.best_plan(state: GameState) -> Plan`：枚舉 3 個方塊的放置排列組合，透過 `core.rules` 模擬每一步，並用 `heuristics.evaluate(board) -> float` 對結果盤面打分，回傳分數最高的 `Plan`。
   - `heuristics.evaluate(board) -> float`：評估指標範例 — 已消除行列數、空格連通性/碎片化、最大連續空間、貼邊/填角偏好等加權總和。
   - AI 只依賴 `core`，不碰 Vision/Control。

4. **Control（ADB 控制）**
   - `adb_client.screencap() -> np.ndarray`：被 Vision 使用。
   - `actuator.execute(plan: Plan)`：依 `calibration.yaml` 把每個 `Move` 換算為「從方塊起點拖到棋盤目標格」的螢幕座標，呼叫 `adb_client.swipe(...)` 執行。

> **解耦重點**：Vision 與 Control 是唯二碰觸外部世界（模擬器）的模組；Core 與 AI 為純運算，可在無模擬器環境下用 fixtures 完整測試。

---

## 3. 階段性開發任務（Milestones）

每個 Milestone 都可獨立測試、獨立交付，依序建構。

- [ ] **M1 — 專案骨架與資料結構**
  建立目錄結構、`requirements.txt`、`pyproject.toml`；實作 `core/state.py` 的 `Board`/`Piece`/`GameState`/`Move`/`Plan` 與 `logger.py`。
  *驗收*：可 `import block_blast`，資料結構有基本建構與序列化測試通過。

- [ ] **M2 — 核心遊戲邏輯（純函式）**
  實作 `core/board.py` 與 `core/rules.py`：放置、合法性檢查、行列消除、合法錨點列舉。
  *驗收*：`tests/test_board.py`、`tests/test_rules.py` 以固定盤面驗證放置與消除結果全部通過（完全不需模擬器）。

- [ ] **M3 — AI 決策引擎**
  實作 `ai/heuristics.py` 評估函數與 `ai/search.py` 貪婪搜尋（枚舉 3 方塊排列 × 合法位置）。
  *驗收*：`tests/test_heuristics.py`、`tests/test_search.py` 對已知盤面回傳預期最佳 `Plan`；可餵入手刻盤面觀察決策合理性。

- [ ] **M4 — ADB 控制層**
  實作 `control/adb_client.py`（screencap / swipe / tap 封裝）與 `scripts/calibrate.py` 校準工具，產生 `calibration.yaml`；`control/actuator.py` 將 `Move` 轉為螢幕拖曳。
  *驗收*：能對模擬器擷圖、能依手動指定的 `Plan` 在模擬器上正確拖放方塊。

- [ ] **M5 — 視覺辨識層**
  實作 `vision/capture.py`、`board_reader.py`、`piece_reader.py`，搭配 `utils/visualize.py` 除錯疊圖與 `scripts/debug_vision.py`。
  *驗收*：對多張真實截圖辨識出的 `Board` 與 3 個 `Piece` 與人工標註一致（建立辨識準確率基準）。

- [ ] **M6 — 主迴圈整合與端對端驗證**
  在 `main.py` 串接 Vision → AI → Control 的完整回合迴圈，加入 `config/settings.yaml` 控制頻率/除錯，處理「無合法步＝遊戲結束」等終止條件。
  *驗收*：在模擬器上連續自動遊玩數十回合不崩潰，並能在日誌中追蹤每回合的盤面辨識與決策。

---

## 風險與注意事項
- **辨識穩定性**：不同模擬器解析度/主題會影響辨識，故 M5 之前先以 `calibration.yaml` 抽離座標、樣板圖抽離至 `assets/`，降低硬編碼。
- **搜尋成本**：3 方塊全排列 × 各自合法位置仍屬小規模（8x8），貪婪枚舉可即時運算，無需剪枝；若日後加深前瞻再優化。
- **時序問題**：放置與動畫消除間需留緩衝，`actuator` 與主迴圈應有可設定的延遲。

## 驗收方式總覽
- **離線可測（M1–M3）**：`pytest tests/` 全綠，純邏輯不需模擬器。
- **線上需模擬器（M4–M6）**：以 `scripts/calibrate.py`、`scripts/debug_vision.py` 半自動驗證，最後以 `main.py` 跑端對端實機遊玩。
