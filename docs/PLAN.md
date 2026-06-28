# Block Blast 自動遊玩 AI — 架構規劃書

## Context（背景）

目標：打造一個能自動遊玩 Block Blast（8x8 棋盤、每回合給 3 個方塊、填滿整行或整列即消除）的 AI。AI 透過 ADB 連接電腦端 Android 模擬器，用 OpenCV 辨識畫面狀態，以貪婪演算法配合啟發式評估函數決定落子，再透過 ADB 模擬點擊與拖曳完成操作。形成「截圖 → 辨識 → 決策 → 操作 → 再截圖」的閉環。

不使用強化學習；策略為 Greedy + Heuristic Search，可在無訓練資料下即時運作、易於除錯與調參。

---

## 1. 目錄與檔案結構

```
BlockBlastCheat/
├── main.py                     # 程式進入點，組裝各模組並驅動主迴圈（截圖→辨識→決策→操作）
├── config.yaml                 # 全域設定：ADB 位址、棋盤螢幕座標、啟發式權重、迴圈間隔
├── requirements.txt            # 套件相依清單（opencv-python, numpy, pyyaml 等）
├── README.md                   # 專案說明與執行方式
│
├── src/                        # 原始碼根目錄
│   ├── __init__.py
│   │
│   ├── vision/                 # 視覺辨識模組：像素 → 矩陣
│   │   ├── __init__.py
│   │   ├── screen_capture.py   # 透過 ADB 截圖並回傳 OpenCV 影像（np.ndarray）
│   │   ├── board_reader.py     # 辨識 8x8 棋盤狀態，輸出 8x8 二值矩陣
│   │   └── piece_reader.py     # 辨識下方 3 個待放方塊的形狀，輸出方塊矩陣清單
│   │
│   ├── engine/                 # 環境邏輯模組：純規則運算，無 IO
│   │   ├── __init__.py
│   │   ├── board.py            # Board 類別：放置合法性檢查、模擬放置、消除行列、計分
│   │   └── piece.py            # Piece 類別：方塊形狀表示與佔格座標
│   │
│   ├── ai/                     # AI 決策模組：選出最佳放置序列
│   │   ├── __init__.py
│   │   ├── solver.py           # 貪婪搜尋：列舉 3 方塊所有放置排列，回傳最佳動作序列
│   │   └── heuristics.py       # 啟發式評估函數：對盤面打分（空洞、平整度、消除數等）
│   │
│   ├── control/                # ADB 控制模組：動作 → 實機操作
│   │   ├── __init__.py
│   │   ├── adb_client.py       # 封裝 ADB 指令（connect/screencap/input swipe/tap）
│   │   └── action_executor.py  # 將 AI 動作（方塊→格座標）轉成螢幕像素拖曳並執行
│   │
│   └── utils/                  # 共用工具
│       ├── __init__.py
│       ├── config_loader.py    # 讀取並驗證 config.yaml
│       ├── geometry.py         # 棋盤格座標 ↔ 螢幕像素座標 互轉
│       └── logger.py           # 統一日誌
│
├── calibration/                # 校正工具與資料
│   ├── calibrate.py            # 半自動工具：點選螢幕標定棋盤四角與方塊區位置
│   └── templates/              # 顏色/形狀樣板圖（供辨識比對）
│
└── tests/                      # 單元測試（每模組可獨立測）
    ├── test_board.py
    ├── test_heuristics.py
    ├── test_solver.py
    └── fixtures/               # 測試用截圖與預期矩陣
```

---

## 2. 核心模組介面設計（資料流）

四模組以**標準化資料結構**解耦，互不依賴實作細節。核心資料型別：

| 型別 | 定義 | 說明 |
|------|------|------|
| `BoardState` | `np.ndarray` shape `(8,8)`，dtype int | `1`=已填，`0`=空 |
| `Piece` | `np.ndarray` 小矩陣 + 形狀 id | 例如 L 形為 `(3,2)` 二值矩陣 |
| `Placement` | `(piece_index, row, col)` | 第 N 個方塊放在棋盤 (row,col) 為左上角 |
| `Action` | `List[Placement]` | 本回合 3 個方塊的放置序列（含順序） |

**閉環資料流**（main.py 每回合驅動一次）：

```
┌─────────────┐  np.ndarray(BGR)   ┌──────────────┐
│ control      │ ───────screenshot──▶│ vision        │
│ (adb_client) │                     │ board_reader  │
└─────────────┘                     │ piece_reader  │
       ▲                            └──────┬───────┘
       │                                   │ BoardState + List[Piece]
       │ Action                            ▼
┌──────┴────────┐  Action(List[Placement]) ┌──────────┐
│ control        │◀────────────────────────│ ai        │
│ action_executor│                          │ solver    │
└────────────────┘                          │ heuristics│
                                            └─────┬────┘
                                                  │ 用 engine 模擬
                                                  ▼
                                            ┌──────────┐
                                            │ engine    │
                                            │ Board     │（純函式，被 ai 呼叫）
                                            └──────────┘
```

**逐模組介面契約：**

- **vision → ai**
  - `board_reader.read(img) -> BoardState`
  - `piece_reader.read(img) -> List[Piece]`（長度 3，若方塊已用完則該位為 None）
  - 職責邊界：vision 只做「像素→矩陣」，不懂遊戲規則。

- **ai ↔ engine**
  - `solver.solve(board: BoardState, pieces: List[Piece]) -> Action`
  - solver 內部呼叫 `engine.Board.can_place()`, `place()`, `clear_lines()` 做模擬，呼叫 `heuristics.evaluate(board) -> float` 評分。
  - engine 為**純規則層**：輸入矩陣、輸出矩陣與分數，無任何 IO/截圖/ADB。可單獨測。

- **ai → control**
  - `action_executor.execute(action: Action)`：把每個 `Placement` 透過 `geometry` 換算成「方塊托盤像素座標 → 棋盤目標格像素座標」的拖曳，呼叫 `adb_client.swipe()`。

- **utils/geometry** 為座標轉換唯一真實來源：`grid_to_pixel(row,col)` / `tray_to_pixel(piece_index)`，所有像素映射集中於此，校正值來自 `config.yaml`。

**關鍵設計原則：** engine 與 ai 完全不碰螢幕/ADB，可用 `tests/fixtures` 的矩陣離線測；vision 與 control 為唯一接觸真實裝置的層。

---

## 3. 階段性開發任務（Milestones）

- [ ] **M1 — 環境邏輯核心（engine，純離線，最先做）**
  - 實作 `Board`（放置合法性、模擬放置、消除行列、計分）與 `Piece`（形狀庫）。
  - 完成 `tests/test_board.py`，用手寫矩陣驗證放置/消除/邊界。
  - 驗收：不需模擬器即可全綠通過。

- [ ] **M2 — AI 決策（ai，純離線）**
  - 實作 `heuristics.evaluate`（空洞數、盤面密度/平整度、本回合消除行列數、剩餘空間等加權）。
  - 實作 `solver.solve`：列舉 3 方塊的 6 種排列 × 各合法落點，用 engine 模擬，取總分最高序列。
  - 驗收：給定 `fixtures` 盤面，輸出合理且合法的 `Action`；`test_solver.py` 通過。

- [ ] **M3 — ADB 連線與截圖（control + vision 截圖）**
  - `adb_client`：connect、screencap、tap、swipe 封裝。
  - `screen_capture`：取得模擬器畫面為 OpenCV 影像並存檔。
  - 驗收：能穩定截到模擬器當前畫面 PNG。

- [ ] **M4 — 視覺辨識（vision，含校正）**
  - `calibrate.py` 標定棋盤四角與方塊托盤區，寫入 `config.yaml`。
  - `board_reader` / `piece_reader`：依格切割 + 顏色/亮度門檻 → 矩陣。
  - 驗收：對多張 `fixtures` 截圖辨識出的矩陣與人工標註一致。

- [ ] **M5 — 動作執行與座標映射（control 執行）**
  - `geometry` 完成格↔像素互轉；`action_executor` 把 `Placement` 轉為 ADB 拖曳。
  - 驗收：手動丟一個固定 `Action`，模擬器中方塊被正確拖到目標格。

- [ ] **M6 — 主迴圈整合與穩定化（main.py）**
  - 串起完整閉環，加入回合偵測、辨識失敗重試、遊戲結束判定、日誌。
  - 調整 `heuristics` 權重做實戰調參。
  - 驗收：模擬器中連續自動遊玩多回合不卡死，分數穩定成長。

---

## 驗證方式（Verification）

- **離線層（M1–M2）**：`pytest tests/` 全綠，不需模擬器。核心邏輯與決策可獨立回歸測試。
- **裝置層（M3–M5）**：各模組寫獨立 smoke 腳本——截一張圖、辨識一張圖、執行一個固定動作——肉眼或對照 fixture 驗證。
- **整合（M6）**：`python main.py` 連模擬器實跑，觀察自動遊玩是否合法、是否持續消除、是否穩定不崩。
- **回歸保護**：每次辨識/拖曳出錯時把當下截圖存入 `fixtures`，補成測試案例，防止再犯。
