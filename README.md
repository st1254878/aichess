# AI Chess - AlphaZero Dark Chess AI

本專案是一個基於 **AlphaZero** 算法實現的**中國象棋暗棋** AI。它結合了強化學習、蒙特卡洛樹搜索（MCTS）和深度神經網路，致力於打造一個強大的暗棋對弈智能體。

## 🌟 核心特性

- **雙框架支持**: 同時支持 **PyTorch** 與 **PaddlePaddle**，可根據環境自由切換。
- **深度殘差網路**: 採用帶有 **SE Block (Squeeze-and-Excitation)** 的改進型殘差網路 (ResNet)，增強特徵提取能力。
- **異步分布式訓練**: 支援通過 **Redis** 進行數據存儲與多進程並行數據收集，極大提升訓練效率。
- **GUI 對弈界面**: 基於 Pygame 實現的圖形化介面，支持人機對弈與 AI 性能實時觀測。
- **全方位算法實現**: 包含 MCTS、純 MCTS (Pure MCTS)、貪婪算法 (Greedy)、甚至支持與 ChatGPT API 對接。

---

## 📂 項目結構

| 文件名 | 說明 |
| :--- | :--- |
| `game.py` | 核心邏輯：實現暗棋規則、棋盤狀態編碼、合法動作生成等。 |
| `config.py` | 配置文件：訓練參數、模型路徑、框架選擇及 Redis 設定。 |
| `mcts.py` | AlphaZero MCTS：結合神經網路評估的蒙特卡洛樹搜索。 |
| `pytorch_net.py` | PyTorch 網路實現：核心 DarkChessNet 架構。 |
| `paddle_net.py` | PaddlePaddle 網路實現。 |
| `collect.py` | 數據收集：讓 AI 進行自我對弈，產生訓練樣本。 |
| `train.py` | 模型訓練：使用自我對弈數據進行網路參數更新。 |
| `UIplay.py` | GUI 介面：使用 Pygame 進行交互式人機/AI對弈。 |
| `players.py` | 玩家代理庫：Greedy、Random、MCTS、Minimax 及 GPT 玩家。 |
| `server.py` | 網頁/分布式服務端：提供 Flask API 支援通訊、Session 管理與 Ngrok 穿透部署。 |

---

## 🔐 安全與環境配置

專案使用 `.env` 進行環境變數管理。

### 1. 手動設定環境變數
在專案根目錄手動建立 `.env` 檔案，並根據需求填入以下內容：

```env
# Flask 與服務器設定
PORT=5000
SESSION_TTL=3600
MAX_SESSIONS=50

# AI 模擬設定
MCTS_PLAYOUTS=300

NGROK_AUTH_TOKEN=你的_NGROK_TOKEN
```

---

### 1. 環境配置

建議使用 Python 3.8+ 及 NVIDIA GPU。

```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> 為了達到最佳性能，請務必安裝對應 CUDA 版本的 PyTorch 或 PaddlePaddle。MCTS 的模擬需要高效的神經網路推理速度。

### 2. 配置文件

在 `config.py` 中設置您使用的框架：

```python
CONFIG = {
    'use_frame': 'pytorch',  # 或 'paddle'
    'use_redis': False,       
    # ... 其他參數
}
```

### 3. 開始對弈

運行 GUI 界面與 AI 進行對話：

```bash
python UIplay.py
```

### 4. 自我強化訓練

1.  **開啟數據收集**: 在終端運行 `collect.py`（可開啟多個進程）。
    ```bash
    python collect.py
    ```
2.  **開啟更新模型**: 同時運行 `train.py` 監聽數據並定時訓練。
    ```bash
    python train.py
    ```

---

## 🧠 技術細節

- **狀態編碼**: 棋盤狀態被編碼為 `(62, 4, 8)` 的張量，包含歷史步數、棋子位置、未翻開棋子期望等信息。
- **動作空間**: 定義了 `384` 種可能的動作。
- **網路架構**: 輸入層 -> SE Block -> 多層帶 SE 的殘差快 -> 策略頭 (Policy Head) & 價值頭 (Value Head)。



## 🙏 致謝與參考

特別感謝以下開源項目與資料的啟發：

https://github.com/tensorfly-gpu/aichess

---
*後續將持續完善性能，目標是打造一個接近甚至超越人類頂尖水平的暗棋 AI！*
