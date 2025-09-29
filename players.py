# players.py
import random

from matplotlib.ticker import MaxNLocator

from game import Game, state_list2state_array, array2string, string2array, move_id2move_action, move_action2move_id

import csv
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from openai import OpenAI

from pytorch_net import PolicyValueNet

# 設定中文字體（以 Windows 為例）
rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
rcParams['axes.unicode_minus'] = False


class ChatGPTPlayer:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key="")  # 放 API Key

        self.model = model
        self.agent = 'AI'

    def set_player_ind(self, p):
        self.player = p

    def board_to_state(self, _state_array):
        # _state_array: [10, 9, 7], HWC
        all_board = []

        for i in range(4):
            board_line = []
            for j in range(8):
                board_line.append(array2string(_state_array[i][j]))
            all_board.append(board_line)
        return all_board

    def get_action(self, board):
        # 轉換棋盤
        state_text = self.board_to_state(state_list2state_array(board.state_deque[-1]))

        # 把所有合法行動列出來
        eat_move_list, fallback_movelist = board.greedys()
        all_moves = eat_move_list + fallback_movelist
        all_moves_relabel = [move_id2move_action[m] for m in all_moves]
        # 生成 prompt
        # print(all_moves_relabel)
        prompt = f"""
你正在玩中國暗棋 (4x8)。
現在棋盤狀態如下：
{state_text}

可行的動作有：
{all_moves_relabel} 
格式是4個數字 ABCD AB為起始位置 CD為結束位置 若A=C B=D則代表翻棋

請從中選擇一個最佳動作，直接輸出該動作 (不要多餘的解釋)。
"""

        try:
            # 呼叫 ChatGPT
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            reply = response.choices[0].message.content.strip()
            reply_str = "".join(str(x) for x in reply if x.isdigit())  # 過濾非數字

            # 嘗試轉換
            if reply_str in move_action2move_id:
                reply_index = move_action2move_id[reply_str]
                # 確保動作合法
                if reply_index in all_moves:
                    return reply_index
                else:
                    print(f"⚠️ ChatGPT 回傳非法動作 {reply_str}（不在合法動作列表），改用隨機動作")
            else:
                print(f"⚠️ ChatGPT 回傳無效動作字串 {reply}，改用隨機動作")

        except Exception as e:
            print(f"⚠️ ChatGPT API 錯誤: {e}，改用隨機動作")

            # fallback
        return random.choice(all_moves) if all_moves else None

class RandomPlayer:
    def __init__(self):
        self.agent = 'AI'
        self.type = "random"

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        eat_move_list, fallback_move_list = board.greedys()
        all_move = eat_move_list + fallback_move_list
        if not all_move:
            return None
        return random.choice(all_move)


class GreedyPlayer:
    def __init__(self):
        self.agent = 'AI'
        self.type = "greedy"

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        eat_move_list, fallback_move_list = board.greedys()
        if eat_move_list:
            return random.choice(eat_move_list)
        elif fallback_move_list:
            return random.choice(fallback_move_list)
        else:
            return None


class Human:
    def __init__(self):
        self.agent = 'Human'

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        # UIplay 會用，這裡先放佔位
        return None

def battle(player1, player2, board, playouts, n_games=100, is_shown=True,
           plot_interval=10, save_csv=True, csv_dir="battle_results"):
    game = Game(board)
    results = {"player1_win": 0, "player2_win": 0, "tie": 0}

    # 累積吃子 / 非吃子 統計
    total_eat1 = total_non1 = 0
    total_eat2 = total_non2 = 0

    # 繪圖資料
    p1_rates, p2_rates, tie_rates, rounds = [], [], [], []
    p1_eat_ratios, p2_eat_ratios = [], []

    # --- CSV 初始化 ---
    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)
        fname = os.path.join(csv_dir, f"battle_{player1.agent}_vs_{player2.agent}.csv")
        f = open(fname, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["Round", "P1_rate", "P2_rate", "Tie_rate", "P1_eat_ratio", "P2_eat_ratio"])
    else:
        writer = None
        f = None

    print("battle start!")
    for i in range(n_games):
        board.init_board(1)
        players = {1: player1, 2: player2}

        if i % 2 == 0:
            player1.set_player_ind(1); player2.set_player_ind(2)
        else:
            player1.set_player_ind(2); player2.set_player_ind(1)

        # 單局吃子/非吃子
        p1_eat_game = p1_non_game = 0
        p2_eat_game = p2_non_game = 0

        while True:
            if is_shown:
                game.graphic(board)

            current_player_id = board.current_player_id
            player_in_turn = players[current_player_id]

            eat_moves, fallback_moves = board.greedys()
            move = player_in_turn.get_action(board)

            if move is None:
                break

            if move in eat_moves:
                if player_in_turn == player1:
                    p1_eat_game += 1
                else:
                    p2_eat_game += 1
            else:
                if player_in_turn == player1:
                    p1_non_game += 1
                else:
                    p2_non_game += 1

            board.do_move(move)
            end, winner = board.game_end()
            if end:
                if winner == -1:
                    results["tie"] += 1
                elif players[winner] == player1:
                    results["player1_win"] += 1
                else:
                    results["player2_win"] += 1
                break

        # 累加
        total_eat1 += p1_eat_game
        total_non1 += p1_non_game
        total_eat2 += p2_eat_game
        total_non2 += p2_non_game

        # --- 每場結束就馬上算勝率、吃子比例 ---
        total_played = results["player1_win"] + results["player2_win"] + results["tie"]
        p1_rate = results["player1_win"] / total_played if total_played > 0 else 0
        p2_rate = results["player2_win"] / total_played if total_played > 0 else 0
        tie_rate = results["tie"] / total_played if total_played > 0 else 0

        p1_total_moves = total_eat1 + total_non1
        p2_total_moves = total_eat2 + total_non2
        p1_eat_ratio = total_eat1 / p1_total_moves if p1_total_moves > 0 else 0.0
        p2_eat_ratio = total_eat2 / p2_total_moves if p2_total_moves > 0 else 0.0

        # 保存到列表（用於畫圖）
        rounds.append(i + 1)
        p1_rates.append(p1_rate)
        p2_rates.append(p2_rate)
        tie_rates.append(tie_rate)
        p1_eat_ratios.append(p1_eat_ratio)
        p2_eat_ratios.append(p2_eat_ratio)

        # --- 每場結束就寫進 CSV ---
        if writer:
            writer.writerow([i + 1, p1_rate, p2_rate, tie_rate, p1_eat_ratio, p2_eat_ratio])
            f.flush()  # 立即寫入硬碟，避免中途掛掉資料消失

    # --- 關閉 CSV 檔案 ---
    if f:
        f.close()
        print(f"📊 對戰數據已即時存到 {fname}")
    # --- 畫圖 ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rounds, p1_rates, label=f"Player1 勝率 ({player1.agent})", marker="o")
    ax1.plot(rounds, p2_rates, label=f"Player2 勝率 ({player2.agent})", marker="s")
    ax1.plot(rounds, tie_rates, label="平局率", linestyle="--")
    ax1.set_xlabel("對戰場數")
    ax1.set_ylabel(f"累積勝率 ({playouts} playout)")
    ax1.set_title(f"{player1.agent} vs {player2.agent}")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    # 下面：吃子比例長條圖
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    width = 0.4  # 長條寬度

    ax2.bar([r - width / 2 for r in rounds], p1_eat_ratios, width=width, label=f"Player1 ({player1.agent})")
    ax2.bar([r + width / 2 for r in rounds], p2_eat_ratios, width=width, label=f"Player2 ({player2.agent})")

    ax2.set_xlabel("對戰場數")
    ax2.set_ylabel("累積吃子比例")
    ax2.set_title("吃子比例比較")
    ax2.set_xticks(rounds[::max(1, len(rounds) // 10)])  # 只顯示部分 xtick 避免擠爆
    ax2.set_xticklabels(rounds[::max(1, len(rounds) // 10)])
    ax2.set_ylim(0, 1.0)  # 吃子比例必在 [0,1]
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # --- 存 CSV ---
    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)
        fname = os.path.join(csv_dir, f"battle_{player1.agent}_vs_{player2.agent}.csv")
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "P1_rate", "P2_rate", "Tie_rate", "P1_eat_ratio", "P2_eat_ratio"])
            for r, p1r, p2r, tr, p1e, p2e in zip(rounds, p1_rates, p2_rates, tie_rates, p1_eat_ratios, p2_eat_ratios):
                writer.writerow([r, p1r, p2r, tr, p1e, p2e])
        print(f"📊 對戰數據已儲存到 {fname}")

    return results

def battle_summary(player1, opponents, board, playouts, n_games=100, save_csv=True, csv_file="battle_summary.csv"):
    game = Game(board)
    results = {}

    # --- CSV 初始化 ---
    if save_csv:
        f = open(csv_file, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["Opponent", "Wins", "Losses", "Draws"])
    else:
        writer = None
        f = None

    for opp_name, player2 in opponents.items():
        stats = {"win": 0, "loss": 0, "draw": 0}
        print(f"⚔️ {player1.agent} vs {opp_name} 開始對戰，共 {n_games} 場...")

        for i in range(n_games):
            board.init_board(1)
            players = {1: player1, 2: player2}

            if i % 2 == 0:
                player1.set_player_ind(1); player2.set_player_ind(2)
            else:
                player1.set_player_ind(2); player2.set_player_ind(1)

            while True:
                move = players[board.current_player_id].get_action(board)
                if move is None:
                    break
                board.do_move(move)
                end, winner = board.game_end()
                if end:
                    if winner == -1:
                        stats["draw"] += 1
                    elif players[winner] == player1:
                        stats["win"] += 1
                    else:
                        stats["loss"] += 1
                    break

        results[opp_name] = stats
        print(f"✅ {player1.agent} vs {opp_name} 完成: {stats}")

        # --- 存到 CSV ---
        if writer:
            writer.writerow([opp_name, stats["win"], stats["loss"], stats["draw"]])
            f.flush()

    if f:
        f.close()
        print(f"📊 對戰結果已存到 {csv_file}")
    return results


def plot_battle_results_from_csv(csv_file="battle_summary.csv"):
    opponents = []
    wins = []

    # --- 讀取 CSV ---
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opponents.append(row["Opponent"])
            wins.append(int(row["Wins"]))

    # --- 畫圖 (黑白 + 斜線) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(opponents, wins, color="white", edgecolor="black", hatch="//")

    ax.set_xlabel("對手")
    ax.set_ylabel("勝利場數")
    ax.set_title("對戰結果 (只顯示勝場數)")
    ax.set_ylim(0, max(wins) + 5)

    # 在柱狀圖上加數字
    for bar, v in zip(bars, wins):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()