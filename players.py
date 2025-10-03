# players.py
import random

from matplotlib.ticker import MaxNLocator

from game import Game, state_list2state_array, array2string, string2array, move_id2move_action, move_action2move_id
from mcts import MCTSPlayer
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

def evaluate_policy_against_checkpoints(board,
                                        model_dir="models",
                                        start=1000, end=6000, step=1000,
                                        n_games=100,
                                        csv_file="post_policy_evaluate.csv"):

    current_policy = PolicyValueNet(model_file='current_policy.pth')
    current_player = MCTSPlayer(current_policy.policy_value_fn,
                                c_puct=1, n_playout=500, is_selfplay=0)
    current_player.agent = f"Current-policy"

    # 收集對手 (舊的 checkpoint)
    opponents = {}
    for batch in range(start, end + 1, step):
        filename = f"current_policy_batch{batch}.pth"
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            old_policy = PolicyValueNet(model_file=path)
            old_player = MCTSPlayer(old_policy.policy_value_fn,
                                    c_puct=1, n_playout=400, is_selfplay=0)
            old_player.agent = f"Batch{batch}"
            opponents[f"Batch{batch}"] = old_player

        # 跑 battle_summary
        results = battle_summary(current_player, opponents, board, playouts=1000,
                                 n_games=n_games, save_csv=True, csv_file=csv_file)
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