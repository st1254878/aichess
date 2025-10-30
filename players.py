# players.py
import copy
import random

from matplotlib.ticker import MaxNLocator

from game import Game, state_list2state_array, array2string, string2array, move_id2move_action, move_action2move_id, Board
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


class MinimaxDarkChessPlayer:
    """暗棋 Alpha–Beta + 向上傳遞 + 寧靜搜尋 AI 玩家"""

    def __init__(self, search_depth=4):
        self.game = None
        self.depth = search_depth
        self.player = None
        self.agent = "AI"

    def set_player_ind(self, p):
        """設定玩家編號：1=紅, 2=黑"""
        self.player = p

    # ====================== 改良版搜尋 ======================
    def quiescence_search(self, board, alpha, beta, side_player_id, maxDepth, depth):
        """寧靜搜尋：只展開吃子直到穩定。"""
        eat_moves, _ = board.greedys()
        if not eat_moves:
            return 0

        best = -float("inf")
        for move in eat_moves:
            backup = self._backup_board(board)
            reward = board.do_move(move)  # reward 為吃子分數（正代表吃對方）
            # value = 該吃子分數 + 同層深度 bonus
            bonus = (maxDepth - depth)
            value = reward + bonus
            # 遞迴吃到底
            value -= self.quiescence_search(board, -beta, -alpha, side_player_id, maxDepth, depth + 1)
            self._restore_board(board, backup)

            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def search_upward(self, board, depth, alpha, beta, side_player_id, maxDepth):
        """向上傳遞 + 同分步處理 + 單向搜尋 (negamax style)"""
        # --- 終止條件 ---
        if depth == 0:
            eat_moves, _ = board.greedys()
            if eat_moves:
                return self.quiescence_search(board, alpha, beta, side_player_id, maxDepth, 0)
            return 0

        eat_moves, fallback_moves = board.greedys()
        legal_moves = eat_moves + fallback_moves
        if not legal_moves:
            return 0

        value_accum = -float("inf")
        all_zero_scores = True

        for move in legal_moves:
            backup = self._backup_board(board)
            reward = board.do_move(move)  # reward 為吃子加權值（正：吃對手，負：被吃）
            # 初始分值：若有吃子則加上同分步bonus
            value_here = 0
            if reward != 0:
                bonus = (maxDepth - depth)
                value_here += reward + bonus
                child_depth = depth if depth == maxDepth - 1 else depth - 1
                value_here -= self.search_upward(board, child_depth, -beta, -alpha, side_player_id, maxDepth)
            else:
                value_here -= self.search_upward(board, depth - 1, -beta, -alpha, side_player_id, maxDepth)

            self._restore_board(board, backup)

            if value_here > value_accum:
                value_accum = value_here
            if value_accum > alpha:
                alpha = value_accum
            if alpha >= beta:
                break

            if value_here != 0:
                all_zero_scores = False

        # --- 單向搜尋（當所有候選皆為 0） ---
        if all_zero_scores and depth > 1:
            one_way_best = -float("inf")
            eat_moves2, fallback_moves2 = board.greedys()
            for move in eat_moves2 + fallback_moves2:
                backup = self._backup_board(board)
                reward = board.do_move(move)
                val = reward - self.search_upward(board, depth - 1, -beta, -alpha, side_player_id, maxDepth)
                self._restore_board(board, backup)
                if val > one_way_best:
                    one_way_best = val
                if one_way_best > alpha:
                    alpha = one_way_best
                if alpha >= beta:
                    break
            if one_way_best > value_accum:
                value_accum = one_way_best

        return value_accum

    # ====================== 動作選擇 ======================
    def get_action(self, board):
        """取得下一步行動"""
        self.game = Game(board)
        eat_moves, fallback_moves = board.greedys()
        legal_moves = eat_moves + fallback_moves
        if not legal_moves:
            return None

        # 有吃子 → 用新搜尋法
        if eat_moves:
            best_value = -float("inf")
            best_move = None
            alpha, beta = -float("inf"), float("inf")
            side_player_id = board.current_player_id
            maxDepth = self.depth

            for move in eat_moves:
                backup = self._backup_board(board)
                reward = board.do_move(move)
                value_here = reward
                if reward != 0:
                    bonus = (maxDepth - 1)
                    value_here += bonus
                    value_here -= self.search_upward(board, maxDepth - 1, -beta, -alpha, side_player_id, maxDepth)
                else:
                    value_here -= self.search_upward(board, maxDepth - 1, -beta, -alpha, side_player_id, maxDepth)
                self._restore_board(board, backup)

                if value_here > best_value:
                    best_value = value_here
                    best_move = move

            return best_move
        else:
            # 沒吃子 → 翻子策略
            return self._reveal_strategy(board)

        # ================== 翻子策略 ==================

    def _reveal_strategy(self, board):
        """模擬翻子期望值，挑出最有利的翻子位置"""
        dark_positions = board.get_dark_positions()
        if not dark_positions:
            return random.choice(board.availables)  # 沒得翻就隨便走

        best_pos = None
        best_value = -float('inf')

        for pos in dark_positions:
            expected_value = self._simulate_reveal(board, pos)
            if expected_value > best_value:
                best_value = expected_value
                best_pos = pos

        r, c = best_pos
        action = f"{r}{c}{r}{c}"
        return move_action2move_id[action]

    def _simulate_reveal(self, board, pos):
        """模擬翻出不同棋子後的期望值（結合新搜尋法）"""
        re_pieces = board.remain_pieces
        if not re_pieces:
            return 0

        # 統計棋種機率
        counts = {}
        for p in re_pieces:
            counts[p] = counts.get(p, 0) + 1
        total = len(re_pieces)
        possible_pieces = {p: c / total for p, c in counts.items()}

        total_value = 0
        for piece, prob in possible_pieces.items():
            backup = self._backup_board(board)
            board.force_reveal(pos, piece)

            # 🔄 改成呼叫新搜尋器 search_upward
            score = self.search_upward(
                board, depth=2,
                alpha=-float('inf'),
                beta=float('inf'),
                side_player_id=board.current_player_id,
                maxDepth=2
            )

            total_value += prob * score
            self._restore_board(board, backup)

        return total_value

    # ====================== 評估與備份 ======================
    def evaluate(self, board, winner=None):
        """盤面評估"""
        if winner == self.player:
            return 9999
        elif winner != -1 and winner != self.player:
            return -9999

        red_strength = board.calc_side_strength('红')
        black_strength = board.calc_side_strength('黑')
        return red_strength - black_strength if self.player == 1 else black_strength - red_strength

    def _backup_board(self, board):
        """備份棋盤狀態"""
        return {
            "state_deque": copy.deepcopy(board.state_deque),
            "remain_pieces": copy.deepcopy(board.remain_pieces),
            "current_player_color": board.current_player_color,
            "current_player_id": board.current_player_id,
            "last_move": board.last_move,
            "winner": board.winner,
            "kill_action": board.kill_action,
            "first_move": board.first_move,
            "action_count": board.action_count,
        }

    def _restore_board(self, board, backup):
        """還原棋盤狀態"""
        board.state_deque = backup["state_deque"]
        board.remain_pieces = backup["remain_pieces"]
        board.current_player_color = backup["current_player_color"]
        board.current_player_id = backup["current_player_id"]
        board.last_move = backup["last_move"]
        board.winner = backup["winner"]
        board.kill_action = backup["kill_action"]
        board.first_move = backup["first_move"]
        board.action_count = backup["action_count"]


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
                                c_puct=1, n_playout=300, is_selfplay=0)
    current_player.agent = f"Current-policy"

    # 🔸 先建立 CSV（只建立一次）
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Opponent", "Wins", "Losses", "Draws"])

    # 收集對手 (舊的 checkpoint)

    for batch in range(start, end + 1, step):
        opponents = {}
        filename = f"current_policy_batch{batch}.pth"
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            old_policy = PolicyValueNet(model_file=path)
            old_player = MCTSPlayer(old_policy.policy_value_fn,
                                    c_puct=1, n_playout=200, is_selfplay=0)
            old_player.agent = f"Batch{batch}"
            opponents[f"Batch{batch}"] = old_player

        # 跑 battle_summary
        results = battle_summary(current_player, opponents, board, playouts=1000,
                                 n_games=n_games, save_csv=True, csv_file=csv_file, append=True)
    return

def battle_summary(player1, opponents, board, playouts, n_games=100,
                   save_csv=True, csv_file="battle_summary.csv"):
    """
    對每個對手進行對戰並記錄結果。
    若 CSV 已存在，則只會追加新對手的結果（不覆蓋舊資料）。
    """
    game = Game(board)
    results = {}

    existing_opponents = set()
    writer = None
    f = None

    # --- 檢查是否已有結果檔 ---
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f_read:
            reader = csv.reader(f_read)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 1:
                    existing_opponents.add(row[0])

    # --- 開啟 CSV 檔案 ---
    if save_csv:
        file_exists = os.path.exists(csv_file)
        f = open(csv_file, "a", newline="", encoding="utf-8")
        writer = csv.writer(f)
        # 如果是新檔案就寫入標頭
        if not file_exists:
            writer.writerow(["Opponent", "Wins", "Losses", "Draws"])

    # --- 逐一對戰 ---
    for opp_name, player2 in opponents.items():
        if opp_name in existing_opponents:
            print(f"⚠️ {opp_name} 已存在於 {csv_file}，跳過。")
            continue

        stats = {"win": 0, "loss": 0, "draw": 0}
        print(f"⚔️ {player1.agent} vs {opp_name} 開始對戰，共 {n_games} 場...")

        for i in range(n_games):
            board.init_board(1)
            players = {1: player1, 2: player2}

            # 輪流先手
            if i % 2 == 0:
                player1.set_player_ind(1)
                player2.set_player_ind(2)
            else:
                player1.set_player_ind(2)
                player2.set_player_ind(1)

            while True:
                move = players[board.current_player_id].get_action(board)
                if move is None:
                    break
                board.do_move(move)
                end, winner = board.game_end()
                if end:
                    print(f"第{i}場結束")
                    if winner == -1:
                        stats["draw"] += 1
                    elif players[winner] == player1:
                        stats["win"] += 1
                    else:
                        stats["loss"] += 1
                    break

        results[opp_name] = stats
        print(f"✅ {player1.agent} vs {opp_name} 完成: {stats}")

        # --- 寫入結果 ---
        if writer:
            writer.writerow([opp_name, stats["win"], stats["loss"], stats["draw"]])
            f.flush()

    if f:
        f.close()
        print(f"📊 對戰結果已更新至 {csv_file}")

    return results
def battle_capture_summary(player1, opponents, board, n_games=100, save_csv=True, csv_file="battle_capture_summary.csv"):
    """
    對戰統計（含雙方吃棋比例）
    記錄每個對手的平均吃棋比例 = 吃棋數 / 總步數
    分別統計 player1 與對手雙方。
    """
    game = Game(board)
    results = {}

    # --- 初始化 CSV ---
    if save_csv:
        f = open(csv_file, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow([
            "Opponent",
            "Wins", "Losses", "Draws",
            "Player1_CaptureRate", "Opponent_CaptureRate"
        ])
    else:
        writer = None
        f = None

    # --- 主回圈 ---
    for opp_name, player2 in opponents.items():
        stats = {"win": 0, "loss": 0, "draw": 0}
        p1_capture_rates = []  # player1 平均吃棋比例
        p2_capture_rates = []  # 對手 平均吃棋比例

        print(f"⚔️ {player1.agent} vs {opp_name} 開始對戰，共 {n_games} 場...")

        for i in range(n_games):
            board.init_board(1)
            players = {1: player1, 2: player2}

            # 每場初始化
            total_moves = {1: 0, 2: 0}
            total_captures = {1: 0, 2: 0}

            # 輪流先手
            if i % 2 == 0:
                player1.set_player_ind(1); player2.set_player_ind(2)
            else:
                player1.set_player_ind(2); player2.set_player_ind(1)

            # --- 對戰 ---
            while True:
                cur_id = board.current_player_id
                move = players[cur_id].get_action(board)
                if move is None:
                    break

                y1, x1, y2, x2 = map(int, move_id2move_action[move])
                start = board.state_deque[-1][y1][x1]
                target = board.state_deque[-1][y2][x2]

                # 判定是否吃棋
                if target not in ('一一', '暗棋') and board.current_player_color not in target:
                    total_captures[cur_id] += 1

                total_moves[cur_id] += 1
                board.do_move(move)

                end, winner = board.game_end()
                if end:
                    print(f"第{i}場結束")
                    if winner == -1:
                        stats["draw"] += 1
                    elif players[winner] == player1:
                        stats["win"] += 1
                    else:
                        stats["loss"] += 1
                    break

            # --- 計算本場吃棋比例 ---
            p1_id = player1.player
            p2_id = player2.player

            p1_rate = (total_captures[p1_id] / total_moves[p1_id]) if total_moves[p1_id] > 0 else 0
            p2_rate = (total_captures[p2_id] / total_moves[p2_id]) if total_moves[p2_id] > 0 else 0

            p1_capture_rates.append(p1_rate)
            p2_capture_rates.append(p2_rate)

        # --- 場均 ---
        avg_p1_rate = sum(p1_capture_rates) / len(p1_capture_rates)
        avg_p2_rate = sum(p2_capture_rates) / len(p2_capture_rates)

        results[opp_name] = {
            **stats,
            "p1_capture_rate": avg_p1_rate,
            "p2_capture_rate": avg_p2_rate
        }

        print(f"✅ {player1.agent} vs {opp_name} 完成: {stats}")
        print(f"  Player1 平均吃棋比例 = {avg_p1_rate:.3f}")
        print(f"  {opp_name} 平均吃棋比例 = {avg_p2_rate:.3f}")

        # --- 寫入 CSV ---
        if writer:
            writer.writerow([
                opp_name,
                stats["win"], stats["loss"], stats["draw"],
                f"{avg_p1_rate:.3f}", f"{avg_p2_rate:.3f}"
            ])
            f.flush()

    # --- 結尾 ---
    if f:
        f.close()
        print(f"📊 對戰結果（含雙方吃棋比例）已存到 {csv_file}")

    return results

def plot_battle_results_from_csv(csv_file="battle_summary.csv"):
    import csv
    import matplotlib.pyplot as plt

    opponents = []
    wins = []

    # --- 讀取 CSV ---
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opponents.append(row["Opponent"])
            wins.append(int(row["Wins"]))

    # --- 根據勝場數排序 ---
    sorted_data = sorted(zip(opponents, wins), key=lambda x: x[1], reverse=True)
    opponents, wins = zip(*sorted_data)

    # --- 定義不同的填充樣式 (hatch patterns) ---
    hatch_patterns = ["//", "\\\\", "xx", "oo", "--", "++", "..", "**"]
    hatch_patterns = (hatch_patterns * ((len(opponents) // len(hatch_patterns)) + 1))[:len(opponents)]

    # --- 畫圖 ---
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = []
    for i, (opponent, win) in enumerate(zip(opponents, wins)):
        bar = ax.bar(opponent, win, color="white", edgecolor="black", hatch=hatch_patterns[i])
        bars.append(bar)

    ax.set_xlabel("對手策略", fontsize=12)
    ax.set_ylabel("勝利場數", fontsize=12)
    ax.set_title("對戰結果", fontsize=14)
    ax.set_ylim(0, max(wins) + 5)

    # 在柱狀圖上加數字
    for bar, v in zip(bars, wins):
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    plot_battle_results_from_csv()