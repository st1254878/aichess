from flask import Flask, render_template, request, jsonify
from game import Board, move_action2move_id
from mcts import MCTSPlayer
from pytorch_net import PolicyValueNet
import os

# ----------------------------------------
# 建立 Flask app，指定模板與靜態資料夾路徑
# ----------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join("web_play", "templates"),
    static_folder=os.path.join("web_play", "static")
)

# ----------------------------------------
# 初始化模型與 MCTS AI
# ----------------------------------------
policy_value_net = PolicyValueNet(model_file='new_current_policy.pth')
ai_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=1, n_playout=300)
board = Board()
human_first = True  # 用來記錄玩家是否先手


# ----------------------------------------
# 主頁
# ----------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------
# 遊戲開始
# ----------------------------------------
@app.route("/start", methods=["POST"])
def start_game():
    """根據選擇初始化棋盤"""
    global board, human_first
    data = request.json
    side = data.get("side")  # 'first' 或 'second'

    board = Board()
    if side == "first":
        human_first = True
        board.init_board(start_player=1)
        ai_player.set_player_ind(2)
    else:
        human_first = False
        board.init_board(start_player=2)
        ai_player.set_player_ind(1)
        # 如果 AI 先手，讓 AI 馬上走一步
        move = ai_player.get_action(board)
        board.do_move(move)

    return jsonify({"status": "started", "human_first": human_first})


# ----------------------------------------
# 取得當前棋盤狀態
# ----------------------------------------
@app.route("/state", methods=["GET"])
def get_state():
    """回傳棋盤狀態"""
    board_str = board.get_board_str()  # 假設這是 4x8 的 list[list[str]]
    return jsonify({
        "board": board_str,
        "current_player": board.current_player_color
    })


# ----------------------------------------
# 玩家走棋
# ----------------------------------------
@app.route("/move", methods=["POST"])
def human_move():
    """玩家走棋"""
    data = request.json
    move_action = data.get("move_action")
    if move_action not in move_action2move_id:
        return jsonify({"error": "invalid move"}), 400

    move_id = move_action2move_id[move_action]
    if move_id not in board.availables:
        return jsonify({"error": "illegal move"}), 400

    board.do_move(move_id)
    return jsonify({"status": "human move ok"})


# ----------------------------------------
# AI 走棋
# ----------------------------------------
@app.route("/ai_move", methods=["POST"])
def ai_move():
    move = int(ai_player.get_action(board))
    board.do_move(move)
    return jsonify({"ai_move": move})


# ----------------------------------------
# 主程式入口
# ----------------------------------------
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
