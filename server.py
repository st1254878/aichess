#server.py
import random

from flask import Flask, render_template, request, jsonify
from game import Board, move_action2move_id
from mcts import MCTSPlayer
from pytorch_net import PolicyValueNet
import os
from dotenv import load_dotenv
load_dotenv()
import uuid
import threading
import time
import torch

# ---------------------------
# Config
# ---------------------------
SESSION_TTL = int(os.environ.get("SESSION_TTL", 60 * 60))  # default 1 hour
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 50))
MCTS_PLAYOUTS = int(os.environ.get("MCTS_PLAYOUTS", 300))

# ---------------------------
# Flask app
# ---------------------------
app = Flask(
    __name__,
    template_folder=os.path.join("web_play", "templates"),
    static_folder=os.path.join("web_play", "static")
)

# ---------------------------
# Model (shared)
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[server] Using device: {device}")
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))

# Create a single PolicyValueNet instance (shared) to avoid heavy reloads
policy_value_net = PolicyValueNet(model_file='new_current_policy.pth', device=device)

# ---------------------------
# Session store
# ---------------------------
# sessions: { session_id: { 'board': Board(), 'ai_player': MCTSPlayer(...), 'human_first': bool, 'last_active': timestamp, 'lock': threading.Lock() } }
sessions = {}
sessions_lock = threading.Lock()


def _prune_sessions():
    """Background thread: remove sessions not active for SESSION_TTL seconds or when over MAX_SESSIONS."""
    while True:
        time.sleep(60)
        now = time.time()
        with sessions_lock:
            # remove stale sessions
            stale = [sid for sid, s in sessions.items() if now - s['last_active'] > SESSION_TTL]
            for sid in stale:
                try:
                    del sessions[sid]
                except KeyError:
                    pass

            # limit total sessions
            if len(sessions) > MAX_SESSIONS:
                # evict oldest
                items = sorted(sessions.items(), key=lambda kv: kv[1]['last_active'])
                for sid, _ in items[: len(sessions) - MAX_SESSIONS]:
                    try:
                        del sessions[sid]
                    except KeyError:
                        pass


# start background thread
_pruner = threading.Thread(target=_prune_sessions, daemon=True)
_pruner.start()


# ---------------------------
# Helpers
# ---------------------------

def _new_session(side='first'):
    """Create a new session and return session_id and initial board state.
    Behavior:
    - Player chooses only 'first' or 'second'.
    - Colors are randomly assigned (human_color randomly '红' or '黑').
    - start_player passed to Board.init_board is the player id who moves first.
    - AI player id is set consistently with the board.
    - If AI moves first (human chose 'second'), AI makes the first move immediately.
    """
    sid = uuid.uuid4().hex
    board = Board()

    # fixed mapping (Board uses the same mapping in init_board)
    color2id = {'红': 1, '黑': 2}

    # randomize human color (player cannot choose color)
    human_color = random.choice(['红', '黑'])
    human_id = color2id[human_color]

    # determine who moves first (start_player id)
    if side == 'first':
        start_player = human_id   # human moves first
        human_first = True
    else:
        # human is second -> AI moves first, so start_player should be the AI's id
        ai_color = '红' if human_color == '黑' else '黑'
        start_player = color2id[ai_color]
        human_first = False

    # initialize board with start_player (1 => 红, 2 => 黑)
    board.init_board(start_player=start_player)

    # create AI player and set its id consistent with board
    ai_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=1, n_playout=MCTS_PLAYOUTS)

    # decide ai_id:
    # - if human_first: current player is human (start_player), AI is the other id
    # - if human_second: current player is AI (start_player), AI id == start_player
    if human_first:
        ai_id = 3 - start_player
    else:
        ai_id = start_player

    ai_player.set_player_ind(ai_id)

    # if AI is to move first, have it play the opening move now
    if not human_first:
        move = ai_player.get_action(board)
        board.do_move(move)

    sessions[sid] = {
        'board': board,
        'ai_player': ai_player,
        'human_first': human_first,
        'human_color': human_color,   # store for front-end convenience
        'last_active': time.time(),
        'lock': threading.Lock()
    }
    return sid

def _get_session(sid):
    with sessions_lock:
        s = sessions.get(sid)
        if s:
            s['last_active'] = time.time()
        return s


# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start_game():
    """Create a new session and return session_id and initial board state.
    Request JSON: { 'side': 'first'|'second' }
    Response JSON: { 'session_id': sid, 'board': ..., 'human_first': bool }
    """
    data = request.get_json(force=True) or {}
    side = data.get('side', 'first')
    sid = _new_session(side=side)
    s = _get_session(sid)
    board_str = s['board'].get_board_str()
    return jsonify({
        'session_id': sid,
        'board': board_str,
        'human_first': s['human_first'],
        'human_color': s.get('human_color')  # '红' 或 '黑'
    })


@app.route('/state', methods=['GET'])
def get_state():
    """Query current board state. Provide session_id as query param or header.
    GET /state?session_id=...
    """
    sid = request.args.get('session_id')
    if not sid:
        return jsonify({'error': 'missing session_id'}), 400
    s = _get_session(sid)
    if not s:
        return jsonify({'error': 'invalid session_id'}), 400

    with s['lock']:
        board_str = s['board'].get_board_str()
        return jsonify({'board': board_str, 'current_player': s['board'].current_player_color})


# ---- 替換 /move route ----
@app.route('/move', methods=['POST'])
def human_move():
    data = request.get_json(force=True) or {}
    sid = data.get('session_id')
    move_action = data.get('move_action')
    if not sid or move_action is None:
        return jsonify({'error': 'missing session_id or move_action'}), 400

    s = _get_session(sid)
    if not s:
        return jsonify({'error': 'invalid session_id'}), 400

    with s['lock']:
        # debug: 印出動作與一些狀態
        try:
            print(f"[session {sid}] human_move: move_action={move_action}")
        except Exception:
            pass

        if move_action not in move_action2move_id:
            print(f"[session {sid}] unknown move_action: {move_action}")
            return jsonify({'error': 'invalid move'}), 400

        move_id = move_action2move_id[move_action]

        # 使用 Board 提供的 game_end 判斷（比單純檢查 availables 更穩健）
        game_over, winner = s['board'].game_end()
        if game_over:
            # 若遊戲已結束，回傳 game_over 給前端
            print(f"[session {sid}] move requested but game already over, winner={winner}")
            return jsonify({'error': 'game_already_over', 'winner': winner}), 400

        avail = getattr(s['board'], 'availables', [])
        # 印出 availables 方便 debug（發生 illegal move 時）
        # print(f"[session {sid}] availables_count={len(avail)} availables_sample={avail[:30]}")

        if move_id not in avail:
            # 在 illegal 狀況下回傳更多除錯資訊（前端或你用 curl 可看到）
            cur_board = s['board'].get_board_str()
            print(f"[session {sid}] illegal move: move_id={move_id} not in availables")
            print(f"[session {sid}] current_board:")
            for row in cur_board:
                print(row)
            return jsonify({
                'error': 'illegal move',
                'move_id': move_id,
                'availables': avail,
                'board': cur_board
            }), 400

        # apply move
        s['board'].do_move(move_id)

        # after applying, ask board if game ended
        game_over, winner = s['board'].game_end()
        board_str = s['board'].get_board_str()
        return jsonify({
            'status': 'ok',
            'game_over': game_over,
            'winner': winner,
            'board': board_str,
            'current_player': s['board'].current_player_color
        })




# ---- 替換 /ai_move route ----
@app.route('/ai_move', methods=['POST'])
def ai_move():
    data = request.get_json(force=True) or {}
    sid = data.get('session_id')
    if not sid:
        return jsonify({'error': 'missing session_id'}), 400

    s = _get_session(sid)
    if not s:
        return jsonify({'error': 'invalid session_id'}), 400

    with s['lock']:
        # 如果遊戲已結束，直接回傳（不用跑 MCTS）
        game_over, winner = s['board'].game_end()
        if game_over:
            print(f"[session {sid}] ai_move requested but game already over. winner={winner}")
            return jsonify({'game_over': True, 'winner': winner, 'board': s['board'].get_board_str()}), 200

        # 程式安全：如果 availables 空也直接回 game_over（避免 MCTS 在空集合上 crash）
        avail = getattr(s['board'], 'availables', [])
        if not avail:
            # 若 board 尚未被判定為 game_over，但 availables 為空，先回傳 game_over=true（以防 get_legal_moves 錯誤）
            print(f"[session {sid}] ai_move: availables empty -> treat as game over (fallback).")
            return jsonify({'game_over': True, 'board': s['board'].get_board_str()}), 200

        try:
            print(f"[session {sid}] AI computing move (n_playout={MCTS_PLAYOUTS}); avail_count={len(avail)}")
            move = int(s['ai_player'].get_action(s['board']))
        except Exception as e:
            print(f"[session {sid}] AI get_action exception: {e}")
            return jsonify({'error': 'ai_failed', 'detail': str(e)}), 500

        s['board'].do_move(move)
        game_over, winner = s['board'].game_end()
        board_str = s['board'].get_board_str()
        #print(f"[session {sid}] AI moved {move}; game_over={game_over}; winner={winner}")
        return jsonify({'ai_move': move, 'game_over': game_over, 'winner': winner, 'board': board_str})


@app.route('/sessions', methods=['GET'])
def list_sessions():
    with sessions_lock:
        return jsonify({'active_sessions': len(sessions), 'session_ids': list(sessions.keys())})


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    # optional ngrok (set NGROK_AUTH_TOKEN env var to enable)
    NGROK_AUTH_TOKEN = os.environ.get('NGROK_AUTH_TOKEN', '')
    try:
        if NGROK_AUTH_TOKEN:
            from pyngrok import ngrok
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            public_url = ngrok.connect(port)
            print(f"[ngrok] public url: {public_url}")
    except Exception as e:
        print("[ngrok] failed to start:", e)

    # production: prefer waitress/gunicorn. Use built-in server for dev/testing only.
    if os.environ.get('USE_WAITRESS') == '1':
        try:
            from waitress import serve
            print('[server] Starting with waitress')
            serve(app, host='0.0.0.0', port=port)
        except Exception as e:
            print('[server] waitress failed, falling back to Flask dev server:', e)
            app.run(host='0.0.0.0', port=port)
    else:
        app.run(host='0.0.0.0', port=port, threaded=True)
