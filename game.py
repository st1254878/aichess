"""棋盘游戏控制"""

import numpy as np
import copy
import time

from matplotlib import pyplot as plt

from config import CONFIG
from collections import deque   # 这个队列用来判断长将或长捉
import random
import random
def generate_dark_chess_board():
    board = []
    for i in range(4):
        row = []
        for j in range(8):
            row.append('暗棋')
        board.append(row)

    return board


def init_full_open_board():
    # 假設 all_pieces 是所有 32 個棋子的列表，例如 ["rP", "rP", ..., "bK"]
    red_pieces = ['红帅'] + ['红士'] * 2 + ['红象'] * 2 + ['红马'] * 2 + ['红车'] * 2 + ['红炮'] * 2 + ['红兵'] * 5
    black_pieces = ['黑帅'] + ['黑士'] * 2 + ['黑象'] * 2 + ['黑马'] * 2 + ['黑车'] * 2 + ['黑炮'] * 2 + ['黑兵'] * 5
    all_pieces = red_pieces + black_pieces
    random.shuffle(all_pieces)
    board = []
    index = 0
    for y in range(4):
        row = []
        for x in range(8):
            row.append(all_pieces[index])
            index += 1
        board.append(row)

    return board
    # 沒有暗棋了，所以 remain_pieces 清空
# 列表来表示棋盘，红方在上，黑方在下。使用时需要使用深拷贝
state_list_init = generate_dark_chess_board()
non_covered_state_list_init = init_full_open_board()

# deque来存储棋盘状态，长度为4
non_covered_state_deque_init = deque(maxlen=4)
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))
for _ in range(4):
    non_covered_state_deque_init.append(copy.deepcopy(non_covered_state_list_init))

# 构建一个字典：字符串到数组的映射，函数：数组到字符串的映射
string2array = dict(红车=np.array([1, 0, 0, 0, 0, 0, 0]), 红马=np.array([0, 1, 0, 0, 0, 0, 0]),
                    红象=np.array([0, 0, 1, 0, 0, 0, 0]), 红士=np.array([0, 0, 0, 1, 0, 0, 0]),
                    红帅=np.array([0, 0, 0, 0, 1, 0, 0]), 红炮=np.array([0, 0, 0, 0, 0, 1, 0]),
                    红兵=np.array([0, 0, 0, 0, 0, 0, 1]), 黑车=np.array([-1, 0, 0, 0, 0, 0, 0]),
                    黑马=np.array([0, -1, 0, 0, 0, 0, 0]), 黑象=np.array([0, 0, -1, 0, 0, 0, 0]),
                    黑士=np.array([0, 0, 0, -1, 0, 0, 0]), 黑帅=np.array([0, 0, 0, 0, -1, 0, 0]),
                    黑炮=np.array([0, 0, 0, 0, 0, -1, 0]), 黑兵=np.array([0, 0, 0, 0, 0, 0, -1]),
                    一一=np.array([0, 0, 0, 0, 0, 0, 0]), 暗棋=np.array([1, 1, 1, 1, 1, 1, 1]))


def array2string(array):
    return list(filter(lambda string: (string2array[string] == array).all(), string2array))[0]


# 改变棋盘状态
def change_state(state_list, move):
    """move : 字符串'0010'"""
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    return copy_list


# 打印盘面，可视化用到
def print_board(_state_array):
    # _state_array: [10, 9, 7], HWC
    board_line = []
    for i in range(4):
        for j in range(8):
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()


# 列表棋盘状态到数组棋盘状态
def state_list2state_array(state_list):
    _state_array = np.zeros([4, 8, 7])
    for i in range(4):
        for j in range(8):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array


# 拿到所有合法走子的集合，2086长度，也就是神经网络预测的走子概率向量的长度
# 第一个字典：move_id到move_action
# 第二个字典：move_action到move_id
# 例如：move_id:0 --> move_action:'0010'
def get_all_legal_moves_darkchess():
    _move_id2move_action = {}
    _move_action2move_id = {}
    idx = 0

    rows = 4
    cols = 8

    for r in range(rows):
        for c in range(cols):
            from_pos = f"{r}{c}"

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r, c
                while 0 <= nr < rows and 0 <= nc < cols:
                    to_pos = f"{nr}{nc}"
                    action = from_pos + to_pos
                    if action not in _move_action2move_id:
                        _move_id2move_action[idx] = action
                        _move_action2move_id[action] = idx
                        idx += 1
                    nr += dr
                    nc += dc


    return _move_id2move_action, _move_action2move_id

move_id2move_action, move_action2move_id = get_all_legal_moves_darkchess()


# 走子翻转的函数，用来扩充我们的数据
def flip_map(string):
    # string 是 4 位數字字串，例如 "0213"
    # 表示從 (2,1) -> (3,3)
    new_str = ''
    for index in range(4):
        digit = int(string[index])
        if index == 1 or index == 3:
            # 翻轉列（左右鏡像）
            digit = 7 - digit
        new_str += str(digit)
    return new_str

# 边界检查
def check_bounds(toY, toX):
    if toY in [0, 1, 2, 3, 4] and toX in [0, 1, 2, 3, 4, 5, 6, 7]:
        return True
    return False


# 不能走到自己的棋子位置
def check_obstruct(piece, current_player_color):
    # 当走到的位置存在棋子的时候，进行一次判断
    if piece != '一一':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True


# 得到当前盘面合法走子集合
# 输入状态队列不能小于10，current_player_color:当前玩家控制的棋子颜色
# 用来存放合法走子的列表，例如[0, 1, 2, 1089, 2085]
def get_legal_moves(state_deque, current_player_color):

    state_list = state_deque[-1]
    old_state_list = state_deque[-4]

    moves = []  # 用来存放所有合法的走子方法
    # state_list是以列表形式表示的, len(state_list) == 10, len(state_list[0]) == 9
    # 遍历移动初始位置
    rows, cols = 4, 8  # 暗棋盤面

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    # 棋子等級 越大越強
    piece_strength = {
        '红炮': 0, '黑炮': 0,
        '红兵': 1, '黑兵': 1,
        '红马': 2, '黑马': 2,
        '红车': 3, '黑车': 3,
        '红象': 4, '黑象': 4,
        '红士': 5, '黑士': 5,
        '红帅': 6, '黑帅': 6,
        '暗棋': 10
    }
    # 棋子轉等級
    def get_strength(piece):
        for name in piece_strength:
            if name in piece:
                return piece_strength[name]
        return 0
    for i in range(rows):
        for j in range(cols):
            piece = state_list[i][j]
            if piece == '暗棋':
                m = str(i) + str(j) + str(i) + str(j)
                if change_state(state_list, m) != old_state_list:
                    moves.append(m)
            elif piece != '一一' and current_player_color in piece:
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        target_piece = state_list[ni][nj]
                        m = str(i) + str(j) + str(ni) + str(nj)
                        if target_piece == '一一':
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        elif current_player_color not in target_piece  and '炮' not in piece:
                            if '帅' in piece and '兵' in target_piece:
                                continue
                            elif '兵' in piece and '帅' in target_piece:
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            else:
                                if  get_strength(piece) >= get_strength(target_piece):
                                    if change_state(state_list, m) != old_state_list:
                                        moves.append(m)
            # 炮的合理吃法
            if '炮' in piece and current_player_color in piece:
                toY = i
                hits = False
                for toX in range(j - 1, -1, -1):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                hits = False
                for toX in range(j + 1, 8):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                toX = j
                hits = False
                for toY in range(i - 1, -1, -1):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break
                hits = False
                for toY in range(i + 1, 4):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                if change_state(state_list, m) != old_state_list:
                                    moves.append(m)
                            break

    moves_id = []
    for move in moves:
        moves_id.append(move_action2move_id[move])
    return moves_id

def get_greedy_move(state_deque, current_player_color):
    state_list = state_deque[-1]
    current_color = current_player_color
    legal_moves = get_legal_moves(state_deque, current_color)

    eat_moves = []
    fallback_moves = []

    for move_id in legal_moves:
        move = move_id2move_action[move_id]
        y, x, to_y, to_x = int(move[0]), int(move[1]), int(move[2]), int(move[3])
        start_piece = state_list[y][x]
        target_piece = state_list[to_y][to_x]

        # 翻棋的情況（座標相同）
        if y == to_y and x == to_x:
            fallback_moves.append(move_id)
        elif target_piece != '一一' and current_color not in target_piece and '暗棋' not in target_piece:
            # 吃敵方棋
            eat_moves.append(move_id)
        else:
            # 普通走棋
            fallback_moves.append(move_id)

    return eat_moves,fallback_moves

# 棋盘逻辑控制
class Board(object):

    def __init__(self):
        self.remain_pieces = []
        self.first_move = True
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque_init)

    # 初始化棋盘的方法
    def init_board(self, start_player = random.choice([1, 2])):   # 传入先手玩家的id
        # 增加一个颜色到id的映射字典，id到颜色的映射字
        # 永远是红方先移动

        # 初始化棋盘状态
        if CONFIG.get("no_dark_mode", True):
            self.remain_pieces = []
            self.state_list = copy.deepcopy(non_covered_state_list_init)
            self.state_deque = copy.deepcopy(non_covered_state_deque_init)
        else:
            red_pieces = ['红帅'] + ['红士'] * 2 + ['红象'] * 2 + ['红马'] * 2 + ['红车'] * 2 + ['红炮'] * 2 + [
                '红兵'] * 5
            black_pieces = ['黑帅'] + ['黑士'] * 2 + ['黑象'] * 2 + ['黑马'] * 2 + ['黑车'] * 2 + ['黑炮'] * 2 + [
                '黑兵'] * 5
            covered_pieces = red_pieces + black_pieces
            random.shuffle(covered_pieces)
            self.remain_pieces = covered_pieces
            self.state_list = copy.deepcopy(state_list_init)
            self.state_deque = copy.deepcopy(state_deque_init)
        self.first_move = True
        self.start_player = start_player
        self.id2color = {1: '红', 2: '黑'}
        self.color2id = {'红': 1, '黑': 2}
        if start_player == 1:
            self.backhand_player = 2
        else:
            self.backhand_player = 1
        # 当前手玩家，也就是先手玩家
        self.current_player_color = self.id2color[start_player]
        if start_player == 1 :
            self.current_player_id = self.color2id['红']
        else:
            self.current_player_id = self.color2id['黑']
        # 初始化最后落子位置
        self.last_move = -1
        # 记录游戏中吃子的回合数
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0   # 游戏动作计数器
        self.winner = None

    @property
    # 获的当前盘面的所有合法走子集合
    def availables(self):
        return get_legal_moves(self.state_deque, self.current_player_color)

    def greedys(self):
        return get_greedy_move(self.state_deque, self.current_player_color)

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]  CHW
    def current_state(self):
        _current_state = np.zeros([9, 4, 8])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        _current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 4, 8]

        if self.game_start:
            # 解构self.last_move
            move = move_id2move_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        # 指出当前是哪个玩家走子
        if self.current_player_id == self.start_player:
            _current_state[8][:, :] = 1.0
        else:
            _current_state[8][:, :] = 0.0

        return _current_state

    # 根据move对棋盘状态做出改变
    def do_move(self, move):
        self.game_start = True  # 游戏开始
        self.action_count += 1  # 移动次数加1
        move_action = move_id2move_action[move]
        flip = False
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        if start_x == end_x and start_y == end_y:
            flip = True
        state_list = copy.deepcopy(self.state_deque[-1])
        reward = 0.0  # 新增 reward
        # 判断是否吃子
        if flip:
            if self.first_move:
                self.first_move = False
                while self.current_player_color not in self.remain_pieces[0]:
                    random.shuffle(self.remain_pieces)
            else:
                random.shuffle(self.remain_pieces)

            state_list[end_y][end_x] = self.remain_pieces[0]
            self.remain_pieces.pop(0)
        elif state_list[end_y][end_x] != '一一':
            # 如果吃掉对方的帅，则返回当前的current_player胜利
            reward = self.get_piece_value(state_list[end_y][end_x])
            self.kill_action = 0
        else:
            self.kill_action += 1
        # 更改棋盘状态
        if not flip:
            state_list[end_y][end_x] = state_list[start_y][start_x]
            state_list[start_y][start_x] = '一一'
        self.current_player_color = '黑' if self.current_player_color == '红' else '红'  # 改变当前玩家
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        # 记录最后一次移动的位置
        self.last_move = move
        self.state_deque.append(state_list)
        eat, fallback = self.greedys()
        if eat == [] and fallback == []:
            if self.current_player_color == '红':
                self.winner = self.color2id['黑']
            else:
                self.winner = self.color2id['红']
        return reward
    # 是否产生赢家
    def has_a_winner(self):
        """一共有三种状态，红方胜，黑方胜，平局"""
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局先手判负
            red_strength = self.calc_side_strength('红')
            black_strength = self.calc_side_strength('黑')
            if red_strength > black_strength:
                return True, self.color2id['红']
            elif black_strength > red_strength:
                return True, self.color2id['黑']
            else:
                return True, -1  # 平局
        return False, -1

    # 检查当前棋局是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner

        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id

    def get_piece_value(self, piece):
        """給每個棋子一個分值"""
        values = {
            '帅': 8, '士': 6, '象': 4, '马': 2, '车': 3, '炮': 4, '兵': 1
        }
        if piece == '一一':  # 空位
            return 0
        # 去掉顏色，只保留棋子名稱
        name = piece[1:]
        return values.get(name, 0)

    def calc_side_strength(self, color):
        """計算指定顏色的棋力總分"""
        strength = 0
        for row in self.state_deque[-1]:
            for p in row:
                if p.startswith(color):
                    strength += self.get_piece_value(p)
        return strength

# 在Board类基础上定义Game类，该类用于启动并控制一整局对局的完整流程，并收集对局过程中的数据，以及进行棋盘的展示
class Game(object):

    def __init__(self, board):
        self.board = board

    # 可视化
    def graphic(self, board):
        print_board(state_list2state_array(board.state_deque[-1]))

    # 用于人机对战，人人对战等
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) '
                            'or 2 (player2 first)')
        self.board.init_board(start_player)  # 初始化棋盘
        if start_player==1:
            p1, p2 = 1, 2
            player1.set_player_ind(1)
            player2.set_player_ind(2)
            players = {p1: player1, p2: player2}
        else:
            p1, p2 = 1, 2
            player1.set_player_ind(1)
            player2.set_player_ind(2)
            players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board)

        while True:
            current_player = self.board.get_current_player_id()  # 红子对应的玩家id
            player_in_turn = players[current_player]  # 决定当前玩家的代理
            move = player_in_turn.get_action(self.board)  # 当前玩家代理拿到动作

            # print(self.board.remain_pieces)
            self.board.do_move(move)  # 棋盘做出改变

            if is_shown:
                self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    red_strength = self.board.calc_side_strength('红')
                    black_strength = self.board.calc_side_strength('黑')
                    if red_strength != 0 and black_strength != 0:
                        print("紅 : ", red_strength,sep="")
                        print("黑 : ", black_strength, sep="")
                if winner != -1:
                    print("Game end. Winner is", players[winner]," ",self.board.current_player_color,sep="")
                else:
                    print("Game end. Tie")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=True, temp=1e-3):
        self.board.init_board()     # 初始化棋盘, start_player=1
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        eat_count = 0
        rewards = []
        while True:
            _count += 1
            self.graphic(self.board)
            eat_move_list, fallback_movelist = self.board.greedys()
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
                print('走一步要花: ', time.time() - start_time)
            else:
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            # 保存自我对弈的数据
            # 轉換成 numpy array

            # 执行一步落子
            reward = self.board.do_move(move)
            if reward != 0:
                if self.board.current_player_id == 1:  # 現在輪到紅，剛剛是黑吃子
                    rewards.append(-reward)
                else:  # 現在輪到黑，剛剛是紅吃子
                    rewards.append(reward)
            else:
                rewards.append(0.0)
            print(reward)
            end, winner = self.board.game_end()
            if end:
                red_strength = self.board.calc_side_strength('红')
                black_strength = self.board.calc_side_strength('黑')
                total_abs = sum(abs(r) for r in rewards) + 1e-8
                rewards = [0.8 * r / total_abs for r in rewards]
                # 从每一个状态state对应的玩家的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                # 重置蒙特卡洛根节点
                winner_z += np.array(rewards)

                red_rewards = [r for r, p in zip(winner_z, current_players) if p == 1]
                black_rewards = [r for r, p in zip(winner_z, current_players) if p == 2]

                print("Red rewards:", red_rewards)
                print("Black rewards:", black_rewards)
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is:", winner)
                    else:
                        print('Game end. Tie')

                return winner, zip(states, mcts_probs, winner_z)


if __name__ == '__main__':
    board = Board()
    board.init_board()




