"""棋盘游戏控制"""

import numpy as np
import copy
import time
from config import CONFIG
from collections import deque   # 这个队列用来判断长将或长捉
import random
import random
def generate_dark_chess_board():
    red_pieces = ['红帅'] + ['红士'] * 2 + ['红象'] * 2 + ['红马'] * 2 + ['红车'] * 2 + ['红炮'] * 2 + ['红兵'] * 5
    black_pieces = ['黑帅'] + ['黑士'] * 2 + ['黑象'] * 2 + ['黑马'] * 2 + ['黑车'] * 2 + ['黑炮'] * 2 + ['黑兵'] * 5
    all_pieces = red_pieces + black_pieces
    random.shuffle(all_pieces)
    board = []
    for i in range(4):
        row = []
        for j in range(8):
            row.append('暗棋')
        board.append(row)

    return board
# 列表来表示棋盘，红方在上，黑方在下。使用时需要使用深拷贝
state_list_init = generate_dark_chess_board()


# deque来存储棋盘状态，长度为4
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))


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
def get_all_legal_moves():
    _move_id2move_action = {}
    _move_action2move_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 士的全部走法
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    # 象的全部走法
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    _move_id2move_action[idx] = action
                    _move_action2move_id[action] = idx
                    idx += 1

    for action in advisor_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    for action in bishop_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    return _move_id2move_action, _move_action2move_id


move_id2move_action, move_action2move_id = get_all_legal_moves_darkchess()


# 走子翻转的函数，用来扩充我们的数据
def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str


# 边界检查
def check_bounds(toY, toX):
    if toY in [0, 1, 2, 3, 4] and toX in [0, 1, 2, 3, 4, 5, 6, 7]:
        return True
    return False


# 不能走到自己的棋子位置
'''def check_obstruct_dark_chess(piece, current_player_color, current_piece):
    # 空格：可以走
    if piece == '一一':
        return True

    # 同色：不能吃
    if (current_player_color == '红' and '红' in piece) or \
       (current_player_color == '黑' and '黑' in piece):
        return False

    # 提取棋子強度（越大數字越強）
    def get_strength(p):
        if '兵' in p:
            return 1
        elif '炮' in p:
            return 0
        elif '馬' in p:
            return 2
        elif '象' in p:
            return 4
        elif '士' in p:
            return 5
        elif '車' in p:
            return 3
        elif '帅' in p:
            return 6
        else:
            return 0  # 預設（或未知）

    # 特判：兵可以吃兵與帥
    if ('兵' in current_piece) and \
       (('兵' in piece) or ('帅' in piece)):
        return True

    # 一般情況：大吃小
    return get_strength(current_piece) >= get_strength(piece)'''
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
        '紅兵': 1, '黑兵': 1,
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
                                break
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
    def init_board(self, start_player=1):   # 传入先手玩家的id
        # 增加一个颜色到id的映射字典，id到颜色的映射字典
        # 永远是红方先移动
        red_pieces = ['红帅'] + ['红士'] * 2 + ['红象'] * 2 + ['红马'] * 2 + ['红车'] * 2 + ['红炮'] * 2 + ['红兵'] * 5
        black_pieces = ['黑帅'] + ['黑士'] * 2 + ['黑象'] * 2 + ['黑马'] * 2 + ['黑车'] * 2 + ['黑炮'] * 2 + ['黑兵'] * 5
        covered_pieces = red_pieces + black_pieces
        start_player = random.choice([1,2])
        random.shuffle(covered_pieces)
        self.remain_pieces = covered_pieces
        self.first_move = True
        self.start_player = start_player

        if start_player == 1:
            self.id2color = {1: '红', 2: '黑'}
            self.color2id = {'红': 1, '黑': 2}
            self.backhand_player = 2
        elif start_player == 2:
            self.id2color = {2: '黑', 1: '红'}
            self.color2id = {'黑': 2, '红': 1}
            self.backhand_player = 1
        # 当前手玩家，也就是先手玩家
        self.current_player_color = self.id2color[start_player]     # 红
        if start_player == 1 :
            self.current_player_id = self.color2id['红']
        else:
            self.current_player_id = self.color2id['黑']
        # 初始化棋盘状态
        self.state_list = copy.deepcopy(state_list_init)
        self.state_deque = copy.deepcopy(state_deque_init)
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

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]  CHW
    def current_state(self):
        _current_state = np.zeros([9, 4, 8])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        _current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 10, 9]

        if self.game_start:
            # 解构self.last_move
            move = move_id2move_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        # 指出当前是哪个玩家走子
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0

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
        # 判断是否吃子
        if flip:
            if self.first_move:
                self.first_move = False
                while self.current_player_color not in self.remain_pieces[0]:
                    self.remain_pieces = self.remain_pieces[1:] + self.remain_pieces[:1]
            state_list[end_y][end_x] = self.remain_pieces[0]
            self.remain_pieces.pop(0)
        elif state_list[end_y][end_x] != '一一':
            # 如果吃掉对方的帅，则返回当前的current_player胜利
            self.kill_action = 0
            black_remain = False
            red_remain = False
            pieces_remain = False
            rows, cols = 4, 8
            for i in range(rows):
                for j in range(cols):
                    piece = state_list[i][j]
                    if i == end_y and j == end_x:
                        continue
                    if piece == '暗棋':
                        pieces_remain = True
                    if '黑' in piece:
                        black_remain = True
                    if '红' in piece:
                        red_remain = True
            if not black_remain and not pieces_remain:
                self.winner = self.color2id['红']
            elif not red_remain and not pieces_remain:
                self.winner = self.color2id['黑']
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

    # 是否产生赢家
    def has_a_winner(self):
        """一共有三种状态，红方胜，黑方胜，平局"""
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局先手判负
            # return False, -1
            return True, self.backhand_player
        return False, -1

    # 检查当前棋局是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局，没有赢家
            return True, -1
        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id


# 在Board类基础上定义Game类，该类用于启动并控制一整局对局的完整流程，并收集对局过程中的数据，以及进行棋盘的展示
class Game(object):

    def __init__(self, board):
        self.board = board

    # 可视化
    def graphic(self, board, player1_color, player2_color):
        print('player1 take: ', player1_color)
        print('player2 take: ', player2_color)
        print_board(state_list2state_array(board.state_deque[-1]))

    # 用于人机对战，人人对战等
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) '
                            'or 2 (player2 first)')
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player_id()  # 红子对应的玩家id
            player_in_turn = players[current_player]  # 决定当前玩家的代理
            move = player_in_turn.get_action(self.board)  # 当前玩家代理拿到动作
            print(move_id2move_action[move])
            print(self.board.remain_pieces)
            self.board.do_move(move)  # 棋盘做出改变

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Tie")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=False, temp=1e-3):
        self.board.init_board()     # 初始化棋盘, start_player=1
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        while True:
            _count += 1
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
            # 保存自我对弈的数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            # 执行一步落子
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                # 从每一个状态state对应的玩家的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                # 重置蒙特卡洛根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is:", winner)
                    else:
                        print('Game end. Tie')

                return winner, zip(states, mcts_probs, winner_z)


if __name__ == '__main__':
    # 测试array2string
    # _array = np.array([0, 0, 0, 0, 0, 0, 0])
    # print(array2num(_array))

    """# 测试change_state
    new_state = change_state(state_list_init, move='0010')
    for row in range(10):
        print(new_state[row])"""

    """# 测试print_board
    _state_list = copy.deepcopy(state_list_init)
    print_board(state_list2state_array(_state_list))"""

    """# 测试get_legal_moves
    moves = get_legal_moves(state_deque_init, current_player_color='黑')
    move_actions = []
    for item in moves:
        move_actions.append(move_id2move_action[item])
    print(move_actions)"""

    # 测试Board中的start_play
    # class Human1:
    #     def get_action(self, board):
    #         # print('当前是player1在操作')
    #         # print(board.current_player_color)
    #         # move = move_action2move_id[input('请输入')]
    #         move = random.choice(board.availables)
    #         return move
    #
    #     def set_player_ind(self, p):
    #         self.player = p
    #
    #
    # class Human2:
    #     def get_action(self, board):
    #         # print('当前是player2在操作')
    #         # print(board.current_player_color)
    #         # move = move_action2move_id[input('请输入')]
    #         move = random.choice(board.availables)
    #         return move
    #
    #     def set_player_ind(self, p):
    #         self.player = p
    #
    # human1 = Human1()
    # human2 = Human2()
    # game = Game(board=Board())
    # for i in range(20):
    #     game.start_play(human1, human2, start_player=2, is_shown=0)
    board = Board()
    board.init_board()




