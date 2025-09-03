import pygame
import sys
import copy
import random
from game import move_action2move_id, Game, Board
from mcts import MCTSPlayer
import time
from config import CONFIG
from mcts_pure import MCTS_Pure

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')
class greedy_player:
    def __init__(self):
        self.agent = 'AI'

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        eat_move_list,fallback_movelist = board.greedys()
        if eat_move_list:
            move = random.choice(eat_move_list)
        else:
            move = random.choice(fallback_movelist)
        time.sleep(0.5)
        return move

class Human:

    def __init__(self):
        self.agent = 'HUMAN'

    def get_action(self, move):
        # move从鼠标点击事件触发
        # print('当前是player2在操作')
        # print(board.current_player_color)
        if  move_action2move_id.__contains__(move):
            move = move_action2move_id[move]
        else:
            move = -1
        # move = random.choice(board.availables)
        return move

    def set_player_ind(self, p):
        self.player = p


if CONFIG['use_frame'] == 'paddle':
    policy_value_net = PolicyValueNet(model_file='current_policy.model')
elif CONFIG['use_frame'] == 'pytorch':
    if CONFIG.get('no_dark_mode', True):
        policy_value_net = PolicyValueNet(model_file='current_policy_no_dark.pth')
    else:
        policy_value_net = PolicyValueNet(model_file='current_policy.pth')

    #policy_value_net = PolicyValueNet(model_file=None)
else:
    print('暂不支持您选择的框架')

# 初始化pygame
pygame.init()
pygame.mixer.init()
size = width, height = 700, 700
bg_image = pygame.image.load('imgs/board.png')  #图片位置
bg_image = pygame.transform.smoothscale(bg_image, size)

clock = pygame.time.Clock()
fullscreen = False
# 创建指定大小的窗口
screen = pygame.display.set_mode(size)
# 设置窗口标题
pygame.display.set_caption("暗棋")

# 加载一个列表进行图像的绘制
# 列表表示的棋盘初始化，红子在上，黑子在下，禁止对该列表进行编辑，使用时必须使用深拷贝

board_list_init = []
for i in range(4):
    row = []
    for j in range(8):
        row.append('暗棋')
    board_list_init.append(row)

# 加载棋子被选中的图片
fire_image = pygame.transform.smoothscale(pygame.image.load("imgs/fire.png").convert_alpha(), (width // 10, height // 10))
fire_image.set_alpha(200)

# 制作一个从字符串到pygame.surface对象的映射
str2image = {
    '暗棋': pygame.transform.smoothscale(pygame.image.load("imgs/blankchess.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红车': pygame.transform.smoothscale(pygame.image.load("imgs/hongche.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红马': pygame.transform.smoothscale(pygame.image.load("imgs/hongma.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红象': pygame.transform.smoothscale(pygame.image.load("imgs/hongxiang.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红士': pygame.transform.smoothscale(pygame.image.load("imgs/hongshi.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红帅': pygame.transform.smoothscale(pygame.image.load("imgs/hongshuai.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红炮': pygame.transform.smoothscale(pygame.image.load("imgs/hongpao.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红兵': pygame.transform.smoothscale(pygame.image.load("imgs/hongbing.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑车': pygame.transform.smoothscale(pygame.image.load("imgs/heiche.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑马': pygame.transform.smoothscale(pygame.image.load("imgs/heima.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑象': pygame.transform.smoothscale(pygame.image.load("imgs/heixiang.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑士': pygame.transform.smoothscale(pygame.image.load("imgs/heishi.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑帅': pygame.transform.smoothscale(pygame.image.load("imgs/heishuai.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑炮': pygame.transform.smoothscale(pygame.image.load("imgs/heipao.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑兵': pygame.transform.smoothscale(pygame.image.load("imgs/heibing.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
}
str2image_rect = {
    '暗棋': str2image['暗棋'].get_rect(),
    '红车': str2image['红车'].get_rect(),
    '红马': str2image['红马'].get_rect(),
    '红象': str2image['红象'].get_rect(),
    '红士': str2image['红士'].get_rect(),
    '红帅': str2image['红帅'].get_rect(),
    '红炮': str2image['红炮'].get_rect(),
    '红兵': str2image['红兵'].get_rect(),
    '黑车': str2image['黑车'].get_rect(),
    '黑马': str2image['黑马'].get_rect(),
    '黑象': str2image['黑象'].get_rect(),
    '黑士': str2image['黑士'].get_rect(),
    '黑帅': str2image['黑帅'].get_rect(),
    '黑炮': str2image['黑炮'].get_rect(),
    '黑兵': str2image['黑兵'].get_rect()
}

str2image_selected = {}
for key, img in str2image.items():
    w, h = img.get_size()
    scale_img = pygame.transform.smoothscale(img, (int(w * 1.15), int(h * 1.15)))  # 放大 1.2 倍
    str2image_selected[key] = scale_img
# 根据棋盘列表获得最新位置
# 返回一个由image和rect对象组成的列表
x_ratio = 80
y_ratio = 72
x_bais = 70
y_bais = 65
def board2image(board):
    return_image_rect = []
    for i in range(4):
        for j in range(8):
            piece = board[i][j]
            if piece not in str2image:
                continue

            is_selected = 'start_i_j' in globals() and (j, i) == start_i_j

            # 中心位置
            center_x = j * x_ratio + x_bais
            center_y = i * y_ratio + y_bais

            if is_selected:
                img = str2image_selected[piece]
            else:
                img = str2image[piece]

            rect = img.get_rect()
            rect.center = (center_x, center_y)
            return_image_rect.append((img, rect))
    return return_image_rect


fire_rect = fire_image.get_rect()
fire_rect.center = (0 * x_ratio + x_bais, 3 * y_ratio + y_bais)

# 加载两个玩家，AI对AI，或者AI对human
board=Board()
# 开始的玩家
start_player = 1

player_human = Human()

player_random = MCTS_Pure(500)

player_RL = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=5,
                                 n_playout=1000,
                                 is_selfplay=0)
player_RL2 = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=5,
                                 n_playout=1000,
                                 is_selfplay=0)
player_greedy = greedy_player()

# player2 = Human()

board.init_board(start_player)
p1, p2 = 1, 2
player_RL2.set_player_ind(1)
player_RL.set_player_ind(2)
player_human.set_player_ind(2)
players = {p1: player_RL, p2: player_human}


# 切换玩家
swicth_player = True
draw_fire = False
move_action = ''
first_button = False
while True:

    # 填充背景
    screen.blit(bg_image, (0, 0))
    for image, image_rect in board2image(board=board.state_deque[-1]):
        screen.blit(image, image_rect)
    if draw_fire:
        screen.blit(fire_image, fire_rect)
    # 更新界面
    pygame.display.update()
    # 不高于60帧
    clock.tick(60)
    player_color = board.current_player_color
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:    #按下按键
            # print("[MOUSEBUTTONDOWN]", event.pos, event.button)
            mouse_x, mouse_y = event.pos
            if not first_button:
                for i in range(4):
                    for j in range(8):
                        if abs(80 * j + 70 - mouse_x) < 30 and abs(72 * i + 70 - mouse_y) < 30:
                            piece_name = board.state_deque[-1][i][j]
                            if piece_name != '暗棋' and piece_name != '':
                                if player_color == '红' and not piece_name.startswith('红'):
                                    continue  # 紅方玩家不能點黑棋
                                if player_color == '黑' and not piece_name.startswith('黑'):
                                    continue  # 黑方玩家不能點紅棋
                            first_button = True
                            start_i_j = j, i
                            fire_rect.center = (start_i_j[0] * x_ratio + x_bais, start_i_j[1] * y_ratio + y_bais)
                            # print(start_i_j)
                            # screen.blit(fire_image, fire_rect)
                            break

            elif first_button:
                for i in range(4):
                    for j in range(8):
                        if abs(80 * j + 70 - mouse_x) < 30 and abs(72 * i + 70 - mouse_y) < 30:
                            first_button = False
                            end_i_j = j, i
                            move_action = str(start_i_j[1]) + str(start_i_j[0]) + str(end_i_j[1]) + str(end_i_j[0])
                            # screen.blit(fire_image, fire_rect)

    if swicth_player:
        current_player = board.get_current_player_id()  # 红子对应的玩家id
        player_in_turn = players[current_player]  # 决定当前玩家的代理

    if player_in_turn.agent == 'AI':
        pygame.display.update()
        start_time = time.time()
        move = player_in_turn.get_action(board)  # 当前玩家代理拿到动作
        print('耗时：', time.time() - start_time)
        board.do_move(move)  # 棋盘做出改变
        swicth_player = True
        if 'start_i_j' in globals():
            del start_i_j
    elif player_in_turn.agent == 'HUMAN':
        swicth_player = False
        if len(move_action) == 4:
            move = player_in_turn.get_action(move_action)  # 人類從UI滑鼠操作產生的move_action
            if move != -1 and move in board.availables:
                board.do_move(move)
                swicth_player = True
                move_action = ''
                if 'start_i_j' in globals():
                    del start_i_j

    end, winner = board.game_end()
    if end:
        if winner != -1:
            print("Game end. Winner is", players[winner])
        else:
            print("Game end. Tie")
        sys.exit()