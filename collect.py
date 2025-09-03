"""自我对弈收集数据"""
import random
from collections import deque
import copy
import os
import pickle
import time

import numpy as np

from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from mcts import MCTSPlayer
from config import CONFIG

if CONFIG['use_redis']:
    import my_redis, redis

import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


# 定义整个对弈收集数据流程
class CollectPipeline:

    def __init__(self, init_model=None):
        # 象棋逻辑和棋盘
        self.board = Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.take_multi = CONFIG['take_multiplier']
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

        # 根據 no_dark_mode 決定不同緩存檔案路徑
        if CONFIG.get('no_dark_mode', False):
            self.buffer_path = CONFIG.get('train_data_buffer_path_no_dark', 'train_data_buffer_no_dark.pth')
        else:
            self.buffer_path = CONFIG.get('train_data_buffer_path', 'train_data_buffer.pth')

        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()

    # 从主体加载模型
    def load_model(self):
        if CONFIG['use_frame'] == 'paddle':
            model_path = CONFIG['paddle_model_path']
        elif CONFIG['use_frame'] == 'pytorch':
            if CONFIG.get('no_dark_mode', False):
                model_path = 'current_policy_no_dark.pth'
            else:
                model_path = CONFIG['pytorch_model_path']
        else:
            print('暂不支持所选框架')
        try:
            self.policy_value_net = PolicyValueNet(model_file=model_path)
            print('已加载最新模型')
        except:
            self.policy_value_net = PolicyValueNet()
            print('已加载初始模型')
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        # 收集自我对弈的数据
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=True)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)

            if CONFIG['use_redis']:
                while True:
                    try:
                        for d in play_data:
                            self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                        self.redis_cli.incr('iters')
                        self.iters = self.redis_cli.get('iters')
                        print("存储完成")
                        break
                    except:
                        print("存储失败")
                        time.sleep(1)
            else:
                if os.path.exists(self.buffer_path):
                    while True:
                        try:
                            with open(self.buffer_path, 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                old_buffer = data_file['data_buffer']
                                self.iters = data_file['iters']
                            self.data_buffer.extend(old_buffer)
                            self.data_buffer.extend(play_data)
                            self.iters += 1
                            print('成功載入並追加資料')
                            break
                        except:
                            time.sleep(30)
                else:
                    self.data_buffer.extend(play_data)
                    self.iters += 1

                while len(self.data_buffer) > self.buffer_size:
                    self.data_buffer.popleft()

                # 寫入到指定 buffer_path
                data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
                with open(self.buffer_path, 'wb') as data_file:
                    pickle.dump(data_dict, data_file)
        return self.iters

    def run(self):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch i: {}, episode_len: {}'.format(
                    iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rquit')


# 初始化並運行
collecting_pipeline = CollectPipeline(init_model='current_policy.pth')
collecting_pipeline.run()

if CONFIG['use_frame'] == 'paddle':
    collecting_pipeline = CollectPipeline(init_model='current_policy.model')
    collecting_pipeline.run()
elif CONFIG['use_frame'] == 'pytorch':
    collecting_pipeline = CollectPipeline(init_model='current_policy.pth')
    collecting_pipeline.run()
else:
    print('暂不支持您选择的框架')
    print('训练结束')
