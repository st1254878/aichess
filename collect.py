"""自我对弈收集数据"""
import random
from collections import deque
import copy
import os
import pickle
import time
import numpy as np
from multiprocessing import Process, Queue, cpu_count

from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from mcts import MCTSPlayer
from config import CONFIG
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
        self.board = Board()
        self.game = Game(self.board)

        # 对弈参数
        self.temp = 1
        self.take_multi = CONFIG['take_multiplier']
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

        if CONFIG.get('no_dark_mode', False):
            self.buffer_path = CONFIG.get('train_data_buffer_path_no_dark', 'train_data_buffer_no_dark.pth')
        else:
            self.buffer_path = CONFIG.get('train_data_buffer_path', 'train_data_buffer.pth')

    def load_model(self):
        if CONFIG['use_frame'] == 'paddle':
            model_path = CONFIG['paddle_model_path']
        elif CONFIG['use_frame'] == 'pytorch':
            model_path = 'current_policy_no_dark.pth' if CONFIG.get('no_dark_mode', False) else CONFIG['pytorch_model_path']
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
        extend_data = []
        for state, mcts_prob, winner in play_data:
            extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
        return extend_data

    def collect_selfplay_data(self, n_games=1, is_shown=False):
        play_data_all = []
        for _ in range(n_games):
            self.load_model()
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=is_shown)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data_all.extend(self.get_equi_data(play_data))
        return play_data_all

    def save_data(self, play_data):
        """存到本地 buffer 檔"""
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, 'rb') as data_dict:
                data_file = pickle.load(data_dict)
                old_buffer = data_file['data_buffer']
                self.iters = data_file['iters']
            self.data_buffer.extend(old_buffer)

        self.data_buffer.extend(play_data)
        self.iters += 1

        while len(self.data_buffer) > self.buffer_size:
            self.data_buffer.popleft()

        data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
        with open(self.buffer_path, 'wb') as data_file:
            pickle.dump(data_dict, data_file)

    def run(self):
        """单进程蒐集"""
        total_time, total_batches = 0, 0
        try:
            while True:
                start_time = time.time()
                play_data = self.collect_selfplay_data(n_games=1, is_shown=True)
                self.save_data(play_data)
                elapsed = time.time() - start_time
                total_time += elapsed
                total_batches += 1
                print(f"batch i: {self.iters}, episode_len: {self.episode_len}, 本批耗時: {elapsed:.2f} 秒")
        except KeyboardInterrupt:
            if total_batches > 0:
                print(f"\n總共完成 {total_batches} 個 batch，平均每批耗時 {total_time / total_batches:.2f} 秒")
            print('\nquit')

    # -------------------- 平行版本 --------------------
    @staticmethod
    def _worker(proc_id, n_games, queue):
        pipeline = CollectPipeline()
        pipeline.load_model()
        data = pipeline.collect_selfplay_data(n_games=n_games, is_shown=False)
        queue.put((proc_id, data))

    def parallel_batch(self, total_games=20, n_procs=None):
        """執行一次平行蒐集"""
        if n_procs is None:
            n_procs = min(cpu_count(), 3)

        games_per_proc = total_games // n_procs
        queue = Queue()
        procs = []

        start_time = time.time()
        for pid in range(n_procs):
            p = Process(target=CollectPipeline._worker, args=(pid, games_per_proc, queue))
            p.start()
            procs.append(p)

        all_data = []
        for _ in range(n_procs):
            _, data = queue.get()
            all_data.extend(data)

        for p in procs:
            p.join()

        self.save_data(all_data)
        elapsed = time.time() - start_time
        print(f"完成 {total_games} 局 ({n_procs} 進程)，耗時 {elapsed:.2f} 秒，buffer大小 {len(self.data_buffer)}")
        return elapsed

    def parallel_run(self, total_games=20, n_procs=None):
        """平行長時間蒐集 (類似 run)"""
        total_time, total_batches = 0, 0
        try:
            while True:
                elapsed = self.parallel_batch(total_games=total_games, n_procs=n_procs)
                total_time += elapsed
                total_batches += 1
        except KeyboardInterrupt:
            if total_batches > 0:
                print(f"\n總共完成 {total_batches} 個 batch，平均每批耗時 {total_time / total_batches:.2f} 秒")
            print('\nquit')


# -------------------- 初始化 --------------------
if __name__ == "__main__":
    collecting_pipeline = CollectPipeline(init_model='current_policy.pth')
    # 單進程
    #collecting_pipeline.run()
    # 多進程 (每批 40 局，4 進程)
    collecting_pipeline.parallel_run(total_games=40, n_procs=4)
