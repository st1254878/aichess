"""使用收集到数据进行训练"""
import os
import random
from collections import defaultdict, deque

import numpy as np
import pickle
import time

import zip_array
from config import CONFIG
from game import Game, Board
from mcts import MCTSPlayer
from mcts_pure import MCTS_Pure

if CONFIG['use_redis']:
    import my_redis, redis
    import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


# 定义整个训练流程
class TrainPipeline:

    def __init__(self, init_model=None):
        # 训练参数
        self.best_policy_net = PolicyValueNet(model_file='best_policy.pth') if os.path.exists(
            'best_policy.pth') else None
        self.board = Board()
        self.game = Game(self.board)
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.learn_rate = 1e-3
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']  # 训练的batch大小
        self.epochs = CONFIG['epochs']  # 每次更新的train_step数量
        self.kl_targ = CONFIG['kl_targ']  # kl散度控制
        self.check_freq = 100  # 保存模型的频率
        self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()
        self.buffer_size = maxlen=CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('已加载上次最终模型')
            except:
                # 从零开始训练
                print('模型路径不存在，从零开始训练')
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()


    def policy_evaluate(self, n_games=50):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        if self.best_policy_net is None:
            # 第一次訓練，還沒有 best_policy
            print("尚未有 best_policy，跳過對戰評估")
            return 1.0

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=200)

        best_mcts_player = MCTSPlayer(self.best_policy_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=200)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            # 輪流先手
            if i % 2 == 0:
                winner = self.game.start_play(current_mcts_player, best_mcts_player,
                                              start_player=1, is_shown=0)
                if winner == 1:
                    win_cnt["current"] += 1
                elif winner == 2:
                    win_cnt["best"] += 1
                else:
                    win_cnt["tie"] += 1
            else:
                winner = self.game.start_play(best_mcts_player, current_mcts_player,
                                              start_player=1, is_shown=0)
                if winner == 1:
                    win_cnt["best"] += 1
                elif winner == 2:
                    win_cnt["current"] += 1
                else:
                    win_cnt["tie"] += 1

        win_ratio = (win_cnt["current"] + 0.5 * win_cnt["tie"]) / n_games
        print(f"[模型對戰] new vs best → win: {win_cnt['current']}, lose: {win_cnt['best']}, tie: {win_cnt['tie']}")
        print(f"勝率（對best）: {win_ratio:.2f}")
        return win_ratio

    def policy_updata(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten(),new_v.flatten())
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.9f},"
               "explained_var_new:{:.9f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        """開始訓練"""
        try:
            for i in range(self.game_batch_num):
                # 載入數據
                if not CONFIG['use_redis']:
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = data_file['data_buffer']
                                self.iters = data_file['iters']
                                del data_file
                            print('已載入本地資料')
                            break
                        except:
                            print("等待本地數據...")
                            time.sleep(30)
                else:
                    while True:
                        try:
                            l = len(self.data_buffer)
                            data = my_redis.get_list_range(
                                self.redis_cli, 'train_data_buffer', l if l == 0 else l - 1, -1
                            )
                            self.data_buffer.extend(data)
                            self.iters = self.redis_cli.get('iters')
                            if self.redis_cli.llen('train_data_buffer') > self.buffer_size:
                                self.redis_cli.lpop('train_data_buffer', self.buffer_size // 10)
                            print("已載入 Redis 數據")
                            break
                        except:
                            print("等待 Redis 數據...")
                            time.sleep(5)

                print(f"訓練批次 {i + 1}，累積樣本數：{len(self.data_buffer)}")

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_updata()

                    # 每 train_update_interval 時間再做下一輪
                    time.sleep(CONFIG['train_update_interval'])

                    if (i + 1) % self.check_freq == 0:
                        win_ratio = self.policy_evaluate()
                        print(f"第 {i + 1} 批自對弈訓練，勝率：{win_ratio:.3f}")

                        # 條件：勝率高於 55%，且是歷史新高
                        if win_ratio > 0.55 :
                            print("🎯 新最佳策略發現！勝率 {:.2f}%".format(win_ratio * 100))
                            self.best_win_ratio = win_ratio
                            self.policy_value_net.save_model('./current_policy.pth')
                            self.policy_value_net.save_model('./best_policy.pth')
                        else:
                            print("勝率不足 55% 或非最佳，跳過保存 best_policy")

                        # 無論勝率如何，每批次保留 checkpoint
                        self.policy_value_net.save_model(f'models/current_policy_batch{i + 1}.pth')

        except KeyboardInterrupt:
            print('\n⛔️ 手動終止訓練')

if CONFIG['use_frame'] == 'paddle':
    training_pipeline = TrainPipeline(init_model='current_policy.model')
    training_pipeline.run()
elif CONFIG['use_frame'] == 'pytorch':
    training_pipeline = TrainPipeline(init_model='current_policy.pth')
    training_pipeline.run()
else:
    print('暂不支持您选择的框架')
    print('训练结束')
