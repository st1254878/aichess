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

# 根據 no_dark_mode 設定不同檔案位置
if CONFIG.get('no_dark_mode', True):
    TRAIN_DATA_PATH = CONFIG.get('train_data_buffer_path_no_dark', 'train_data_buffer_no_dark.pth')
    BEST_POLICY_PATH = 'best_policy_no_dark.pth'
    CURRENT_POLICY_PATH = 'current_policy_no_dark.pth'
    MODELS_DIR = 'models_no_dark'
    REDIS_KEY = 'train_data_buffer_no_dark'
    REDIS_ITERS_KEY = 'iters_no_dark'
else:
    TRAIN_DATA_PATH = CONFIG.get('train_data_buffer_path', 'train_data_buffer.pth')
    BEST_POLICY_PATH = 'best_policy.pth'
    CURRENT_POLICY_PATH = 'current_policy.pth'
    MODELS_DIR = 'models'
    REDIS_KEY = 'train_data_buffer'
    REDIS_ITERS_KEY = 'iters'

os.makedirs(MODELS_DIR, exist_ok=True)

class TrainPipeline:
    def __init__(self, init_model=None):
        # 若磁碟已有 best model，嘗試載入（可能為 None）
        self.best_policy_net = None
        if os.path.exists(BEST_POLICY_PATH):
            try:
                self.best_policy_net = PolicyValueNet(model_file=BEST_POLICY_PATH)
                print(f"載入 best policy: {BEST_POLICY_PATH}")
            except Exception as e:
                print(f"載入 best policy 失敗: {e}")

        self.board = Board()
        self.game = Game(self.board)
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.learn_rate = 1e-3
        self.lr_multiplier = 1
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ']
        self.check_freq = 100
        self.game_batch_num = CONFIG['game_batch_num']
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 50  # 用於 pure MCTS baseline
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)

        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('已加载上次最终模型')
            except Exception as e:
                print('模型路径不存在或载入失败，从零开始训练', e)
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()

    def _fight(self, player1, player2, n_games=30):
        """雙方對戰，返回勝率、詳細勝負"""
        win_cnt = defaultdict(int)
        show = 0
        for i in range(n_games):
            if n_games == 10:
                show = 1
            if i % 2 == 0:
                winner = self.game.start_play(player1, player2, start_player=1, is_shown=show)
                if winner == 1:
                    win_cnt["p1"] += 1
                elif winner == 2:
                    win_cnt["p2"] += 1
                else:
                    win_cnt["tie"] += 1
            else:
                winner = self.game.start_play(player2, player1, start_player=1, is_shown=show)
                if winner == 1:
                    win_cnt["p2"] += 1
                    print()
                elif winner == 2:
                    win_cnt["p1"] += 1
                else:
                    win_cnt["tie"] += 1
            print("第",i+1,"場結束",sep="")
        win_ratio = (win_cnt["p1"] + 0.5 * win_cnt["tie"]) / n_games
        return win_ratio, win_cnt

    def multi_evaluate(self, n_games=30):
        """同時測試 current vs best, current vs pureMCTS, best vs pureMCTS"""
        # 印出檔案時間
        for path, name in [(CURRENT_POLICY_PATH, "current_policy"), (BEST_POLICY_PATH, "best_policy")]:
            if os.path.exists(path):
                print(f"[檔案時間] {name} 最後修改: {time.ctime(os.path.getmtime(path))}")

        current_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                    c_puct=self.c_puct,
                                    n_playout=100)

        # baseline: pure MCTS
        pure_player = MCTS_Pure(self.pure_mcts_playout_num)

        # best_policy
        best_player = None
        if os.path.exists(BEST_POLICY_PATH):
            try:
                best_net = PolicyValueNet(model_file=BEST_POLICY_PATH)
                best_player = MCTSPlayer(best_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=100)
            except Exception as e:
                print("[EVAL] 載入 best_policy 失敗:", e)

        results = {}

        # 1. current vs best
        if best_player:
            print("存在best_policy 開始evaluate",BEST_POLICY_PATH)
            r, cnt = self._fight(current_player, best_player, n_games=n_games)
            results["current_vs_best"] = (r, cnt)
            print(f"[EVAL] current vs best → {cnt}, 勝率={r:.3f}")
            results["current_vs_pure"] = (0, 0)
        else:
            "目前不存在best_policy"
            r, cnt = self._fight(current_player, pure_player, n_games=10)
            results["current_vs_pure"] = (r, cnt)
            print(f"[EVAL] current vs pureMCTS → {cnt}, 勝率={r:.3f}")
        # 2. current vs pureMCTS


        return results

    def policy_evaluate(self, n_games=30):
        """
        Evaluate current policy against best policy if exists,
        otherwise against a pure MCTS baseline.
        返回胜率 (float)
        """
        # 重新嘗試載入 disk 上的 best model（若被外部程序更新）
        if os.path.exists(BEST_POLICY_PATH):
            try:
                self.best_policy_net = PolicyValueNet(model_file=BEST_POLICY_PATH)
                print(f"[EVAL] 使用已存在的 best model: {BEST_POLICY_PATH}")
            except Exception as e:
                print(f"[EVAL] 載入 best model 失敗，改用 pure MCTS baseline: {e}")
                self.best_policy_net = None
        else:
            print("[EVAL] 無 best model 檔案，改用 pure MCTS baseline")

        # 構造對手
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=max(50, self.n_playout // 4))  # 評估可以用較少 playout
        if self.best_policy_net is not None:
            best_mcts_player = MCTSPlayer(self.best_policy_net.policy_value_fn,
                                          c_puct=self.c_puct,
                                          n_playout=max(50, self.n_playout // 4))
            baseline_is_pure_mcts = False
            print("[EVAL] baseline = best_policy_net")
        else:
            # 用 pure MCTS 作 baseline（不使用神經網路）
            best_mcts_player = MCTS_Pure(self.pure_mcts_playout_num)
            baseline_is_pure_mcts = True
            print(f"[EVAL] baseline = pure MCTS ({self.pure_mcts_playout_num} playouts)")

        win_cnt = defaultdict(int)
        for i in range(n_games):
            # 交換先手/後手
            if i % 2 == 0:
                winner = self.game.start_play(current_mcts_player, best_mcts_player, start_player=1, is_shown=0)
                if winner == 1:
                    win_cnt["current"] += 1
                elif winner == 2:
                    win_cnt["best"] += 1
                else:
                    win_cnt["tie"] += 1
            else:
                winner = self.game.start_play(best_mcts_player, current_mcts_player, start_player=1, is_shown=0)
                # 注意：當 baseline 為 pure MCTS，best_mcts_player 並非 "best" 但計數方式一致
                if winner == 1:
                    # winner 1 對於傳入的 (best, current) 代表 best 贏
                    win_cnt["best"] += 1
                elif winner == 2:
                    win_cnt["current"] += 1
                else:
                    win_cnt["tie"] += 1

        win_ratio = (win_cnt["current"] + 0.5 * win_cnt["tie"]) / n_games
        print(f"[模型對戰] new vs baseline → win: {win_cnt['current']}, lose: {win_cnt['best']}, tie: {win_cnt['tie']}")
        print(f"[EVAL] 勝率（current vs baseline）: {win_ratio:.3f}")
        return win_ratio

    def policy_updata(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = np.array([data[0] for data in mini_batch], dtype='float32')
        mcts_probs_batch = np.array([data[1] for data in mini_batch], dtype='float32')
        winner_batch = np.array([data[2] for data in mini_batch], dtype='float32')

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = 0.0
        loss = None
        entropy = None
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                             self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 保險：避免除以零
        explained_var_old = 1.0
        explained_var_new = 1.0
        try:
            explained_var_old = 1 - np.var(winner_batch - old_v.flatten()) / np.var(winner_batch)
            explained_var_new = 1 - np.var(winner_batch - new_v.flatten()) / np.var(winner_batch)
        except:
            pass

        print(f"kl:{kl:.5f}, lr_multiplier:{self.lr_multiplier:.3f}, loss:{loss}, entropy:{entropy}, "
              f"explained_var_old:{explained_var_old:.9f}, explained_var_new:{explained_var_new:.9f}")
        return loss, entropy

    def run(self):
        try:
            for i in range(self.game_batch_num):
                # 載入資料（根據設定的 TRAIN_DATA_PATH）
                if not CONFIG['use_redis']:
                    while True:
                        try:
                            with open(TRAIN_DATA_PATH, 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = data_file['data_buffer']
                                self.iters = data_file['iters']
                            print(f'已載入本地資料 ({TRAIN_DATA_PATH})')
                            break
                        except Exception as e:
                            print("等待本地數據...", e)
                            time.sleep(30)
                else:
                    while True:
                        try:
                            l = len(self.data_buffer)
                            data = my_redis.get_list_range(self.redis_cli, REDIS_KEY, l if l == 0 else l - 1, -1)
                            self.data_buffer.extend(data)
                            self.iters = self.redis_cli.get(REDIS_ITERS_KEY)
                            if self.redis_cli.llen(REDIS_KEY) > self.buffer_size:
                                self.redis_cli.lpop(REDIS_KEY, self.buffer_size // 10)
                            print(f"已載入 Redis 數據 ({REDIS_KEY})")
                            break
                        except Exception as e:
                            print("等待 Redis 數據...", e)
                            time.sleep(5)

                print(f"訓練批次 {i + 1}，累積樣本數：{len(self.data_buffer)}")
                # 等待資料足夠
                while len(self.data_buffer) <= self.batch_size:
                    print("⚠️ 資料量不足，等待更多資料再訓練...")
                    time.sleep(600)

                # 執行一次更新
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_updata()
                    time.sleep(CONFIG['train_update_interval'])

                    # 定期評估
                    if (i + 1) % self.check_freq == 0:
                        # 重新載入 best model（如果檔案存在），避免 state mismatch
                        if os.path.exists(BEST_POLICY_PATH):
                            try:
                                self.best_policy_net = PolicyValueNet(model_file=BEST_POLICY_PATH)
                            except Exception as e:
                                print("重新載入 best model 失敗:", e)
                                self.best_policy_net = None

                        results = self.multi_evaluate()
                        # 取 current vs best 的勝率來決定是否更新 best_policy
                        win_ratio = results.get("current_vs_best", results["current_vs_pure"])[0]
                        print(f"第 {i + 1} 批自對弈訓練，勝率：{win_ratio:.3f}")

                        # 儲存條件：勝率要超過閾值且比歷史最佳好
                        if win_ratio > 0.55 and win_ratio > self.best_win_ratio:
                            print(f"🎯 新最佳策略發現！勝率 {win_ratio * 100:.2f}% (超過歷史最佳 {self.best_win_ratio:.3f})")
                            self.best_win_ratio = win_ratio
                            self.policy_value_net.save_model(CURRENT_POLICY_PATH)
                            self.policy_value_net.save_model(BEST_POLICY_PATH)
                        else:
                            print("勝率不足 55% 或沒有超越歷史最佳，跳過保存 best_policy")

                        # checkpoint
                        self.policy_value_net.save_model(f'{MODELS_DIR}/current_policy_batch{i + 1}.pth')
        except KeyboardInterrupt:
            print('\n⛔️ 手動終止訓練')


if CONFIG['use_frame'] == 'paddle':
    training_pipeline = TrainPipeline(init_model=CURRENT_POLICY_PATH)
    training_pipeline.run()
elif CONFIG['use_frame'] == 'pytorch':
    training_pipeline = TrainPipeline(init_model=CURRENT_POLICY_PATH)
    training_pipeline.run()
else:
    print('暂不支持您选择的框架')
    print('训练结束')