from game import Board
from players import RandomPlayer, GreedyPlayer, battle, ChatGPTPlayer, battle_summary
from mcts import MCTSPlayer
import time
from config import CONFIG
from mcts_pure import MCTS_Pure
from pytorch_net import PolicyValueNet
board = Board()
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
playout_num = 1000
# 定義不同策略
player_RL = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=1,
                                 n_playout=playout_num,
                                 is_selfplay=0)
random_agent = RandomPlayer()
greedy_agent = GreedyPlayer()
gpt_player = ChatGPTPlayer()
opponents = {
    "Random": RandomPlayer(),
    "Greedy": GreedyPlayer(),
    "GPT": ChatGPTPlayer()
}
# 跑 100 場 RL vs Random

results = battle_summary(player_RL, opponents, board, playouts=400, n_games=100)
print(results)

'''res= battle(greedy_agent, gpt_player, board, playout_num, n_games=100, is_shown=False, plot_interval=10)
print(res)'''


# 跑 100 場 RL vs Greedy
#res = battle(player_RL, greedy_agent, board, playout_num, n_games=100, is_shown=False, plot_interval=10)
#print(res)

