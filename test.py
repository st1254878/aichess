from game import move_id2move_action, move_action2move_id
import os

import pandas as pd
'''
# 讀取原本的 CSV
file_path = "new_new_battle_summary.csv"

# 檢查檔案是否存在
if os.path.exists(file_path):
    # 若存在則讀取
    df = pd.read_csv(file_path)
else:
    # 若不存在則建立一個新的 DataFrame
    df = pd.DataFrame(columns=["Model", "Opponent", "Wins", "Losses", "Draws"])

# 新增多筆資料
new_data = [
    {"Model": "new", "Opponent": "Random", "Wins": 100, "Losses": 0, "Draws": 0},
    {"Model": "new", "Opponent": "Greedy", "Wins": 76, "Losses": 23, "Draws": 1},
    {"Model": "new", "Opponent": "GPT", "Wins": 94, "Losses": 5, "Draws": 1},
    {"Model": "new", "Opponent": "DarkCraftLite", "Wins": 59, "Losses": 41, "Draws": 0},
    {"Model": "old", "Opponent": "Random", "Wins": 97, "Losses": 3, "Draws": 0},
    {"Model": "old", "Opponent": "Greedy", "Wins": 27, "Losses": 66, "Draws": 1},
    {"Model": "old", "Opponent": "GPT", "Wins": 60, "Losses": 40, "Draws": 1},
    {"Model": "old", "Opponent": "DarkCraftLite", "Wins": 2, "Losses": 98, "Draws": 0},
]

# 合併新資料
df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

# 存回 CSV
df.to_csv(file_path, index=False)
'''
# 讀取原本的 CSV
file_path = "new_battle_summary.csv"

# 檢查檔案是否存在
if os.path.exists(file_path):
    # 若存在則讀取
    df = pd.read_csv(file_path)
else:
    # 若不存在則建立一個新的 DataFrame
    df = pd.DataFrame(columns=["Opponent", "Wins", "Losses", "Draws"])

# 新增多筆資料
new_data = [
    {"Model": "new", "Opponent": "Random", "Wins": 100, "Losses": 0, "Draws": 0},
    {"Model": "new", "Opponent": "Greedy", "Wins": 76, "Losses": 23, "Draws": 1},
    {"Model": "new", "Opponent": "GPT", "Wins": 94, "Losses": 5, "Draws": 1},
]

# 合併新資料
df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

# 存回 CSV
df.to_csv(file_path, index=False)