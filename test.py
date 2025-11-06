from game import move_id2move_action, move_action2move_id
'''import os

import pandas as pd

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
    {"Opponent": "Random", "Wins": 100, "Losses": 0, "Draws": 0},
    {"Opponent": "Greedy", "Wins": 76, "Losses": 23, "Draws": 1},
    {"Opponent": "GPT", "Wins": 94, "Losses": 5, "Draws": 1},
    {"Opponent": "DarkCraftLite", "Wins": 59, "Losses": 41, "Draws": 0},
]

# 合併新資料
df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

# 存回 CSV
df.to_csv(file_path, index=False)'''
print(move_id2move_action[257])
