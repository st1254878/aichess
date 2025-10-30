import pandas as pd

# 讀取原本的 CSV
df = pd.read_csv("new_battle_summary.csv")

# 新增一行資料
new_row = {"Opponent": "Random", "Wins": 100, "Losses": 0, "Draws": 0}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# 存回檔案
df.to_csv("new_battle_summary.csv", index=False)
print("✅ 已新增 Random 戰績")