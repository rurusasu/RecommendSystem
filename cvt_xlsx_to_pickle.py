import pickle

import pandas as pd

pickle_name = "sample_merged_full_rating_conv"
base_dir = "/home/user/core/data"

# エクセルファイルの読み込み
# df = pd.read_excel('your_excel_file.xlsx')
data = pd.read_excel(f"{base_dir}/{pickle_name}.xlsx")

# DataFrameをpickle形式で保存
with open(f"{base_dir}/{pickle_name}.pkl", "wb") as f:
    pickle.dump(data, f)

# 保存したDataFrameの読み込み
with open(f"{base_dir}/{pickle_name}.pkl", "rb") as f:
    loaded_df = pickle.load(f)

# 読み込んだDataFrameの表示
print(loaded_df)
