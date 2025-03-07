import os
import glob
import pandas as pd
import numpy as np

folder_path = "odds_clean"

def combine_odds(folder_path, save_path):
    pattern = os.path.join(folder_path, "**", "*")
    df = pd.DataFrame()
    for i, file_path in enumerate(glob.glob(pattern, recursive=True)):
        if os.path.isfile(file_path):
            df_new = pd.read_csv(file_path)
            df = pd.concat([df, df_new], join='outer', ignore_index=True)
    df.to_csv(save_path, index=False)
    print(df)