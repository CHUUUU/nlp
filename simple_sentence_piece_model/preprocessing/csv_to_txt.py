import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import config as c
import pandas as pd

def convert_csv_to_txt():
    all_dialogue = []
    with open(c.data_csv_path, newline='') as csvfile:
        df = pd.read_csv(csvfile)
        all_dialogue.extend(df['Q'].values.tolist())
        all_dialogue.extend(df['A'].values.tolist())

    with open(c.data_text_path, 'w') as f:
        for item in all_dialogue:
            f.write("%s\n" % item)

    print("convert txt, done")