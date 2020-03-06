import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import Config

from pathlib import Path
import warnings

cfg = Config.load()
warnings.filterwarnings("error")

def get_data_file_path_list():
    paths = [str(x) for x in Path(cfg.path_data).glob("**/*.txt")]
    if len(paths) == 0:
        warnings.warn("no text data, check ./data")
    return paths

def get_data():
    paths = get_data_file_path_list()
    dialog_list = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            dialogs = f.readlines()
            dialog = []
            for turn in dialogs:
                if len(turn) == 1:  # len(enter) 로 대화 구분 나누기
                    dialog_list.append(dialog)
                    dialog = []
                    continue
                dialog.append(turn[:-1])  # /n 제거
    return dialog_list
