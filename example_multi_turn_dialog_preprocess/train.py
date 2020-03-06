from preprocess.dialog_dataset import dialog_dataset
from torch.utils.data import DataLoader

import argparse
from config.config import Config

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", default=2, type=int, required=True)
    parser.add_argument("-c", "--config", default="./config/config_default.json", type=str, required=False)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    cfg.batch = args.batch
    return cfg

if __name__ == '__main__':
    cfg = get_config()
    dataset = dialog_dataset(cfg)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=dataset.collate)

    for epoch in range(cfg.epoch):
        data_iter = iter(dataloader)
        step = 0
        while step < len(dataloader):
            step += 1
            train, lm_label, turn_id, session_id = data_iter.next()
            print(train.shape)
            print(lm_label.shape)
            print(turn_id.shape)
            print(session_id.shape)
            break




# print(dataset.index_to_sentence(sample))


