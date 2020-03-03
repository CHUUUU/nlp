import util as u
import vocab.vocab as v
import config.config as c
from PIL import Image
from nltk.tokenize import RegexpTokenizer
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as transforms
import pandas as pd
import random
import torch

class text_img_pair_dataset(data.Dataset):
    def __init__(self):
        # filenames is the key which connecting image and text for pair
        self.train_file_names = u.pickle_load(c.train_filename_path)
        self.test_file_names = u.pickle_load(c.test_filename_path)

        self.train_file_names = sorted(self.train_file_names)
        self.test_file_names = sorted(self.test_file_names)


        self.train_class_id = c.train_class_info
        self.test_class_id = c.test_class_info
        # self.train_class_id = u.pickle_load2(c.train_class_id_txt_path)
        # self.test_class_id = u.pickle_load2(c.test_class_id_txt_path)

        # text
        self.tokenizer = RegexpTokenizer(r'\w+')

        if not os.path.isfile(c.vocab_path):
            train_text = self.__load_all_text(self.train_file_names)
            test_text = self.__load_all_text(self.test_file_names)
            all_text = train_text + test_text

            self.__word_count_statistics(all_text)

            vocab = v.vocab()
            vocab.create(all_text)

        self.word_2_index = u.json_load(c.vocab_path)
        self.index_2_word = u.json_load(c.index_2_word_path)
        # self.index_2_word = {v: k for k, v in self.word_2_index.items()}
        self.vocab_size = len(self.word_2_index)
        print("vocab_size : ", self.vocab_size)

        # image
        self.base_img_size = [64, 128, 256]
        label_image_size = 256
        rate = 76 / 64
        self.image_transform = transforms.Compose([
            transforms.Resize(int(label_image_size * rate)),
            transforms.RandomCrop(label_image_size),
            transforms.RandomHorizontalFlip()
        ])

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.filenames_bbox = self.__load_bbox()

    def __load_single_text(self, f_name):
        all_text = []
        file_path = c.base_text_path + f_name + ".txt"
        text_lines = u.text_load(file_path)

        if text_lines[-1] == "":
            text_lines = text_lines[:-1]  # if 11th line is blank

        if not len(text_lines) == 10:
            print(file_path)
            print(len(text_lines))
            print("1 folder == 10 text samples, but it is not")
            assert False

        for text in text_lines:
            if len(text)==0:
                continue
            text = text.replace("\ufffd\ufffd", " ")
            text = text.lower()
            text = self.tokenizer.tokenize(text)
            text.append(c.EOS_TOKEN)
            all_text.append(text)
        return all_text

    def __load_all_text(self, file_names):
        all_text = []
        for n, f_name in enumerate(file_names):
            text_10 = self.__load_single_text(f_name)
            all_text.extend(text_10)
        return all_text

    def __load_bbox(self):
        bbox_txt = pd.read_csv(c.bbox_txt_path, delim_whitespace=True, header=None).astype(int)
        bbox_image_txt = pd.read_csv(c.bbox_image_txt_path, delim_whitespace=True, header=None)

        # 0 = index, 1 = filename.jpg
        all_img_filenames = bbox_image_txt[1].tolist()

        # set blank json
        filename_bbox = {img_file[:-4]: [] for img_file in all_img_filenames}
        for i in range(0, len(all_img_filenames)):
            bbox = bbox_txt.iloc[i][1:].tolist()  # bbox = [x-left, y-top, width, height]
            key = all_img_filenames[i][:-4]  # filename except .jpg
            filename_bbox[key] = bbox  # fill in blank json with bbox

        return filename_bbox

    def __get_image(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        # actually, now image label size is not 256 x 256, so refer to bbox, crop
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

        # augmentation
        img = self.image_transform(img)

        # 256 -> 64, 128 image label, because stack D need them for real label
        ret = []
        for i in range(len(self.base_img_size)):
            if i < (len(self.base_img_size) - 1):
                re_img = transforms.Resize(self.base_img_size[i])(img)
            else:
                re_img = img

        # normalize
            ret.append(self.norm(re_img))
        return ret

    def __word_count_statistics(self, all_text):
        word_length_list = []
        for text in all_text:
            word_length_list.append(len(text))
        for i in range(max(word_length_list)):
            print(i + 1, " : ", word_length_list.count(i + 1))

    def __PAD(self, text):
        while len(text) < c.max_seq:
            text.append(c.PAD_TOKEN)
        return text

    def __len__(self):
        return len(self.train_file_names)

    def __getitem__(self, index):
        key = self.train_file_names[index]
        cls_id = self.train_class_id[index]

        # 1 picture = 10 text description, choose one description
        selected_index = random.randint(0, 9)
        text_10 = self.__load_single_text(key)
        text = text_10[selected_index]
        text = self.__PAD(text)
        index_text = [self.word_2_index[word] for word in text]

        image_name = c.base_image_path + key + ".jpg"
        bbox = self.filenames_bbox[key]
        img = self.__get_image(image_name, bbox)
        # print(np.asarray(img[0]).shape)  # (3, 64, 64)
        # print(np.asarray(img[1]).shape)  # (3, 128, 128)
        # print(np.asarray(img[2]).shape)  # (3, 256, 256)

        return text, index_text, img, cls_id, key


if __name__ == "__main__":
    a = text_img_pair_dataset()