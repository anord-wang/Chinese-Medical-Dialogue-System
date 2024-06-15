# 数据集统计
# 文件个数

# 训练集测试集分布（个数、比例）

# 文本长度list（分布图）

# 文本长度超过510的个数、比例

# 文本最大长度

# 文本平均长度

# 标签类别个数

# 标签分布情况（图）
from time import strptime
import csv
import numpy
from transformers import BertTokenizer
from config import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from os.path import join
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib.pyplot as plt
import pandas as pd


class IMdbDataset_sta(Dataset):

    def __init__(self, configs, data_dir=DATA_DIR):
        self.data_dir = data_dir

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def read_data_file(self):
        data_file_train = self.train_file
        data_file_test = self.test_file
        temp_data_path = [os.path.join(data_file_train, 'pos'), os.path.join(data_file_train, 'neg'),
                          os.path.join(data_file_test, 'pos'), os.path.join(data_file_test, 'neg')]
        self.total_file_path = []
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)
        texts = []
        length = []
        labels = []
        for item in self.total_file_path:
            with open(item, 'r', encoding='utf-8') as f:
                raw = f.readlines()
                f.close()
            text = raw[0]
            texts.append(text)
            text.strip()
            text = text.split(' ')
            length.append(len(text))
            label_str = item.split('/')[-2]
            label = 0 if label_str == 'neg' else 1
            labels.append(label)
        print(np.size(labels))
        print(np.size(length))

        tokenizer = self.bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        length_1 = []
        input_ids = tokenizer['input_ids']
        print(input_ids)
        for i in range(len(input_ids)):
            now = input_ids[i]
            sum = 0
            for j in range(1024):
                if now[j] != 0:
                    sum = sum + 1
                else:
                    length_1.append(sum)
                    break
            # legth_1 = len(input_ids[i])
            # length_1.append(legth_1)
        print(length)
        print('word part')
        sum1 = 0
        sum2 = 0
        sum3 = max(length)
        for i in range(len(length)):
            sum1 = sum1 + length[i]
            if length[i] > 510:
                sum2 = sum2 + 1

        print('Average length', sum1 / len(length))
        print('Average length more than 500', sum2 / len(length))
        print('more than 500', sum2)
        print('longest', sum3)
        print('total number', len(length))

        print('token part')
        sum1_1 = 0
        sum2_1 = 0
        sum3_1 = max(length_1)
        for i in range(len(length_1)):
            sum1_1 = sum1_1 + length_1[i]
            if length_1[i] > 510:
                sum2_1 = sum2_1 + 1

        print('Average length', sum1_1 / len(length_1))
        print('Average length more than 500', sum2_1 / len(length_1))
        print('more than 500', sum2_1)
        print('longest', sum3_1)
        print('total number', len(length_1))

        # plt.hist(length, 100, cumulative=True)
        # plt.hist(length, 100)
        plt.hist(length_1, 100, alpha=0.5, label='IMDB')
        # plt.show()


from sklearn.datasets import fetch_20newsgroups


class news_20_Dataset_sta(Dataset):

    def __init__(self, configs, data_dir=DATA_DIR):
        self.data_dir = data_dir

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

    def read_data_file(self):
        newsgroups_all = fetch_20newsgroups(subset='all')
        text_file = newsgroups_all['data']
        # labels = newsgroups_all.target
        texts = []
        length = []
        labels = []
        for i in range(len(text_file)):
            line = text_file[i]
            # print(line)
            texts.append(line)
            line.strip()
            line = line.split(' ')
            length.append(len(line))

        print(np.size(labels))
        print(np.size(length))

        tokenizer = self.bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        length_1 = []
        input_ids = tokenizer['input_ids']
        print(len(texts))
        print(len(input_ids))
        for i in range(len(input_ids)):
            now = input_ids[i]
            sum = 0
            for j in range(2048):
                if now[j] != 0:
                    sum = sum + 1
                else:
                    length_1.append(sum)
                    break
            # legth_1 = len(input_ids[i])
            # length_1.append(legth_1)
        print(length)
        print('word part')
        sum1 = 0
        sum2 = 0
        sum3 = max(length)
        for i in range(len(length)):
            sum1 = sum1 + length[i]
            if length[i] > 510:
                sum2 = sum2 + 1

        print('Average length', sum1 / len(length))
        print('Average length more than 500', sum2 / len(length))
        print('more than 500', sum2)
        print('longest', sum3)
        print('total number', len(length))

        print('token part')
        sum1_1 = 0
        sum2_1 = 0
        sum3_1 = max(length_1)
        for i in range(len(length_1)):
            sum1_1 = sum1_1 + length_1[i]
            if length_1[i] > 510:
                sum2_1 = sum2_1 + 1

        print('Average length', sum1_1 / len(length_1))
        print('Average length more than 500', sum2_1 / len(length_1))
        print('more than 500', sum2_1)
        print('longest', sum3_1)
        print('total number', len(length_1))

        # plt.hist(length, 100, cumulative=True)
        # plt.hist(length, 100)
        plt.hist(length_1, 100, alpha=0.5, label='20 Newsgroup')
        # plt.show()


class AG_news_Dataset_sta(Dataset):

    def __init__(self, configs, data_dir=DATA_DIR_AG_NEWS):
        self.data_dir = data_dir
        self.data_dir = data_dir

        self.train_file = join(data_dir, TRAIN_FILE_AG_NEWS)
        self.test_file = join(data_dir, TEST_FILE_AG_NEWS)

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

    def read_data_file(self):
        data_file_train = self.train_file
        data_file_test = self.test_file
        texts = []
        length = []
        labels = []

        ag_data_train = pd.read_csv(data_file_train, header=None, low_memory=False)  # 防止弹出警告
        ag_df_train = pd.DataFrame(ag_data_train)
        self.text_len_train = ag_df_train.shape[0]
        print(ag_df_train.shape)

        for i in range(self.text_len_train):
            raw = str(ag_df_train.iloc[i, 2])
            raw = raw.replace('\n', '')
            texts.append(raw)
            raw.strip()
            raw = raw.split(' ')
            length.append(len(raw))

        ag_data_test = pd.read_csv(data_file_test, header=None, low_memory=False)  # 防止弹出警告
        ag_df_test = pd.DataFrame(ag_data_test)
        self.text_len_test = ag_df_test.shape[0]
        print(ag_df_test.shape)

        for i in range(self.text_len_test):
            raw = str(ag_df_test.iloc[i, 2])
            raw = raw.replace('\n', '')
            texts.append(raw)
            raw.strip()
            raw = raw.split(' ')
            length.append(len(raw))

        print(np.size(labels))
        print(np.size(length))

        # tokenizer = self.bert_tokenizer(
        #     texts,
        #     padding=True,
        #     truncation=True,
        #     max_length=2048,
        #     return_tensors='pt'
        # )
        # length_1 = []
        # input_ids = tokenizer['input_ids']
        # print(len(texts))
        # print(len(input_ids))
        # for i in range(len(input_ids)):
        #     now = input_ids[i]
        #     sum = 0
        #     for j in range(2048):
        #         if now[j] != 0:
        #             sum = sum + 1
        #         else:
        #             length_1.append(sum)
        #             break
        #     # legth_1 = len(input_ids[i])
        #     # length_1.append(legth_1)
        print(length)
        print('word part')
        sum1 = 0
        sum2 = 0
        sum3 = max(length)
        for i in range(len(length)):
            sum1 = sum1 + length[i]
            if length[i] > 510:
                sum2 = sum2 + 1

        print('Average length', sum1 / len(length))
        print('Average length more than 500', sum2 / len(length))
        print('more than 500', sum2)
        print('longest', sum3)
        print('total number', len(length))

        # print('token part')
        # sum1_1 = 0
        # sum2_1 = 0
        # sum3_1 = max(length_1)
        # for i in range(len(length_1)):
        #     sum1_1 = sum1_1 + length_1[i]
        #     if length_1[i] > 510:
        #         sum2_1 = sum2_1 + 1
        #
        # print('Average length', sum1_1 / len(length_1))
        # print('Average length more than 500', sum2_1 / len(length_1))
        # print('more than 500', sum2_1)
        # print('longest', sum3_1)
        # print('total number', len(length_1))

        # plt.hist(length, 100, cumulative=True)
        plt.hist(length, 100, alpha=0.5, label='AG News')
        # plt.hist(length_1, 100, alpha=0.5)
        # plt.show()


class WOS_11967_Dataset_sta(Dataset):

    def __init__(self, configs, data_dir=DATA_DIR_WOS):
        self.data_dir = data_dir
        self.data_dir = data_dir

        self.final_file = join(data_dir, FINAL_FILE_WOS)

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

    def read_data_file(self):
        data_file_final = self.final_file
        texts = []
        length = []
        labels = []

        ag_data_final = pd.read_csv(data_file_final, header=None, low_memory=False)  # 防止弹出警告
        ag_df_final = pd.DataFrame(ag_data_final)
        self.text_len_final = ag_df_final.shape[0]
        print(ag_df_final.shape)

        for i in range(self.text_len_final):
            # raw = str(ag_df_train.iloc[i, 2])
            raw = str(ag_df_final.iloc[i, 0])
            # raw = raw.replace('\n', '')
            texts.append(raw)
            raw.strip()
            raw = raw.split(' ')
            length.append(len(raw))

        print(np.size(labels))
        print(np.size(length))

        tokenizer = self.bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        length_1 = []
        input_ids = tokenizer['input_ids']
        print(len(texts))
        print(len(input_ids))
        for i in range(len(input_ids)):
            now = input_ids[i]
            sum = 0
            for j in range(1119):
                if now[j] != 0:
                    sum = sum + 1
                else:
                    length_1.append(sum)
                    break
            # legth_1 = len(input_ids[i])
            # length_1.append(legth_1)
        print(length)
        print('word part')
        sum1 = 0
        sum2 = 0
        sum3 = max(length)
        for i in range(len(length)):
            sum1 = sum1 + length[i]
            if length[i] > 510:
                sum2 = sum2 + 1

        print('Average length', sum1 / len(length))
        print('Average length more than 500', sum2 / len(length))
        print('more than 500', sum2)
        print('longest', sum3)
        print('total number', len(length))

        print('token part')
        sum1_1 = 0
        sum2_1 = 0
        sum3_1 = max(length_1)
        for i in range(len(length_1)):
            sum1_1 = sum1_1 + length_1[i]
            if length_1[i] > 510:
                sum2_1 = sum2_1 + 1

        print('Average length', sum1_1 / len(length_1))
        print('Average length more than 500', sum2_1 / len(length_1))
        print('more than 500', sum2_1)
        print('longest', sum3_1)
        print('total number', len(length_1))

        # plt.hist(length, 100, cumulative=True)
        # plt.hist(length, 100)
        plt.hist(length_1, 100, alpha=0.5, label='WOS 11967')
        # plt.show()


class native_speaker_Dataset_sta(Dataset):

    def __init__(self, configs, data_dir=DATA_DIR_NS):
        self.data_dir = data_dir
        self.data_dir = data_dir

        # self.final_file = join(data_dir, FINAL_FILE_NS)
        self.train_file = join(data_dir, TRAIN_FILE_NS)
        self.test_file = join(data_dir, TEST_FILE_NS)

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

    def read_data_file(self):
        # data_file_final = self.final_file
        data_file_train = self.train_file
        data_file_test = self.test_file
        texts = []
        length = []
        labels = []

        # ag_data_final = pd.read_csv(data_file_final, header=None, low_memory=False)  # 防止弹出警告
        NS_data_train = pd.read_csv(data_file_train, header=None, low_memory=False)  # 防止弹出警告
        NS_data_test = pd.read_csv(data_file_test, header=None, low_memory=False)  # 防止弹出警告

        # ag_df_final = pd.DataFrame(ag_data_final)
        NS_df_train = pd.DataFrame(NS_data_train)
        NS_df_test = pd.DataFrame(NS_data_test)
        NS_df_final = pd.concat([NS_df_train, NS_df_test])
        self.text_len_final = NS_df_final.shape[0]
        print(NS_df_final.shape)

        for i in range(self.text_len_final):
            # raw = str(ag_df_train.iloc[i, 2])
            raw = str(NS_df_final.iloc[i, 0])
            # raw = raw.replace('\n', '')
            texts.append(raw)
            raw.strip()
            raw = raw.split(' ')
            length.append(len(raw))

        print(np.size(labels))
        print(np.size(length))

        tokenizer = self.bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )
        length_1 = []
        input_ids = tokenizer['input_ids']
        print(len(texts))
        print(len(input_ids))
        for i in range(len(input_ids)):
            now = input_ids[i]
            sum = 0
            for j in range(3372):
                if now[j] != 0:
                    sum = sum + 1
                else:
                    length_1.append(sum)
                    break
            # legth_1 = len(input_ids[i])
            # length_1.append(legth_1)
        print(length)
        print('word part')
        sum1 = 0
        sum2 = 0
        sum3 = max(length)
        for i in range(len(length)):
            sum1 = sum1 + length[i]
            if length[i] > 510:
                sum2 = sum2 + 1

        print('Average length', sum1 / len(length))
        print('Average length more than 500', sum2 / len(length))
        print('more than 500', sum2)
        print('longest', sum3)
        print('total number', len(length))

        print('token part')
        sum1_1 = 0
        sum2_1 = 0
        sum3_1 = max(length_1)
        for i in range(len(length_1)):
            sum1_1 = sum1_1 + length_1[i]
            if length_1[i] > 510:
                sum2_1 = sum2_1 + 1

        print('Average length', sum1_1 / len(length_1))
        print('Average length more than 500', sum2_1 / len(length_1))
        print('more than 500', sum2_1)
        print('longest', sum3_1)
        print('total number', len(length_1))

        # plt.hist(length, 100, cumulative=True)
        # plt.hist(length, 100)
        plt.hist(length_1, 100, alpha=0.5, label='Native Speaker')
        # 画图
        plt.xlim(-50, 1000)
        plt.ylim(0, 5000)
        plt.xlabel('Tokens per Document')
        plt.ylabel('Number of Documents')
        plt.axvline(510, color='k', linestyle='--')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    configs = Config()
    IMdbDataset_sta(configs).read_data_file()
    news_20_Dataset_sta(configs).read_data_file()
    AG_news_Dataset_sta(configs).read_data_file()
    WOS_11967_Dataset_sta(configs).read_data_file()
    native_speaker_Dataset_sta(configs).read_data_file()
