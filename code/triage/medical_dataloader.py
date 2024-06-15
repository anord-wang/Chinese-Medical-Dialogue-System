from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
from medical_config import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from os.path import join
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
import torchtext.vocab as vocab
from torchtext.data import get_tokenizer
import pandas as pd
import jieba
from gensim.models import KeyedVectors

from matplotlib import pyplot as plt


def build_train_data(configs):
    train_dataset = Medical_Dataset_label_1_BERT(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_1_lite_BERT(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_1_verylite_BERT(configs, data_type='train')
    # train_dataset = Medical_Dataset_PERT(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_1_RoBERTa(configs, data_type='train')
    # train_dataset = Medical_Dataset_lite_RoBERTa(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_1_NN(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_3_BERT(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_3_NN(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_3_RoBERTa(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_3_electra(configs, data_type='train')
    # train_dataset = Medical_Dataset_label_3_verylite_BERT(configs, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset.get_data(), batch_size=configs.batch_size,
                                               shuffle=True)
    return train_loader


def build_inference_data(configs, data_type):
    dataset = Medical_Dataset_label_1_BERT(configs, data_type)
    # dataset = Medical_Dataset_label_1_lite_BERT(configs, data_type)
    # dataset = Medical_Dataset_label_1_verylite_BERT(configs, data_type)
    # dataset = Medical_Dataset_PERT(configs, data_type)
    # dataset = Medical_Dataset_label_1_RoBERTa(configs, data_type)
    # dataset = Medical_Dataset_label_1_NN(configs, data_type)
    # dataset = Medical_Dataset_label_3_BERT(configs, data_type)
    # dataset = Medical_Dataset_label_3_NN(configs, data_type)
    # dataset = Medical_Dataset_label_3_RoBERTa(configs, data_type)
    # dataset = Medical_Dataset_label_3_electra(configs, data_type)
    # dataset = Medical_Dataset_label_3_verylite_BERT(configs, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset.get_data(), batch_size=configs.batch_size,
                                              shuffle=False)
    return data_loader


def transfer_to_coarse(finegrained_prob):
    corse_prob = []
    for p in finegrained_prob:
        corse_prob.append([p[0] + p[2] + p[4], p[1] + p[3] + p[5]])

    return np.array(corse_prob)


def transfer_to_onehot(finegrained_prob):
    corse_prob = []
    for p in finegrained_prob:
        corse_prob.append([1 - int(p), int(p)])

    return np.array(corse_prob)


def tokenlize(content):
    content = re.sub('<.*?>', ' ', content)

    fileters = ['\/', '\(', '\)', ':', '\.', '\t', '\n', '\x97', '\x96', '#', '$', '%', '&']
    content = re.sub('|'.join(fileters), ' ', content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


class Medical_Dataset_label_1_BERT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_1):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_1)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_1)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        # self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.pretrain_macbert_all)
        # self.bert_tokenizer = BertTokenizer.from_pretrained(configs.pretrain_bert_all_more_test)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        infos, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            infos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

        return data


class Medical_Dataset_label_3_BERT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_3):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_3)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_3)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        # self.bert_tokenizer = BertTokenizer.from_pretrained(configs.macbert_cache_path_Chinese)
        # self.bert_tokenizer = AutoTokenizer.from_pretrained(configs.CirBERTa_path_Chinese)
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.pretrain_bert_all_final)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        infos, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            infos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

        return data


class Medical_Dataset_label_3_electra(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_3):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_3)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_3)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.electra_tokenizer = AutoTokenizer.from_pretrained(configs.electra_cache_path_Chinese)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        infos, labels = self.read_data_file(self.data_type)

        tokenizer = self.electra_tokenizer(
            infos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

        return data


class Medical_Dataset_label_1_lite_BERT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_lite_1):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_lite_1)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_lite_1)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        # self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.pretrain_bert_all_final)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        infos, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            infos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

        return data


class Medical_Dataset_label_1_verylite_BERT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_verylite_1):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_verylite_1)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_verylite_1)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        # self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.pretrain_bert_all_final)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        infos, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            infos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

        return data


class Medical_Dataset_label_3_verylite_BERT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_verylite_3):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_verylite_3)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_verylite_3)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        # self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.pretrain_bert_all_final)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        infos, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            infos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

        return data



class Medical_Dataset_label_1_NN(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_1):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_1)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_1)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs
        self.embedding_size = configs.GloVe_embedding_length

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        global_vectors = vocab.GloVe(name='6B', dim=self.embedding_size, cache=glove_cache_dir)
        texts, labels = self.read_data_file(self.data_type)
        tokenizer = get_tokenizer("basic_english")
        texts = [tokenizer(x) for x in texts]
        texts = [tokens + [""] * (self.max_length - len(tokens)) if len(tokens) < self.max_length else tokens[
                                                                                                       :self.max_length]
                 for tokens in texts]
        X_tensor = torch.zeros(len(texts), self.max_length, self.embedding_size)
        for i, tokens in enumerate(texts):
            X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
            # print(X_tensor[i])
            # print(X_tensor[i].shape)
            # print(X_tensor[i].shape)

        labels = torch.tensor(labels)

        data = TensorDataset(X_tensor, labels)
        print(X_tensor.shape)
        print(labels.shape)

        return data


class Medical_Dataset_label_3_NN(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_3):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_3)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_3)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs
        self.embedding_size = configs.GloVe_embedding_length

        self.max_length = configs.max_length
        self.nn_max_length = configs.nn_max_length
        self.jieba_embedding_size = configs.jieba_embedding_size

    def compute_ngrams(self, word, min_n, max_n):
        # BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word = word
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return list(set(ngrams))

    def wordVec(self, word, wv_from_text, min_n=1, max_n=3):
        '''
        ngrams_single/ngrams_more,主要是为了当出现oov的情况下,最好先不考虑单字词向量
        '''
        # 确认词向量维度
        word_size = wv_from_text.vectors.shape[0]
        # 计算word的ngrams词组
        ngrams = self.compute_ngrams(word, min_n=min_n, max_n=max_n)
        # 如果在词典之中，直接返回词向量
        if word in wv_from_text.key_to_index.keys():
            return wv_from_text[word]
        else:
            # 不在词典的情况下
            word_vec = np.zeros(word_size, dtype=np.float32)
            ngrams_found = 0
            ngrams_single = [ng for ng in ngrams if len(ng) == 1]
            ngrams_more = [ng for ng in ngrams if len(ng) > 1]
            # 先只接受2个单词长度以上的词向量
            for ngram in ngrams_more:
                if ngram in wv_from_text.key_to_index.keys():
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
                    # print(ngram)
            # 如果，没有匹配到，那么最后是考虑单个词向量
            if ngrams_found == 0:
                for ngram in ngrams_single:
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
            if word_vec.any():
                return word_vec / max(1, ngrams_found)
            else:
                raise KeyError('all ngrams for word %s absent from model' % word)

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        # titles = []
        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):
        texts, labels = self.read_data_file(self.data_type)
        # stopwords = [line.strip() for line in
        #              open('/data0/wxy/bert/medical/data/hit_stopwords.txt', encoding='UTF-8').readlines()]
        # tfidf_vec = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")


        tc_file = '/data0/wxy/bert/medical/data/Tencent_AILab_ChineseEmbedding.txt'
        wv_from_text = KeyedVectors.load_word2vec_format(tc_file, binary=False)
        wv_from_text.init_sims(replace=True)


        X_tensor = np.zeros((len(texts), self.nn_max_length, self.jieba_embedding_size), dtype=float)
        stopwords = [line.strip() for line in
                     open('/data0/wxy/bert/medical/data/hit_stopwords.txt', encoding='UTF-8').readlines()]
        stopwords.append(' ')
        for i in range(len(texts)):
            final_seg = []
            seg_list = jieba.cut(texts[i])
            for seg in seg_list:
                if seg not in stopwords:
                    final_seg.append(seg)
            for j in range(min(self.nn_max_length, len(final_seg))):
                X_tensor[i][j] = self.wordVec('you', wv_from_text, min_n=1, max_n=3)


        X_tensor = torch.tensor(X_tensor)
        labels = torch.tensor(labels)

        data = TensorDataset(X_tensor, labels)
        print(X_tensor.shape)
        print(labels.shape)

        return data


# class Medical_Dataset_PERT(Dataset):
#
#     def __init__(self, configs, data_type, data_dir=DATA_DIR):
#         self.data_dir = data_dir
#         self.data_type = data_type
#
#         self.train_file = join(data_dir, TRAIN_FILE)
#         self.valid_file = join(data_dir, VALID_FILE)
#         self.test_file = join(data_dir, TEST_FILE)
#
#         self.batch_size = configs.batch_size
#         self.epochs = configs.epochs
#
#         self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
#         self.max_length = configs.max_length
#
#     def read_data_file(self, data_type):
#         if data_type == 'train':
#             data_file = self.train_file
#         elif data_type == 'valid':
#             data_file = self.valid_file
#         elif data_type == 'test':
#             data_file = self.test_file
#
#         with open(data_file, 'r', encoding='utf-8') as f:
#
#             raw = f.readlines()
#
#             f.close()
#
#         text = []
#         for r in raw:
#             text.append(r.strip())
#
#         labels = np.load(join(self.data_dir, '{}_mood_prob.npy'.format(data_type)))
#         c_labels = transfer_to_coarse(labels)
#
#         return text, c_labels
#
#     def get_data(self):
#
#         text, labels = self.read_data_file(self.data_type)
#
#         tokenizer = self.bert_tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#
#         input_ids = tokenizer['input_ids']
#         token_type_ids = tokenizer['token_type_ids']
#         attention_mask = tokenizer['attention_mask']
#
#         labels = torch.tensor(labels)
#
#         data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
#
#         return data


class Medical_Dataset_label_1_RoBERTa(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_1):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_1)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_1)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.roberta_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            # title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.roberta_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        # token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, attention_mask, labels)

        return data

class Medical_Dataset_label_3_RoBERTa(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_Medical_label_3):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_Medical_label_3)
        # self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE_Medical_label_3)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        # self.roberta_tokenizer = BertTokenizer.from_pretrained(configs.pretrain_roberta_all)
        self.roberta_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)

        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        # elif data_type == 'valid':
        #     data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        infos = []
        labels = []
        data = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
        df = pd.DataFrame(data)
        self.text_len = df.shape[0]
        print(df.shape)
        for i in range(self.text_len):
            labels.append(int(df.iloc[i, 3]))
            # title = str(df.iloc[i, 0])
            info = str(df.iloc[i, 1])
            # titles.append(title)
            infos.append(info)

        c_labels = np.array(labels)

        return infos, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.roberta_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        # token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, attention_mask, labels)

        return data


# class Medical_Dataset_NN(Dataset):
#
#     def __init__(self, configs, data_type, data_dir=DATA_DIR_NS):
#         self.data_dir = data_dir
#         self.data_type = data_type
#
#         self.train_file = join(data_dir, TRAIN_FILE_NS)
#         self.test_file = join(data_dir, TEST_FILE_NS)
#
#         self.embedding_size = configs.GloVe_embedding_length
#         self.pieces_size = configs.pieces_size
#         self.split_len = configs.split_len
#         self.overlap_len = configs.overlap_len
#
#         self.batch_size = configs.batch_size
#         self.epochs = configs.epochs
#
#         self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
#         self.max_length = configs.max_length
#
#     def read_data_file(self, data_type):
#         if data_type == 'train':
#             data_file = self.train_file
#         elif data_type == 'test':
#             data_file = self.test_file
#
#         texts = []
#         labels = []
#
#         ag_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
#         ag_df = pd.DataFrame(ag_data)
#         self.text_len = ag_df.shape[0]
#         print(ag_df.shape)
#         for i in range(self.text_len):
#             labels.append(int(ag_df.iloc[i, 1]))
#             raw = str(ag_df.iloc[i, 0])
#             texts.append(raw.strip())
#
#         c_labels = np.array(labels)
#
#         return texts, c_labels
#
#     def get_data(self):
#
#         global_vectors = vocab.GloVe(name='6B', dim=self.embedding_size, cache=glove_cache_dir)
#         texts, labels = self.read_data_file(self.data_type)
#         tokenizer = get_tokenizer("basic_english")
#         texts = [tokenizer(x) for x in texts]
#         texts = [tokens + [""] * (self.max_length - len(tokens)) if len(tokens) < self.max_length else tokens[
#                                                                                                        :self.max_length]
#                  for tokens in texts]
#         X_tensor = torch.zeros(len(texts), self.max_length, self.embedding_size)
#         for i, tokens in enumerate(texts):
#             X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
#             # print(X_tensor[i])
#             # print(X_tensor[i].shape)
#             # print(X_tensor[i].shape)
#
#         labels = torch.tensor(labels)
#
#         data = TensorDataset(X_tensor, labels)
#         print(X_tensor.shape)
#         print(labels.shape)
#
#         return data
