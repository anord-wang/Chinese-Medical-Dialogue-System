from transformers import BertTokenizer, RobertaTokenizer
from config import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from os.path import join
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
import torchtext.vocab as vocab
from torchtext.data import get_tokenizer

from matplotlib import pyplot as plt


def build_train_data(configs):
    # train_dataset = MyDataset(configs, data_type='train')
    # train_dataset = IMdbDataset_normal(configs, data_type='train')
    # train_dataset = IMdbDataset(configs, data_type='train')
    # train_dataset = IMdbDataset_GAT(configs, data_type='train')
    # train_dataset = news_20_GAT_Dataset(configs, data_type='train')
    # train_dataset = news_20_Dataset_normal(configs, data_type='train')
    # train_dataset = news_20_Dataset(configs, data_type='train')
    # train_dataset = AG_news_Dataset(configs, data_type='train')
    # train_dataset = AG_news_Dataset_normal(configs, data_type='train')
    # train_dataset = AG_news_Dataset_GAT(configs, data_type='train')
    # train_dataset = WOS_11967_Dataset(configs, data_type='train')
    # train_dataset = WOS_11967_Dataset_normal(configs, data_type='train')
    # train_dataset = WOS_11967_Dataset_GAT(configs, data_type='train')
    # train_dataset = Ohmused_Dataset(configs, data_type='train')
    # train_dataset = Ohmused_Dataset_GAT(configs, data_type='train')
    # train_dataset = native_speaker_Dataset_normal(configs, data_type='train')
    # train_dataset = native_speaker_Dataset_normal_ro(configs, data_type='train')
    # train_dataset = native_speaker_Dataset(configs, data_type='train')
    # train_dataset = native_speaker_Dataset_GAT(configs, data_type='train')
    train_dataset = native_speaker_Dataset_NN(configs, data_type='train')
    # train_dataset = yanbao_Dataset(configs, data_type='train')
    # train_dataset = yanbao_BERT_Dataset(configs, data_type='train')
    # train_dataset = yanbao_Dataset_GAT(configs, data_type='train')
    # train_dataset = yanbao_BERT_summary_Dataset(configs, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset.get_data(), batch_size=configs.batch_size,
                                               shuffle=True)
    return train_loader


def build_inference_data(configs, data_type):
    # dataset = MyDataset(configs, data_type)
    # dataset = IMdbDataset_normal(configs, data_type)
    # dataset = IMdbDataset(configs, data_type)
    # dataset = IMdbDataset_GAT(configs, data_type)
    # dataset = news_20_GAT_Dataset(configs, data_type)
    # dataset = news_20_Dataset(configs, data_type)
    # dataset = news_20_Dataset_normal(configs, data_type)
    # dataset = AG_news_Dataset(configs, data_type)
    # dataset = AG_news_Dataset_normal(configs, data_type)
    # dataset = AG_news_Dataset_GAT(configs, data_type)
    # dataset = WOS_11967_Dataset(configs, data_type)
    # dataset = WOS_11967_Dataset_normal(configs, data_type)
    # dataset = WOS_11967_Dataset_GAT(configs, data_type)
    # dataset = Ohmused_Dataset(configs, data_type)
    # dataset = Ohmused_Dataset_GAT(configs, data_type)
    # dataset = native_speaker_Dataset(configs, data_type)
    # dataset = native_speaker_Dataset_GAT(configs, data_type)
    dataset = native_speaker_Dataset_NN(configs, data_type)
    # dataset = native_speaker_Dataset_normal(configs, data_type)
    # dataset = native_speaker_Dataset_normal_ro(configs, data_type)
    # dataset = yanbao_Dataset(configs, data_type)
    # dataset = yanbao_BERT_Dataset(configs, data_type)
    # dataset = yanbao_Dataset_GAT(configs, data_type)
    # dataset = yanbao_BERT_summary_Dataset(configs, data_type)
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


class MyDataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        with open(data_file, 'r', encoding='utf-8') as f:

            raw = f.readlines()

            f.close()

        text = []
        for r in raw:
            text.append(r.strip())

        labels = np.load(join(self.data_dir, '{}_mood_prob.npy'.format(data_type)))
        c_labels = transfer_to_coarse(labels)

        return text, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            text,
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


class yanbao_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_yanbao):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_yanbao)
        self.test_file = join(data_dir, TEST_FILE_yanbao)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        contents = []
        labels = []
        with open(data_file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('@_@')
                contents.append(content)
                labels.append(int(label))
        print(labels)
        c_labels = np.array(labels)
        print(c_labels)
        return contents, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            text,
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


class yanbao_BERT_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_yanbao):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_yanbao)
        self.test_file = join(data_dir, TEST_FILE_yanbao)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = [one for one in text]
        # text = text.split(' ')
        print(len(text))
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            # text_piece = ' '.join(text_piece_word)
            text_piece = ''.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        contents = []
        length = []
        labels = []

        with open(data_file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('@_@')
                split, lens = self.get_split_text(content)
                # print(split)
                contents.append(split)
                length.append(lens)
                labels.append(int(label))
        c_labels = np.array(labels)
        self.text_len = len(length)

        # print(np.size(text))
        text_new = []
        print(len(contents))
        for i in range(len(contents)):
            line = contents[i] + ['padding'] * max(0, self.pieces_size - len(contents[i]))
            text_new.append(line)

        # sum1 = 0
        # sum0 = 0
        # sum2 = 0
        # for i in range(len(contents)):
        #     sum1 = sum1 + len(contents[i])
        #     if len(contents[i])>8:
        #         sum0=sum0+len(contents[i])-8
        # for i in range(len(text_new)):
        #     sum2 = sum2 + len(text_new[i])
        # print(sum1)
        # print(sum0)
        # print(sum2)
        #
        # print(c_labels.shape)
        # # print(contents[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length

    def get_data(self):

        text, labels, length = self.read_data_file(self.data_type)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        print(text1)
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


class yanbao_BERT_summary_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_yanbao):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_yanbao)
        self.test_file = join(data_dir, TEST_FILE_yanbao)
        self.train_summary_file = join(data_dir, TRAIN_FILE_yanbao_summary_75)
        self.test_summary_file = join(data_dir, TEST_FILE_yanbao_summary_75)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.max_length = configs.max_length
        self.max_length_summary = configs.max_length_summary

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = [one for one in text]
        # text = text.split(' ')
        print(len(text))
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            # text_piece = ' '.join(text_piece_word)
            text_piece = ''.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
            data_file_summary = self.train_summary_file
        elif data_type == 'test':
            data_file = self.test_file
            data_file_summary = self.test_summary_file

        contents = []
        length = []
        labels = []

        with open(data_file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('@_@')
                split, lens = self.get_split_text(content)
                # print(split)
                contents.append(split)
                length.append(lens)
                labels.append(int(label))
        c_labels = np.array(labels)

        summaries = []
        with open(data_file_summary, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                summary, label = lin.split('@_@')
                summaries.append(summary)

        self.text_len = len(length)

        text_new = []
        print(len(contents))
        for i in range(len(contents)):
            line = contents[i] + ['padding'] * max(0, self.pieces_size - len(contents[i]))
            text_new.append(line)

        return text_new, c_labels, length, summaries

    def get_data(self):

        text, labels, length, summaries = self.read_data_file(self.data_type)

        tokenizer_summary = self.bert_tokenizer(
            summaries,
            padding=True,
            truncation=True,
            max_length=self.max_length_summary,
            return_tensors='pt'
        )
        input_ids_summary = tokenizer_summary['input_ids']
        token_type_ids_summary = tokenizer_summary['token_type_ids']
        attention_mask_summary = tokenizer_summary['attention_mask']
        zero = torch.zeros([self.text_len, self.embedding_size - input_ids_summary.shape[1]])
        input_ids_summary = torch.cat((input_ids_summary, zero), 1)
        token_type_ids_summary = torch.cat((token_type_ids_summary, zero), 1)
        attention_mask_summary = torch.cat((attention_mask_summary, zero), 1)

        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        print(text1)
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, input_ids_summary,
                             token_type_ids_summary, attention_mask_summary)

        return data


class yanbao_Dataset_GAT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_yanbao):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_yanbao)
        self.test_file = join(data_dir, TEST_FILE_yanbao)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.threshold = 0

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path_Chinese)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = [one for one in text]
        # text = text.split(' ')
        # print(len(text))
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            # text_piece = ' '.join(text_piece_word)
            text_piece = ''.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
            # print(matrix)
        return matrix

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def get_gat_adj(self, text):
        vectorizer = TfidfVectorizer(analyzer="char", token_pattern=u'(?u)\\b\\w+\\b')
        vectors = vectorizer.fit_transform(text).toarray()
        # print(vectorizer.vocabulary_)
        # print(vectors)
        gat_adj = np.zeros((self.pieces_size, self.pieces_size), dtype=np.int)
        for i in range(min(self.pieces_size, len(text))):
            for j in range(min(self.pieces_size, len(text))):
                print(self.cosine_similarity(vectors[i], vectors[j]))
                if self.cosine_similarity(vectors[i], vectors[j]) > self.threshold:
                    gat_adj[i][j] = 1
        # print(gat_adj)
        return gat_adj

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        contents = []
        length = []
        labels = []
        gat_adj = []

        with open(data_file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('@_@')
                split, lens = self.get_split_text(content)
                print(split)
                adj = self.get_gat_adj(split)
                print(adj)
                gat_adj.append(adj)
                contents.append(split)
                length.append(lens)
                labels.append(int(label))
        c_labels = np.array(labels)
        self.text_len = len(length)

        # print(np.size(text))
        text_new = []
        print(len(contents))
        for i in range(len(contents)):
            line = contents[i] + ['padding'] * max(0, self.pieces_size - len(contents[i]))
            text_new.append(line)

        return text_new, c_labels, length, gat_adj
        # return text_new, c_labels, length

    def get_data(self):

        text, labels, length, gat_adj = self.read_data_file(self.data_type)
        # text, labels, length = self.read_data_file(self.data_type)
        # print(labels.shape)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']

            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)

            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)

            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        gat_adj = torch.tensor(gat_adj)

        print(labels.shape)
        print(length.shape)
        # print(gat_adj.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, gat_adj)
        # data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


def tokenlize(content):
    content = re.sub('<.*?>', ' ', content)

    fileters = ['\/', '\(', '\)', ':', '\.', '\t', '\n', '\x97', '\x96', '#', '$', '%', '&']
    content = re.sub('|'.join(fileters), ' ', content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


class IMdbDataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        # lens=[]
        text.strip()
        text = text.split(' ')
        # print(len(text))
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        # lens.append(length)
        for w in range(length):
            # if w == 0:  # 第一次,直接分割长度放进去
            #     text_piece_word = text[:split_len]
            # else:  # 否则, 按照(分割长度-overlap)往后走
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file
        temp_data_path = [os.path.join(data_file, 'pos'), os.path.join(data_file, 'neg')]
        self.total_file_path = []
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)
        text = []
        length = []
        labels = []
        # print(self.total_file_path[547])
        for item in self.total_file_path:
            with open(item, 'r', encoding='utf-8') as f:
                raw = f.readlines()
                f.close()
            # for r in raw:
            #     split, lens = self.get_split_text(r)
            #     # print(ax)
            #     # text.append(r.strip())
            #     length.append(lens)
            #     text.append(split)
            #     # print(text)
            split, lens = self.get_split_text(raw[0])
            length.append(lens)
            text.append(split)
            label_str = item.split('/')[-2]
            label = 0 if label_str == 'neg' else 1
            labels.append(label)
        # print(np.size(labels))
        # print(sum(np.array(labels)))

        # c_labels = transfer_to_onehot(labels)
        c_labels = np.array(labels)

        # print(np.size(text))
        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            # line = text[i] + [] * max(0, 8 - len(text[i]))
            text_new.append(line)
            # line = []
        # sum1 = 0
        # sum0 = 0
        # sum2 = 0
        # for i in range(len(text)):
        #     sum1 = sum1 + len(text[i])
        #     if len(text[i])>8:
        #         sum0=sum0+len(text[i])-8
        # for i in range(len(text_new)):
        #     sum2 = sum2 + len(text_new[i])
        # print(sum1)
        # print(sum0)
        # print(sum2)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length

    def get_data(self):

        text, labels, length = self.read_data_file(self.data_type)
        # print(labels.shape)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        # print(np.size(text1[0]))
        # print(np.size(text1))
        # print(text1[0][547])
        # print(text[547][0])
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            # print(input_ids)
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            # print(token_type_ids)
            # print(attention_mask)
            zero = torch.zeros([25000, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([25000, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([25000, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([25000, self.embedding_size, 1])
            attention_mask = attention_mask.view([25000, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask
            # print(input_ids.shape)
            # print(token_type_ids.shape)
            # print(attention_mask.shape)

        # tokenizer = self.bert_tokenizer(
        #     text,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        #
        # input_ids = tokenizer['input_ids']
        # print(input_ids)
        # print(input_ids.shape)
        # token_type_ids = tokenizer['token_type_ids']
        # print(token_type_ids)
        # print(token_type_ids.shape)
        # attention_mask = tokenizer['attention_mask']
        # print(attention_mask)
        # print(attention_mask.shape)

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, 25000)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


class IMdbDataset_normal(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file
        temp_data_path = [os.path.join(data_file, 'pos'), os.path.join(data_file, 'neg')]
        self.total_file_path = []
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)
        texts = []
        labels = []
        # print(self.total_file_path[547])
        for item in self.total_file_path:
            with open(item, 'r', encoding='utf-8') as f:
                raw = f.readlines()
                f.close()
            text = raw[0].strip()
            texts.append(text)
            label_str = item.split('/')[-2]
            label = 0 if label_str == 'neg' else 1
            labels.append(label)
        c_labels = np.array(labels)

        return texts, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            text,
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


class IMdbDataset_GAT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.threshold = configs.gat_adj_threshold

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def get_gat_adj(self, text):
        vectorizer = TfidfVectorizer(analyzer="char", token_pattern=u'(?u)\\b\\w+\\b')
        vectors = vectorizer.fit_transform(text).toarray()
        # print(vectors)
        gat_adj = np.zeros((self.pieces_size, self.pieces_size), dtype=np.int)
        for i in range(min(self.pieces_size, len(text))):
            for j in range(min(self.pieces_size, len(text))):
                # print(self.cosine_similarity(vectors[i], vectors[j]))
                if self.cosine_similarity(vectors[i], vectors[j]) > self.threshold:
                    gat_adj[i][j] = 1
        # print(gat_adj)
        return gat_adj

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file
        temp_data_path = [os.path.join(data_file, 'pos'), os.path.join(data_file, 'neg')]
        self.total_file_path = []
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)
        text = []
        length = []
        labels = []
        gat_adj = []
        # print(self.total_file_path[547])
        for item in self.total_file_path:
            with open(item, 'r', encoding='utf-8') as f:
                raw = f.readlines()
                f.close()
            # for r in raw:
            #     split, lens = self.get_split_text(r)
            #     # print(ax)
            #     # text.append(r.strip())
            #     length.append(lens)
            #     text.append(split)
            #     # print(text)
            split, lens = self.get_split_text(raw[0])
            adj = self.get_gat_adj(split)
            length.append(lens)
            text.append(split)
            gat_adj.append(adj)
            label_str = item.split('/')[-2]
            label = 0 if label_str == 'neg' else 1
            labels.append(label)

        c_labels = np.array(labels)

        # print(np.size(text))
        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length, gat_adj

    def get_data(self):

        text, labels, length, gat_adj = self.read_data_file(self.data_type)
        # print(labels.shape)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        # print(np.size(text1[0]))
        # print(np.size(text1))
        # print(text1[0][547])
        # print(text[547][0])
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            # print(input_ids)
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            # print(token_type_ids)
            # print(attention_mask)
            zero = torch.zeros([25000, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([25000, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([25000, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([25000, self.embedding_size, 1])
            attention_mask = attention_mask.view([25000, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

            # print(input_ids.shape)
            # print(token_type_ids.shape)
            # print(attention_mask.shape)

        # tokenizer = self.bert_tokenizer(
        #     text,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        #
        # input_ids = tokenizer['input_ids']
        # print(input_ids)
        # print(input_ids.shape)
        # token_type_ids = tokenizer['token_type_ids']
        # print(token_type_ids)
        # print(token_type_ids.shape)
        # attention_mask = tokenizer['attention_mask']
        # print(attention_mask)
        # print(attention_mask.shape)

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, 25000)
        gat_adj = torch.tensor(gat_adj)

        print(labels.shape)
        print(length.shape)
        print(gat_adj.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, gat_adj)

        return data


from sklearn.datasets import fetch_20newsgroups


class news_20_Dataset_normal(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            # newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            newsgroups_train = fetch_20newsgroups(subset='train', remove=('quotes'))
            text_file = newsgroups_train['data']
            labels = newsgroups_train.target
        elif data_type == 'valid':
            # newsgroups_valid = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            newsgroups_valid = fetch_20newsgroups(subset='test', remove=('quotes'))
            text_file = newsgroups_valid['data']
            labels = newsgroups_valid.target
        elif data_type == 'test':
            # newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            newsgroups_test = fetch_20newsgroups(subset='test', remove=('quotes'))
            text_file = newsgroups_test['data']
            labels = newsgroups_test.target

        texts = []

        print(len(text_file))
        self.text_len = len(text_file)
        for i in range(len(text_file)):
            line = text_file[i].replace("\n", "")
            # print(line)
            raw = line.strip()
            texts.append(raw)
        c_labels = np.array(labels)

        return texts, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            text,
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


class news_20_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            # newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            newsgroups_train = fetch_20newsgroups(subset='train', remove=('quotes'))
            text_file = newsgroups_train['data']
            labels = newsgroups_train.target
        elif data_type == 'valid':
            # newsgroups_valid = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            newsgroups_valid = fetch_20newsgroups(subset='test', remove=('quotes'))
            text_file = newsgroups_valid['data']
            labels = newsgroups_valid.target
        elif data_type == 'test':
            # newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            newsgroups_test = fetch_20newsgroups(subset='test', remove=('quotes'))
            text_file = newsgroups_test['data']
            labels = newsgroups_test.target

        text = []
        length = []
        print(len(text_file))
        self.text_len = len(text_file)
        for i in range(len(text_file)):
            line = text_file[i].replace("\n", "")
            # print(line)
            split, lens = self.get_split_text(line)
            length.append(lens)
            text.append(split)
        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 75))
        # print(np.percentile(xxdx, 85))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length

    def get_data(self):

        text, labels, length = self.read_data_file(self.data_type)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


class news_20_GAT_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.threshold = configs.gat_adj_threshold

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def get_gat_adj(self, text):
        vectorizer = TfidfVectorizer(analyzer="char", token_pattern=u'(?u)\\b\\w+\\b')
        tfidf = []
        # for i in range(len(text)):
        vectors = vectorizer.fit_transform(text).toarray()
        # print(vectors)
        # tfidf.append(vector)
        # vectors = vectorizer.fit_transform(text)
        gat_adj = np.zeros((self.pieces_size, self.pieces_size), dtype=np.int)
        for i in range(min(self.pieces_size, len(text))):
            for j in range(min(self.pieces_size, len(text))):
                # print(self.cosine_similarity(vectors[i], vectors[j]))
                if self.cosine_similarity(vectors[i], vectors[j]) > self.threshold:
                    gat_adj[i][j] = 1
        # print(gat_adj)
        return gat_adj

    def read_data_file(self, data_type):
        if data_type == 'train':
            # newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            newsgroups_train = fetch_20newsgroups(subset='train', remove=('quotes'))
            # newsgroups_train = fetch_20newsgroups(subset='train')
            text_file = newsgroups_train['data']
            labels = newsgroups_train.target
        elif data_type == 'valid':
            # newsgroups_valid = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            newsgroups_valid = fetch_20newsgroups(subset='test', remove=('quotes'))
            # newsgroups_valid = fetch_20newsgroups(subset='test')
            text_file = newsgroups_valid['data']
            labels = newsgroups_valid.target
        elif data_type == 'test':
            # newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
            newsgroups_test = fetch_20newsgroups(subset='test', remove=('quotes'))
            # newsgroups_test = fetch_20newsgroups(subset='test')
            text_file = newsgroups_test['data']
            labels = newsgroups_test.target

        text = []
        length = []
        gat_adj = []
        # print(len(text_file))
        self.text_len = len(text_file)
        for i in range(len(text_file)):
            line = text_file[i].replace("\n", "").strip()
            # print(line)
            split, lens = self.get_split_text(line)
            # print(split)
            adj = self.get_gat_adj(split)
            length.append(lens)
            text.append(split)
            gat_adj.append(adj)
        c_labels = np.array(labels)

        text_new = []
        # print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 75))
        # print(np.percentile(xxdx, 85))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length, gat_adj

    def get_data(self):

        text, labels, length, gat_adj = self.read_data_file(self.data_type)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        gat_adj = torch.tensor(gat_adj)

        print(labels.shape)
        print(length.shape)
        print(gat_adj.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, gat_adj)

        return data


import pandas as pd


class AG_news_Dataset_normal(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_AG_NEWS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_AG_NEWS)
        self.test_file = join(data_dir, TEST_FILE_AG_NEWS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        texts = []
        labels = []

        ag_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 0]) - 1)
            raw = str(ag_df.iloc[i, 2])
            texts.append(raw.strip())

        c_labels = np.array(labels)

        return texts, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            text,
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


class AG_news_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_AG_NEWS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_AG_NEWS)
        self.test_file = join(data_dir, TEST_FILE_AG_NEWS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        ag_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 0]) - 1)
            raw = str(ag_df.iloc[i, 2])
            split, lens = self.get_split_text(raw)
            text.append(split)
            length.append(lens)

        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 75))
        # print(np.percentile(xxdx, 85))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length

    def get_data(self):

        text, labels, length = self.read_data_file(self.data_type)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


class AG_news_Dataset_GAT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_AG_NEWS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_AG_NEWS)
        self.test_file = join(data_dir, TEST_FILE_AG_NEWS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.threshold = configs.gat_adj_threshold

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def get_gat_adj(self, text):
        vectorizer = TfidfVectorizer(analyzer="char", token_pattern=u'(?u)\\b\\w+\\b')
        vectors = vectorizer.fit_transform(text).toarray()
        # print(vectors)
        gat_adj = np.zeros((self.pieces_size, self.pieces_size), dtype=np.int)
        for i in range(min(self.pieces_size, len(text))):
            for j in range(min(self.pieces_size, len(text))):
                # print(self.cosine_similarity(vectors[i], vectors[j]))
                if self.cosine_similarity(vectors[i], vectors[j]) > self.threshold:
                    gat_adj[i][j] = 1
        # print(gat_adj)
        return gat_adj

    def read_data_file(self, data_type):

        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        gat_adj = []
        ag_data = pd.read_csv(data_file, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 0]) - 1)
            raw = str(ag_df.iloc[i, 2])
            split, lens = self.get_split_text(raw)
            adj = self.get_gat_adj(split)
            text.append(split)
            length.append(lens)
            gat_adj.append(adj)

        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length, gat_adj

    def get_data(self):

        text, labels, length, gat_adj = self.read_data_file(self.data_type)
        # print(labels.shape)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        # print(np.size(text1[0]))
        # print(np.size(text1))
        # print(text1[0][547])
        # print(text[547][0])
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            # print(input_ids)
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            # print(token_type_ids)
            # print(attention_mask)
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

            # print(input_ids.shape)
            # print(token_type_ids.shape)
            # print(attention_mask.shape)

        # tokenizer = self.bert_tokenizer(
        #     text,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        #
        # input_ids = tokenizer['input_ids']
        # print(input_ids)
        # print(input_ids.shape)
        # token_type_ids = tokenizer['token_type_ids']
        # print(token_type_ids)
        # print(token_type_ids.shape)
        # attention_mask = tokenizer['attention_mask']
        # print(attention_mask)
        # print(attention_mask.shape)

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        gat_adj = torch.tensor(gat_adj)

        print(labels.shape)
        print(length.shape)
        print(gat_adj.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, gat_adj)

        return data


class WOS_11967_Dataset_normal(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_WOS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_WOS)
        self.test_file = join(data_dir, TEST_FILE_WOS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        texts = []
        labels = []

        ag_data = pd.read_csv(data_file, header=0, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 1]))
            raw = str(ag_df.iloc[i, 0])
            texts.append(raw.strip())

        c_labels = np.array(labels)

        return texts, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.bert_tokenizer(
            text,
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


class WOS_11967_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_WOS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_WOS)
        self.test_file = join(data_dir, TEST_FILE_WOS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        ag_data = pd.read_csv(data_file, header=0, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 1]))
            raw = str(ag_df.iloc[i, 0])
            split, lens = self.get_split_text(raw)
            text.append(split)
            length.append(lens)

        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 75))
        # print(np.percentile(xxdx, 85))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length

    def get_data(self):

        text, labels, length = self.read_data_file(self.data_type)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


class WOS_11967_Dataset_GAT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_WOS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_WOS)
        self.test_file = join(data_dir, TEST_FILE_WOS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.threshold = configs.gat_adj_threshold

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def get_gat_adj(self, text):
        vectorizer = TfidfVectorizer(analyzer="char", token_pattern=u'(?u)\\b\\w+\\b')
        vectors = vectorizer.fit_transform(text).toarray()
        # print(vectors)
        gat_adj = np.zeros((self.pieces_size, self.pieces_size), dtype=np.int)
        for i in range(min(self.pieces_size, len(text))):
            for j in range(min(self.pieces_size, len(text))):
                # print(self.cosine_similarity(vectors[i], vectors[j]))
                if self.cosine_similarity(vectors[i], vectors[j]) > self.threshold:
                    gat_adj[i][j] = 1
        # print(gat_adj)
        return gat_adj

    def read_data_file(self, data_type):

        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        gat_adj = []
        WOS_data = pd.read_csv(data_file, header=0, low_memory=False)  # 防止弹出警告
        WOS_df = pd.DataFrame(WOS_data)
        self.text_len = WOS_df.shape[0]
        for i in range(self.text_len):
            labels.append(int(WOS_df.iloc[i, 1]))
            raw = str(WOS_df.iloc[i, 0])
            split, lens = self.get_split_text(raw)
            adj = self.get_gat_adj(split)
            text.append(split)
            length.append(lens)
            gat_adj.append(adj)

        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length, gat_adj

    def get_data(self):

        text, labels, length, gat_adj = self.read_data_file(self.data_type)
        # print(labels.shape)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        # print(np.size(text1[0]))
        # print(np.size(text1))
        # print(text1[0][547])
        # print(text[547][0])
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            # print(input_ids)
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            # print(token_type_ids)
            # print(attention_mask)
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

            # print(input_ids.shape)
            # print(token_type_ids.shape)
            # print(attention_mask.shape)

        # tokenizer = self.bert_tokenizer(
        #     text,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        #
        # input_ids = tokenizer['input_ids']
        # print(input_ids)
        # print(input_ids.shape)
        # token_type_ids = tokenizer['token_type_ids']
        # print(token_type_ids)
        # print(token_type_ids.shape)
        # attention_mask = tokenizer['attention_mask']
        # print(attention_mask)
        # print(attention_mask.shape)

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        gat_adj = torch.tensor(gat_adj)

        print(labels.shape)
        print(length.shape)
        print(gat_adj.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, gat_adj)

        return data


class native_speaker_Dataset_normal(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_NS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_NS)
        self.test_file = join(data_dir, TEST_FILE_NS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        # self.robert_tokenizer = RobertaTokenizer.from_pretrained(configs.roberta_cache_path_english)

        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        texts = []
        labels = []

        ag_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 1]))
            raw = str(ag_df.iloc[i, 0])
            texts.append(raw.strip())

        c_labels = np.array(labels)

        return texts, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.robert_tokenizer(
            text,
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


class native_speaker_Dataset_normal_ro(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_NS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_NS)
        self.test_file = join(data_dir, TEST_FILE_NS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.robert_tokenizer = RobertaTokenizer.from_pretrained(configs.roberta_cache_path_english)

        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        texts = []
        labels = []

        ag_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 1]))
            raw = str(ag_df.iloc[i, 0])
            texts.append(raw.strip())

        c_labels = np.array(labels)

        return texts, c_labels

    def get_data(self):

        text, labels = self.read_data_file(self.data_type)

        tokenizer = self.robert_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenizer['input_ids']
        attention_mask = tokenizer['attention_mask']
        # attention_mask = tokenizer['attention_mask']

        labels = torch.tensor(labels)
        print(tokenizer)
        print(labels.shape)

        data = TensorDataset(input_ids, attention_mask, labels)

        return data


class native_speaker_Dataset_NN(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_NS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_NS)
        self.test_file = join(data_dir, TEST_FILE_NS)

        self.embedding_size = configs.GloVe_embedding_length
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        texts = []
        labels = []

        ag_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 1]))
            raw = str(ag_df.iloc[i, 0])
            texts.append(raw.strip())

        c_labels = np.array(labels)

        return texts, c_labels

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


class native_speaker_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_NS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_NS)
        self.test_file = join(data_dir, TEST_FILE_NS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        ag_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
        ag_df = pd.DataFrame(ag_data)
        self.text_len = ag_df.shape[0]
        print(ag_df.shape)
        for i in range(self.text_len):
            labels.append(int(ag_df.iloc[i, 1]))
            raw = str(ag_df.iloc[i, 0])
            split, lens = self.get_split_text(raw)
            text.append(split)
            length.append(lens)

        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 75))
        # print(np.percentile(xxdx, 85))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length

    def get_data(self):

        text, labels, length = self.read_data_file(self.data_type)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


class native_speaker_Dataset_GAT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_NS):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_NS)
        self.test_file = join(data_dir, TEST_FILE_NS)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.threshold = configs.gat_adj_threshold

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def get_gat_adj(self, text):
        vectorizer = TfidfVectorizer(analyzer="char", token_pattern=u'(?u)\\b\\w+\\b')
        vectors = vectorizer.fit_transform(text).toarray()
        # print(vectors)
        gat_adj = np.zeros((self.pieces_size, self.pieces_size), dtype=np.int)
        for i in range(min(self.pieces_size, len(text))):
            for j in range(min(self.pieces_size, len(text))):
                # print(self.cosine_similarity(vectors[i], vectors[j]))
                if self.cosine_similarity(vectors[i], vectors[j]) > self.threshold:
                    gat_adj[i][j] = 1
        # print(gat_adj)
        return gat_adj

    def read_data_file(self, data_type):

        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        gat_adj = []
        WOS_data = pd.read_csv(data_file, header=None, low_memory=False)  # 防止弹出警告
        WOS_df = pd.DataFrame(WOS_data)
        self.text_len = WOS_df.shape[0]
        for i in range(self.text_len):
            labels.append(int(WOS_df.iloc[i, 1]))
            raw = str(WOS_df.iloc[i, 0])
            split, lens = self.get_split_text(raw)
            adj = self.get_gat_adj(split)
            text.append(split)
            length.append(lens)
            gat_adj.append(adj)

        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length, gat_adj

    def get_data(self):

        text, labels, length, gat_adj = self.read_data_file(self.data_type)
        # print(labels.shape)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        # print(np.size(text1[0]))
        # print(np.size(text1))
        # print(text1[0][547])
        # print(text[547][0])
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            # print(input_ids)
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            # print(token_type_ids)
            # print(attention_mask)
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

            # print(input_ids.shape)
            # print(token_type_ids.shape)
            # print(attention_mask.shape)

        # tokenizer = self.bert_tokenizer(
        #     text,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        #
        # input_ids = tokenizer['input_ids']
        # print(input_ids)
        # print(input_ids.shape)
        # token_type_ids = tokenizer['token_type_ids']
        # print(token_type_ids)
        # print(token_type_ids.shape)
        # attention_mask = tokenizer['attention_mask']
        # print(attention_mask)
        # print(attention_mask.shape)

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        gat_adj = torch.tensor(gat_adj)

        print(labels.shape)
        print(length.shape)
        print(gat_adj.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, gat_adj)

        return data


class Ohmused_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_OH):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_OH)
        self.test_file = join(data_dir, TEST_FILE_OH)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        oh_data = pd.read_csv(data_file, header=0, low_memory=False)  # 防止弹出警告
        oh_df = pd.DataFrame(oh_data)
        self.text_len = oh_df.shape[0]
        print(oh_df.shape)
        for i in range(self.text_len):
            labels.append(int(oh_df.iloc[i, 1].strip('C')) - 1)
            raw = str(oh_df.iloc[i, 0])
            split, lens = self.get_split_text(raw)
            text.append(split)
            length.append(lens)

        c_labels = np.array(labels)
        # print(max(c_labels))
        # print(min(c_labels))
        # plt.hist(c_labels)
        # plt.show()
        # print(range(c_labels))

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # # print(text[0][1])
        # print('max length')
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print('###############################################################statistic of length###############################################################')
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 75))
        # print(np.percentile(xxdx, 85))
        # print(np.percentile(xxdx, 95))
        # print('###############################################################statistic of length###############################################################')
        return text_new, c_labels, length

    def get_data(self):

        text, labels, length = self.read_data_file(self.data_type)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length)

        return data


class Ohmused_Dataset_GAT(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_OH):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_OH)
        self.test_file = join(data_dir, TEST_FILE_OH)

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.threshold = configs.gat_adj_threshold

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.max_length = configs.max_length

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = text.split(' ')
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            text_piece = ' '.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def get_gat_adj(self, text):
        vectorizer = TfidfVectorizer(analyzer="char", token_pattern=u'(?u)\\b\\w+\\b')
        vectors = vectorizer.fit_transform(text).toarray()
        # print(vectors)
        gat_adj = np.zeros((self.pieces_size, self.pieces_size), dtype=np.int)
        for i in range(min(self.pieces_size, len(text))):
            for j in range(min(self.pieces_size, len(text))):
                # print(self.cosine_similarity(vectors[i], vectors[j]))
                if self.cosine_similarity(vectors[i], vectors[j]) > self.threshold:
                    gat_adj[i][j] = 1
        # print(gat_adj)
        return gat_adj

    def read_data_file(self, data_type):

        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'test':
            data_file = self.test_file

        text = []
        length = []
        labels = []
        gat_adj = []
        Oh_data = pd.read_csv(data_file, header=0, low_memory=False)  # 防止弹出警告
        Oh_df = pd.DataFrame(Oh_data)
        self.text_len = Oh_df.shape[0]
        for i in range(self.text_len):
            labels.append(int(Oh_df.iloc[i, 1].strip('C')) - 1)
            raw = str(Oh_df.iloc[i, 0])
            split, lens = self.get_split_text(raw)
            adj = self.get_gat_adj(split)
            text.append(split)
            length.append(lens)
            gat_adj.append(adj)

        c_labels = np.array(labels)

        text_new = []
        print(len(text))
        for i in range(len(text)):
            line = text[i] + ['padding'] * max(0, self.pieces_size - len(text[i]))
            text_new.append(line)

        # print(c_labels.shape)
        # print(text[0][1])
        # print(max(length))
        # plt.hist(length, 100)
        # plt.show()
        # xxdx = np.array(length)
        # print(np.median(xxdx))
        # print(np.percentile(xxdx, 95))
        return text_new, c_labels, length, gat_adj

    def get_data(self):

        text, labels, length, gat_adj = self.read_data_file(self.data_type)
        # print(labels.shape)
        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        # print(np.size(text1[0]))
        # print(np.size(text1))
        # print(text1[0][547])
        # print(text[547][0])
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            # print(input_ids)
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            # print(token_type_ids)
            # print(attention_mask)
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

            # print(input_ids.shape)
            # print(token_type_ids.shape)
            # print(attention_mask.shape)

        # tokenizer = self.bert_tokenizer(
        #     text,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        #
        # input_ids = tokenizer['input_ids']
        # print(input_ids)
        # print(input_ids.shape)
        # token_type_ids = tokenizer['token_type_ids']
        # print(token_type_ids)
        # print(token_type_ids.shape)
        # attention_mask = tokenizer['attention_mask']
        # print(attention_mask)
        # print(attention_mask.shape)

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        gat_adj = torch.tensor(gat_adj)

        print(labels.shape)
        print(length.shape)
        print(gat_adj.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, gat_adj)

        return data

# def get_dataloader(train=True):
#     dataset = IMdbDataset(train)
#     data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
#     return data_loader

# print('1111111111')
# for idx, (input, target) in enumerate(get_dataloader()):
#     print(idx)
#     print(input)
#     print(target)
#     break
