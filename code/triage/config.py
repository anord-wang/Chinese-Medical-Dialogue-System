import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
# DATA_DIR = '/data1/mx/data/gxy_2016_data2/'
DATA_DIR = '/data0/wxy/bert/aclImdb/'
DATA_DIR_AG_NEWS = '/data0/wxy/bert/ag_news_csv/'
DATA_DIR_WOS = '/data0/wxy/bert/WOS/WOS11967/'
DATA_DIR_OH = '/data0/wxy/bert/ohmused/'
TRAIN_FILE_AG_NEWS = 'train.csv'
TEST_FILE_AG_NEWS = 'test.csv'
TRAIN_FILE_WOS = 'train.csv'
TEST_FILE_WOS = 'test.csv'
FINAL_FILE_WOS = 'finals.csv'
TRAIN_FILE_OH = 'oh-train.csv'
TEST_FILE_OH = 'oh-test.csv'
# TRAIN_FILE = 'train_texts.txt'
TRAIN_FILE = 'train/'
VALID_FILE = 'test/'
TEST_FILE = 'test/'
DATA_DIR_NS = '/data0/wxy/bert/native_speaker/'
TRAIN_FILE_NS = 'new_native_speaker3_train_8515.csv'
TEST_FILE_NS = 'new_native_speaker3_test_8515.csv'

DATA_DIR_yanbao = '/data0/wxy/bert/yanbao/'
TRAIN_FILE_yanbao = 'yanbao_train.txt'
TEST_FILE_yanbao = 'yanbao_test.txt'
# TRAIN_FILE_yanbao = 'yanbao_train_raw.txt'
# TEST_FILE_yanbao = 'yanbao_test_raw.txt'
TRAIN_FILE_yanbao_summary_25 = 'train_summary_50.txt'
TEST_FILE_yanbao_summary_25 = 'test_summary_50.txt'
TRAIN_FILE_yanbao_summary_50 = 'train_summary_50.txt'
TEST_FILE_yanbao_summary_50 = 'test_summary_50.txt'
TRAIN_FILE_yanbao_summary_75 = 'train_summary_75.txt'
TEST_FILE_yanbao_summary_75 = 'test_summary_75.txt'
TRAIN_FILE_yanbao_summary_100 = 'train_summary_100.txt'
TEST_FILE_yanbao_summary_100 = 'test_summary_100.txt'
TRAIN_FILE_yanbao_summary_75_new = 'train_summary_75.txt'
TEST_FILE_yanbao_summary_75_new = 'test_summary_75.txt'

# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'

glove_cache_dir = "/data0/wxy/text classification/normal_deep_learning/glove_save"


class Config(object):
    def __init__(self):
        self.split = 'split10'

        # self.bert_cache_path = "/data1/mx/program/paper3/bert_based_smp/bert_model/"
        # self.bert_cache_path = '/data0/wxy/bert/bert_model'
        self.bert_cache_path = '/data0/wxy/bert/pre_trained_models/bert_model_english'
        self.bert_cache_path_Chinese = '/data0/wxy/bert/pre_trained_models/bert_model_Chinese'
        self.roberta_cache_path_Chinese = '/data0/wxy/bert/pre_trained_models/roberta_model_Chinese'

        self.roberta_cache_path_english = '/data0/wxy/bert/pre_trained_models/roberta_model_english'
        self.feat_dim = 768

        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.embedding_size = 128
        self.pieces_size = 8

        self.split_len = 125
        self.overlap_len = 0

        self.epochs = 15
        self.lr = 2e-5
        self.other_lr = 2e-4
        self.batch_size = 8
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.bert_output_dim = 768
        self.num_classes = 2
        self.max_length = 128
        self.max_length_summary = 128
        self.lstm_hidden_dim = 1024

        self.focal_loss_gamma = 2.
        self.focal_loss_alpha = None

        # GAT part
        self.gat_dropout = 0.5
        self.gat_alpha = 0.2
        self.gat_head = 8
        self.gat_adj_threshold = 0.97

        self.save_path = "/data0/wxy/bert/bert_model_save/bert_model.ckpt"
        self.log_save_path = "/data0/wxy/bert/bert_model_save/train_result.txt"

        # Normal part
        self.normal_batch_size = 32
        self.LSTM_hidden_size = 256
        self.RNN_hidden_dim = 256
        self.GloVe_embedding_length = 300
        self.lr_NN = 0.001




class f_Config(object):
    def __init__(self):
        self.split = 'split10'

        self.bert_cache_path = "/data1/mx/program/paper3/bert_based_smp/bert_model/"
        self.feat_dim = 768

        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 50
        self.lr = 0.001
        self.batch_size = 4
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.bert_output_dim = 768
        self.num_classes = 6
        self.max_length = 512

        self.save_path = "bert_model/bert_model.ckpt"
