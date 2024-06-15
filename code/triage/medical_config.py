import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129

DATA_DIR_Medical_label_1 = '/data0/wxy/bert/medical/data/label_1_cls_data/'
TRAIN_FILE_Medical_label_1 = 'label_1_train.csv'
TEST_FILE_Medical_label_1 = 'label_1_test.csv'
DATA_DIR_Medical_label_lite_1 = '/data0/wxy/bert/medical/data/label_1_lite_data/'
TRAIN_FILE_Medical_label_lite_1 = 'label_1_lite_train.csv'
TEST_FILE_Medical_label_lite_1 = 'label_1_lite_test.csv'
DATA_DIR_Medical_label_verylite_1 = '/data0/wxy/bert/medical/data/label_1_verylite_data/'
TRAIN_FILE_Medical_label_verylite_1 = 'label_1_verylite_train.csv'
TEST_FILE_Medical_label_verylite_1 = 'label_1_verylite_test.csv'

DATA_DIR_Medical_label_3 = '/data0/wxy/bert/medical/data/label_3_cls_data/'
TRAIN_FILE_Medical_label_3 = 'label_3_train.csv'
TEST_FILE_Medical_label_3 = 'label_3_test.csv'
DATA_DIR_Medical_label_verylite_3 = "/data0/wxy/bert/medical/data/label_3_verylite_data"
TRAIN_FILE_Medical_label_verylite_3 = "label_3_verylite_train.csv"
TEST_FILE_Medical_label_verylite_3 = "label_3_verylite_test.csv"

glove_cache_dir = "/data0/wxy/text classification/normal_deep_learning/glove_save"


class Medical_Config(object):
    def __init__(self):
        self.split = 'split10'

        self.bert_cache_path_Chinese = '/data0/wxy/bert/pre_trained_models/bert_model_Chinese'
        self.roberta_cache_path_Chinese = '/data0/wxy/bert/pre_trained_models/roberta_model_Chinese'
        self.pert_cache_path_Chinese = '/data0/wxy/bert/pre_trained_models/pert_model_Chinese'
        self.macbert_cache_path_Chinese = '/data0/wxy/bert/pre_trained_models/macbert_model_Chinese'

        self.electra_cache_path_Chinese = '/data0/wxy/bert/pre_trained_models/electra'
        self.CirBERTa_path_Chinese = '/data0/wxy/bert/pre_trained_models/CirBERTa'


        self.pretrain_bert_all_final = '/data0/wxy/bert/medical/pretrain/bert/all/final'
        self.pretrain_bert_all_more_test = '/data0/wxy/bert/medical/pretrain/bert/all/more_test'
        self.pretrain_macbert_all = '/data0/wxy/bert/medical/pretrain/macbert/all'
        self.pretrain_macbert_more_41 = '/data0/wxy/bert/medical/pretrain/macbert/more/41'
        self.pretrain_macbert_more_final = '/data0/wxy/bert/medical/pretrain/macbert/more/final'
        self.pretrain_roberta_all = '/data0/wxy/bert/medical/pretrain/roberta/all'


        self.feat_dim = 768

        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.embedding_size = 256
        self.pieces_size = 8

        self.split_len = 125
        self.overlap_len = 0

        self.epochs = 3
        self.lr = 5e-5
        self.other_lr = 2e-4
        self.batch_size = 48
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.bert_output_dim = 768
        self.num_classes = 14  # 120
        self.max_length = 256
        self.lstm_hidden_dim = 1024

        self.focal_loss_gamma = 2.
        self.focal_loss_alpha = None

        # GAT part
        self.gat_dropout = 0.5
        self.gat_alpha = 0.2
        self.gat_head = 8
        self.gat_adj_threshold = 0.97

        self.save_path = "/data0/wxy/bert/medical/best_model_save/model.ckpt"
        self.log_save_path = "/data0/wxy/bert/medical/train_result_save/train_result.txt"

        # Normal part
        self.normal_batch_size = 32
        self.LSTM_hidden_size = 512
        self.RNN_hidden_dim = 256
        self.GloVe_embedding_length = 300
        self.lr_NN = 0.01
        self.jieba_embedding_size = 200
        self.nn_max_length = 256


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
