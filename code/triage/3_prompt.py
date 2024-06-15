import warnings
from datetime import datetime
import time
import torch
import os
from transformers import BertModel, BertConfig, BertModel, BertTokenizerFast, get_cosine_schedule_with_warmup, \
    BertForMaskedLM
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# hyperparameters
EPOCH = 20
RANDOM_SEED = 2022
TRAIN_BATCH_SIZE = 8  # 小批训练， 批大小增大时需要提升学习率  https://zhuanlan.zhihu.com/p/413656738
TEST_BATCH_SIZE = 24  # 大批测试
EVAL_PERIOD = 20
MODEL_NAME = '/data0/wxy/bert/pre_trained_models/bert_model_Chinese'  # bert-base-chinese
# MODEL_NAME = '/data0/wxy/bert/medical/pretrain/bert/all/final'

# DATA_PATH = "/data0/wxy/bert/medical/data/label_3_cls_data"
# train_file = "label_3_train.csv"
# test_file = "label_3_test.csv"

# DATA_PATH = "/data0/wxy/bert/medical/data/label_1_lite_data"
# train_file = "label_1_lite_train.csv"
# test_file = "label_1_lite_test.csv"

DATA_PATH = "/data0/wxy/bert/medical/data/label_3_verylite_data"
train_file = "label_3_verylite_train.csv"
test_file = "label_3_verylite_test.csv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# log_save_path = "/data0/wxy/bert/medical/train_result_save/3_promote_result_repretrain.txt"
# log_save_path = "/data0/wxy/bert/medical/train_result_save/3_promote_verylite_result.txt"
log_save_path = "/data0/wxy/bert/medical/train_result_save/3_promote_verylite_result_original.txt"

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# label_name = ['胃肠疾病', '妇科炎症', '月经不调', '肛肠科', '白癜风', '性功能障碍', '癫痫', '牙周病', '甲状腺疾病', '皮肤过敏', '乳腺外科', '尖锐湿疣', '糖尿病',
#               '脊髓脊柱外科', '高血压', '小儿呼吸内科', '外伤科', '牛皮癣', '口腔整形', '艾滋病', '头颈外科', '脱发', '宫颈糜烂', '口腔粘膜病', '足踝外科', '骨关节病',
#               '新生儿内科', '鼻炎', '前列腺疾病', '湿疹', '血管外科', '包皮包茎', '瘢痕疙瘩', '烧伤烫伤', '肝胆疾病', '心律失常', '脑出血', '神经衰弱', '咽喉炎',
#               '子宫肌瘤', '梅毒', '尿路感染', '口臭', '肾结石', '贫血', '肺癌', '脑梗塞', '耳鸣', '灰指甲', '腋臭', '股骨头坏死', '荨麻疹', '冠心病', '胃肠外科',
#               '痛风', '甲沟炎', '生殖器疱疹', '心外科', '风湿', '肝胆外科', '带状疱疹', '扁平疣', '类风湿', '胸外科', '强直性脊柱炎', '胃癌', '肝癌', '耳聋', '食管癌',
#               '面瘫', '非淋性尿道炎', '多囊卵巢综合症', '宫外孕', '心血管疾病', '鼻窦炎', '尿失禁', '哮喘', '宫颈癌', '妇科内分泌', '感冒', '小儿神经内科', '直肠癌',
#               '小儿神经康复科', '脑血管疾病', '白血病', '龋齿', '泌尿系结石', '红斑狼疮', '青光眼', '附件炎', '脂溢性皮炎', '拇外翻', '心肌病', '帕金森病', '肾功能衰竭',
#               '鱼鳞病', '三叉神经痛', '结肠癌', '肾炎', '神经心理专科', '视网膜病', '白内障', '尿道下裂', '口腔外科', '唇腭裂', '鼻息肉', '更年期综合征', '唇炎',
#               '重症肌无力', '斜视', '血小板紫癜', '淋巴水肿', '小儿心胸外科', '骨髓增生异常', '骨髓炎', '脑瘫', '小儿消化内科', '小儿血液科', '妊高征', '再障贫血', '肺炎',
#               '脉管炎', '其它疾病', '气管炎', '硬皮病', '干燥综合症', '雷诺氏病', '乳腺癌', '淋病', '泌尿系统感染', '早产', '奥美定', '避孕', '口腔溃疡', '神经遗传病',
#               '心血管介入', '肌无力', '人工流产', '血小板减少症', '风湿性心脏病', '手足癣', '近视眼', '瘙痒症', '疥疮', '阴道炎', '沙眼', '子宫内膜炎', '早搏', '色盲',
#               '膀胱炎', '舌炎', '酒渣鼻', '视神经萎缩', '血液检查', '五官科偏方', '斑秃', '药物存放']
label_name = ['子宫肌瘤', '梅毒', '尿路感染', '口臭', '肾结石', '贫血', '肺癌', '脑梗塞', '耳鸣', '灰指甲', '腋臭', '股骨头坏死', '荨麻疹', '冠心病', '胃肠外科',
              '痛风', '甲沟炎', '生殖器疱疹', '心外科', '风湿', '肝胆外科', '带状疱疹', '扁平疣', '类风湿', '胸外科', '强直性脊柱炎', '胃癌', '肝癌', '耳聋', '食管癌',
              '面瘫', '非淋性尿道炎', '多囊卵巢综合症', '宫外孕', '心血管疾病', '鼻窦炎', '尿失禁', '哮喘', '宫颈癌', '妇科内分泌', '感冒', '小儿神经内科', '直肠癌',
              '小儿神经康复科', '脑血管疾病', '白血病', '龋齿', '泌尿系结石', '红斑狼疮', '青光眼', '附件炎', '脂溢性皮炎', '拇外翻', '心肌病', '帕金森病', '肾功能衰竭',
              '鱼鳞病', '三叉神经痛', '结肠癌', '肾炎', '神经心理专科', '视网膜病', '白内障', '尿道下裂', '口腔外科', '唇腭裂', '鼻息肉', '更年期综合征', '唇炎',
              '重症肌无力', '斜视', '血小板紫癜', '淋巴水肿', '小儿心胸外科', '骨髓增生异常', '骨髓炎', '脑瘫', '小儿消化内科', '小儿血液科', '妊高征', '再障贫血', '肺炎',
              '脉管炎', '其它疾病', '气管炎', '硬皮病', '干燥综合症', '雷诺氏病', '乳腺癌', '淋病', '泌尿系统感染', '早产', '奥美定', '避孕', '口腔溃疡', '神经遗传病',
              '心血管介入', '肌无力', '人工流产', '血小板减少症', '风湿性心脏病', '手足癣', '近视眼', '瘙痒症', '疥疮', '阴道炎', '沙眼', '子宫内膜炎', '早搏', '色盲',
              '膀胱炎', '舌炎', '酒渣鼻', '视神经萎缩', '血液检查', '五官科偏方', '斑秃', '药物存放']
# label_name = ['儿科[padding][padding][padding]', '整形美容科', '心理健康科', '传染科[padding][padding]',
#               '美容[padding][padding][padding]', '中医科[padding][padding]', '肿瘤科[padding][padding]',
#               '药品[padding][padding][padding]', '其他[padding][padding][padding]',  '运动瘦身[padding]', '康复医学科',
#               '家居环境[padding]', '遗传[padding][padding][padding]', '保健养生[padding]', '辅助检查科', '营养保健科',
#               '体检科[padding][padding]', '子女教育[padding]']
# label_name = ['心理健康科', '肿瘤科[padding][padding]', '康复医学科', '整形美容科', '传染科[padding][padding]',
#               '中医科[padding][padding]', '营养保健科', '儿科[padding][padding][padding]', '辅助检查科']

prefix = '这可能是[mask][mask][mask][mask][mask][mask][mask]相关的病症。'


class Bert_Model(nn.Module):
    def __init__(self, bert_path, config_file):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path, config=config_file)  # 加载预训练模型权重

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask,
                            token_type_ids)  # masked LM 输出的是 mask的值 对应的ids的概率 ，输出 会是词表大小，里面是概率
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]

        return logit

    # 构建数据集


class MyDataSet(Data.Dataset):
    def __init__(self, sen, mask, typ, label):
        super(MyDataSet, self).__init__()
        self.sen = torch.tensor(sen, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long)
        self.typ = torch.tensor(typ, dtype=torch.long)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.sen.shape[0]

    def __getitem__(self, idx):
        return self.sen[idx], self.mask[idx], self.typ[idx], self.label[idx]


# load  data

def load_data(tsvpath):
    # data = pd.read_csv(tsvpath, sep="\t", header=None, names=["sn", "polarity", "text"])
    # data = data[data["polarity"] != "neutral"]
    # yy = data["polarity"].replace({"negative": 0, "positive": 1, "neutral": 2})

    # return data.values[:, 2:3].tolist(), yy.tolist()  # data.values[:,1:2].tolist()
    # titles = []
    infos = []
    labels = []
    data = pd.read_csv(tsvpath, encoding='gbk', header=0, low_memory=False)  # 防止弹出警告
    df = pd.DataFrame(data)
    text_len = df.shape[0]
    print(df.shape)
    for i in range(text_len):
        title = str(df.iloc[i, 0])
        info = str(df.iloc[i, 1])
        # titles.append(title)
        text = '这可能是[mask][mask][mask][mask][mask][mask][mask]相关的病症，' + info
        infos.append(text)

        label = int(df.iloc[i, 3])
        label_one = label_name[label] + '[padding]' * (7 - len(label_name[label]))
        label = '这可能是' + label_one + '相关的病症，' + info
        labels.append(label)

    # c_labels = np.array(labels)

    return infos, labels


tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

config = BertConfig.from_pretrained(MODEL_NAME)

model = Bert_Model(bert_path=MODEL_NAME, config_file=config).to(device)


# pos_id = tokenizer.convert_tokens_to_ids("good")  # 9005
# neg_id = tokenizer.convert_tokens_to_ids("bad")  # 12139


# get the data and label

def ProcessData(filepath):
    x_train, y_train = load_data(DATA_PATH + os.sep + filepath)
    # x_train,x_test,y_train,y_test=train_test_split(StrongData,StrongLabel,test_size=0.3, random_state=42)

    Inputid = []
    Labelid = []
    typeid = []
    attenmask = []

    for i in range(len(x_train)):
        text_ = x_train[i]
        label_ = y_train[i]

        encode_dict = tokenizer.encode_plus(text_, max_length=300, padding="max_length", truncation=True)
        input_ids = encode_dict["input_ids"]
        type_ids = encode_dict["token_type_ids"]
        atten_mask = encode_dict["attention_mask"]
        inputid = input_ids[:]

        encode_label = tokenizer.encode_plus(label_, max_length=300, padding="max_length", truncation=True)
        labelid = encode_label["input_ids"]

        # inputid[MASK_POS] = tokenizer.mask_token_id

        Labelid.append(labelid)
        Inputid.append(inputid)
        typeid.append(type_ids)
        attenmask.append(atten_mask)

    return Inputid, Labelid, typeid, attenmask


Inputid_train, Labelid_train, typeids_train, inputnmask_train = ProcessData(train_file)
# Inputid_dev, Labelid_dev, typeids_dev, inputnmask_dev = ProcessData(dev_file)
Inputid_test, Labelid_test, typeids_test, inputnmask_test = ProcessData(test_file)

train_dataset = Data.DataLoader(MyDataSet(Inputid_train, inputnmask_train, typeids_train, Labelid_train),
                                TRAIN_BATCH_SIZE, True)
# valid_dataset = Data.DataLoader(MyDataSet(Inputid_dev, inputnmask_dev, typeids_dev, Labelid_dev), TRAIN_BATCH_SIZE,
#                                 True)
test_dataset = Data.DataLoader(MyDataSet(Inputid_test, inputnmask_test, typeids_test, Labelid_test), TEST_BATCH_SIZE,
                               True)

train_data_num = len(Inputid_train)
# print(train_data_num)
test_data_num = len(Inputid_test)
# print(test_data_num)

print("hello!")

optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # 使用Adam优化器
loss_func = nn.CrossEntropyLoss(ignore_index=-1)
EPOCH = 20
schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset),
                                           num_training_steps=EPOCH * len(train_dataset))
print("正在训练中。。。")
totaltime = 0
for epoch in range(EPOCH):

    starttime_train = datetime.now()

    start = time.time()
    correct = 0
    train_loss_sum = 0
    model.train()

    for idx, (ids, att_mask, type, y) in enumerate(train_dataset):
        ids, att_mask, type, y = ids.to(device), att_mask.to(device), type.to(device), y.to(device)
        out_train = model(ids, att_mask, type)
        # print(out_train.view(-1, tokenizer.vocab_size).shape, y.view(-1).shape)
        loss = loss_func(out_train.view(-1, tokenizer.vocab_size), y.view(-1))
        # print(out_train.view(-1, tokenizer.vocab_size))
        # print(y.view(-1))
        # print('loss in process', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        train_loss_sum += loss.item()

        if (idx + 1) % EVAL_PERIOD == 0:
            print("Epoch {:04d} | Step {:06d}/{:06d} | Loss {:.4f} | Time {:.0f}".format(
                epoch + 1, idx + 1, len(train_dataset), train_loss_sum / (idx + 1), time.time() - start))
            # writer.add_scalar('loss/train_loss', train_loss_sum / (idx + 1), epoch)
            file = open(log_save_path, 'a', encoding='utf-8')
            print('loss/train_loss', train_loss_sum / (idx + 1), epoch, file=file)
            print("Epoch {:04d} | Step {:06d}/{:06d} | Loss {:.4f} | Time {:.0f}".format(
                epoch + 1, idx + 1, len(train_dataset), train_loss_sum / (idx + 1), time.time() - start), file=file)
            file.close()

        # truelabel = y[:, MASK_POS]
        truelabel = y[:, 5:10]
        # print(truelabel)
        # print(truelabel.shape)
        # out_train_mask = out_train[:, MASK_POS, :]
        out_train_mask = out_train[:, 5:10, :]
        # print(out_train_mask)
        # print(out_train_mask.shape)
        predicted = torch.max(out_train_mask, 2)[1]
        # print(predicted)
        # print(predicted.shape)
        for bbb in range(predicted.shape[0]):
            if predicted[bbb].equal(truelabel[bbb]):
                # predicted[bbb] == truelabel[bbb]
                correct += 1
        print(correct)
        correct = float(correct)

    acc = float(correct / train_data_num)

    eval_loss_sum = 0.0
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for ids, att, tpe, y in test_dataset:
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            out_test = model(ids, att, tpe)
            loss_eval = loss_func(out_test.view(-1, tokenizer.vocab_size), y.view(-1))
            eval_loss_sum += loss_eval.item()
            # ttruelabel = y[:, MASK_POS]
            ttruelabel = y[:, 5:10]
            # tout_train_mask = out_test[:, MASK_POS, :]
            tout_train_mask = out_test[:, 5:10, :]
            predicted_test = torch.max(tout_train_mask.data, 2)[1]
            # correct_test += (predicted_test == ttruelabel).sum()
            for bbb in range(predicted_test.shape[0]):
                if predicted_test[bbb].equal(ttruelabel[bbb]):
                    correct_test += 1
            correct_test = float(correct_test)
    acc_test = float(correct_test / test_data_num)

    if epoch % 1 == 0:
        out = ("epoch {}, train_loss {},  train_acc {} , eval_loss {} ,acc_test {}"
               .format(epoch + 1, train_loss_sum / (len(train_dataset)), acc, eval_loss_sum / (len(test_dataset)),
                       acc_test))
        # writer.add_scalar('loss/test_loss', train_loss_sum / (idx + 1), epoch)
        file = open(log_save_path, 'a', encoding='utf-8')
        print('loss/test_loss', train_loss_sum / (idx + 1), epoch, file=file)
        print(out, file=file)
        file.close()
        print(out)
    end = time.time()

    print("epoch {} duration:".format(epoch + 1), end - start)
    file = open(log_save_path, 'a', encoding='utf-8')
    print("epoch {} duration:".format(epoch + 1), end - start, file=file)
    file.close()
    totaltime += end - start

print("total training time: ", totaltime)
file = open(log_save_path, 'a', encoding='utf-8')
print("total training time: ", totaltime, file=file)
file.close()
