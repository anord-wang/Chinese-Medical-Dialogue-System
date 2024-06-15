import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
# import numpy as np
# import torch
# from config import *
from medical_dataloader import *
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from medical_model import *

from scipy.stats import pearsonr
import time
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def pearsonr_correlation_coefficient(pre, rea):
    if len(pre) != len(rea):
        print('error!!!!')
    result = np.zeros(len(pre))
    for i in range(len(pre)):
        result[i] = pearsonr(pre[i], rea[i])[0]
        if result[i] != result[i]:
            result[i] = 0
    average_average = sum(result) / len(pre)
    return average_average


def func(s):
    s = re.sub(r'\d+.', '', s)
    return s


def get_group_parameters(model, lr, lr_NN):
    params = list(model.named_parameters())
    no_decay = ['bias,', 'LayerNorm']
    other = ['lstm_layer', 'fc_layer', 'label', 'fcweight', 'W_sweight', 'W_sbias']
    no_main = no_decay + other

    param_group = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_main)], 'weight_decay': 1e-2, 'lr': lr},
        {'params': [p for n, p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0, 'lr': lr},
        {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0, 'lr': lr_NN},
        {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2, 'lr': lr_NN},
    ]
    return param_group


def run(configs):
    # initialize
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministics = True

    # load data
    train_loader = build_train_data(configs)
    # valid_loader = build_inference_data(configs, data_type='valid')
    test_loader = build_inference_data(configs, data_type='test')

    # load model
    # model = bert_model(configs)
    # model = roberta_model(configs)
    # model = pert_model(configs)
    # model = macbert_model(configs)
    # model = bert_RCNN_model(configs)
    # model = bert_RCNN_cat_model(configs)
    # model = bert_RCNN_cat_DD_model(configs)
    # model = bert_Self_Attn_model(configs)
    # model = bert_Self_Attn_cat_DD_model(configs)
    # model = bert_LSTM_Attn_cat_DD_model(configs)
    model = bert_LSTM_cat_DD_model(configs)
    # model = RoBERTa_LSTM_cat_DD_model(configs)
    # model = electra_LSTM_cat_DD_model(configs)


    # model = LSTMClassifier(configs)
    # model = SelfAttention(configs)
    # model = AttentionModel(configs)
    # model = RNN(configs)
    # model = RCNN(configs)
    # model = CNN(configs)

    # 设置分布训练
    # device_ids = [1,2]
    model = model.cuda()
    # torch.cuda.set_device(1)
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # don't understand this part?
    # params_bert = model.bert.parameters()
    # params_rest = list(model.fc.parameters())
    # assert sum([param.nelement() for param in params]) == \
    #        sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])
    # no_decay = ['bias', 'LayerNorm.weight']
    # params = [
    #     {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
    #     {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0, 'eps': configs.adam_epsilon},
    #     {'params': params_rest,
    #      'weight_decay': configs.l2}
    # ]

    ####################################################################################################################
    ##################################################设置学习率的几种方法##################################################

    # # 分层设置学习率，bert的学习率为configs.lr，其他部分学习率为configs.other_lr
    # optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'bert' in n]},
    #                                 {'params': [p for n, p in model.named_parameters() if 'bert' not in n],
    #                                  'lr': configs.other_lr}]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=configs.lr, eps=configs.adam_epsilon)
    # # 分层设置学习率结束

    bert_test = model
    params_list = [n for n, p in list(bert_test.named_parameters())]
    params_list = set(map(func, params_list))
    print(params_list)

    # 正常一次设置
    no_decay = ['bias', 'gamma', 'beta']
    params = model.parameters()
    optimizer = AdamW(params, lr=configs.lr)
    # optimizer = AdamW(params, lr=configs.lr_NN)
    # param_group = get_group_parameters(model, configs.lr, configs.lr_NN)
    # optimizer = torch.optim.AdamW(param_group, lr=1e-5, eps=1e-7)  # 括号中为除了分组中的默认值

    # optimizer_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay_rate': 0.0}
    # ]
    # optimizer = AdamW(optimizer_parameters, lr=configs.lr)

    # 正常一次设置结束

    #

    #

    ################################################设置学习率变化的几种方法################################################
    # 线性warmup然后线性衰减
    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)  # why warm up?
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_steps_all)
    # 结束

    # # 按照lamda函数进行变化
    # my_lambda = lambda epoch: np.sin(epoch) / epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=my_lambda)
    # # 结束
    #
    # # StepLR–阶梯式衰减,每step_size的epoch，lr会自动乘以gamma进行阶梯式衰减。
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    # # 结束
    #
    # # MultiStepLR–多阶梯式衰减,该衰减为三段式，milestones确定两个阈值，epoch进入milestones范围内即乘以gamma，离开milestones范围之后再乘以gamma。
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80], gamma=0.9)
    # # 结束
    #
    # # ExponentialLR–指数式衰减,每个epoch都乘以gamma
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # # 结束
    #
    # # ReduceLROnPlateau,在发现loss不再降低或者acc不再提高之后，降低学习率。
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
    #                                                        verbose=False, threshold=0.0001, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=0, eps=1e-08)
    # # 结束

    ####################################################设置学习率结束####################################################
    ####################################################################################################################

    model.zero_grad()
    early_stop_flag = None

    total_batch = 0
    start_time = time.time()
    dev_best_loss = float('inf')
    dev_best_acc = 0
    dev_best_f1 = 0
    dev_best_recall = 0
    dev_best_precision = 0
    dev_best_roc_auc = 0
    dev_best_ap = 0

    for epoch in range(configs.epochs):
        for train_step, batch in enumerate(train_loader, 1):
            model.train()

            # train_features, train_token_type, train_bert_mask, train_preds, train_length, train_gat_adj = batch
            # train_features, train_token_type, train_bert_mask, train_preds, train_length = batch
            # BERT
            train_features, train_token_type, train_bert_mask, train_preds = batch
            # train_features, train_token_type, train_bert_mask, train_preds, train_length, train_features_summary, train_token_type_summary, train_bert_mask_summary = batch
            # train_features, train_preds = batch
            # RoBERTa
            # train_features, train_bert_mask, train_preds = batch

            # plus1
            # train_features_summary = train_features_summary.to(DEVICE)
            # train_token_type_summary = train_token_type_summary.to(DEVICE)
            # train_bert_mask_summary = train_bert_mask_summary.to(DEVICE)
            # plus1end

            # print(train_preds.shape)
            train_features = train_features.to(DEVICE)
            # train_features = train_features
            train_token_type = train_token_type.to(DEVICE)
            # train_token_type = train_token_type
            train_bert_mask = train_bert_mask.to(DEVICE)
            # train_bert_mask = train_bert_mask
            train_preds = train_preds.to(DEVICE)
            # train_preds = train_preds
            # train_length = get_length_matrix(train_length, configs.batch_size).to(DEVICE)
            # train_length = train_length.to(DEVICE)
            # train_gat_adj = train_gat_adj.to(DEVICE)

            # train_length = train_length
            # print(train_preds.shape)
            # output_preds = model(train_features, train_bert_mask, train_token_type, train_length, train_gat_adj)
            # output_preds = model(train_features, train_bert_mask, train_token_type, train_length)
            # BERT
            output_preds = model(train_features, train_bert_mask, train_token_type)
            # output_preds = model(train_features, train_bert_mask, train_token_type, train_length, train_features_summary, train_bert_mask_summary, train_token_type_summary)
            # output_preds = model(train_features)
            # RoBERTa
            # output_preds = model(train_features, train_bert_mask)

            # print(output_preds.shape)
            # output_preds = torch.topk(output_preds, 1)[1].squeeze()
            # train_preds = torch.topk(train_preds, 1)[1].squeeze()
            # print(output_preds.shape)
            # print(train_preds.shape)
            # 交叉熵损失
            entroy = nn.CrossEntropyLoss()
            loss = entroy(output_preds, train_preds)
            # 焦点损失
            # focal = FocalLoss(configs.num_classes, alpha=configs.focal_loss_alpha, gamma=configs.focal_loss_gamma)
            # loss = focal(output_preds, train_preds)
            # loss = model.loss_preds(output_preds, train_preds)
            # loss = prob_cross_entropy_loss(output_preds, train_preds)
            loss.backward()

            # 修剪梯度进行归一化防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            # torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters, 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # true = labels.data.cpu()
                output_preds = F.softmax(output_preds, dim=1)
                print('output_preds')
                print(output_preds)
                train_AP = pearsonr_correlation_coefficient(output_preds.data.cpu().numpy(),
                                                            train_preds.data.cpu().numpy())

                # true = torch.max(train_preds.data, 1)[1].cpu()
                true = train_preds.data.cpu()
                predic = torch.max(output_preds.data, 1)[1].cpu()
                print(predic)
                print(true)

                train_acc = metrics.accuracy_score(true, predic)
                # train_f1 = metrics.f1_score(true, predic)
                train_f1 = metrics.f1_score(true, predic, average='macro')
                # train_precision = metrics.precision_score(true, predic)
                train_precision = metrics.precision_score(true, predic, average='macro')
                # train_recall = metrics.recall_score(true, predic)
                train_recall = metrics.recall_score(true, predic, average='macro')
                try:
                    train_roc_auc = metrics.roc_auc_score(true, predic)
                except ValueError:
                    train_roc_auc = 0
                # train_roc_auc = metrics.roc_curve(true, predic)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {},  Train Loss: {},  Train Acc: {},  Train F1: {},  Train Precision: {},  Train Recall: {},  Train ROC_AUC: {},  Time: {}'
                file = open(configs.log_save_path, 'a', encoding='utf-8')
                print(msg.format(total_batch, loss.item(), train_acc, train_f1, train_precision, train_recall,
                                 train_roc_auc, time_dif), file=file)
                file.close()
                print(msg.format(total_batch, loss.item(), train_acc, train_f1, train_precision, train_recall,
                                 train_roc_auc, time_dif))
            total_batch += 1

        with torch.no_grad():
            model.eval()

            dev_loss, dev_acc, dev_f1, dev_precision, dev_recall, train_classification_report, labels_all, predict_all = evaluate(
                configs, model, test_loader)
            if epoch == 0:
                a = np.array(labels_all)
                np.save('/data0/wxy/bert/medical/predict_result_save/labels_all.npy', a)
                # a = np.load('a.npy')
                # a = a.tolist()

            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(model.state_dict(), configs.save_path)
                improve = '*'
                last_improve = total_batch
            else:
                improve = ''

            if dev_acc > dev_best_acc:
                dev_best_acc = dev_acc
                b = np.array(predict_all)
                np.save('/data0/wxy/bert/medical/predict_result_save/predict_all.npy', b)

            if dev_recall > dev_best_recall:
                dev_best_recall = dev_recall

            if dev_precision > dev_best_precision:
                dev_best_precision = dev_precision

            if dev_f1 > dev_best_f1:
                dev_best_f1 = dev_f1

            # if dev_roc_auc > dev_best_roc_auc:
            #     dev_best_roc_auc = dev_roc_auc

            time_dif = get_time_dif(start_time)
            msg = 'Iter: {},  Dev Loss: {},  Dev Acc: {},  Dev F1: {},  Dev Precision: {},  Dev Recall: {},  Time: {}{}'
            print(msg.format(epoch, dev_loss.item(), dev_acc, dev_f1, dev_precision, dev_recall,
                             time_dif, improve))
            file = open(configs.log_save_path, 'a', encoding='utf-8')
            print(msg.format(epoch, dev_loss.item(), dev_acc, dev_f1, dev_precision, dev_recall,
                             time_dif, improve), file=file)
            print('train_classification_report:', file=file)
            print(train_classification_report, file=file)
            file.close()

    print('best dev recall:', dev_best_recall)
    print('best dev roc_auc:', dev_best_roc_auc)
    print('best dev precision:', dev_best_precision)
    print('best dev f1:', dev_best_f1)
    print('best dev acc:', dev_best_acc)
    file = open(configs.log_save_path, 'a', encoding='utf-8')
    print('best dev recall:', dev_best_recall, file=file)
    print('best dev roc_auc:', dev_best_roc_auc, file=file)
    print('best dev precision:', dev_best_precision, file=file)
    print('best dev f1:', dev_best_f1, file=file)
    print('best dev acc:', dev_best_acc, file=file)
    file.close()


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    f_predict_all = np.array([], dtype=int)
    f_labels_all = np.array([], dtype=int)
    predict_prob = []
    labels_prob = []
    f_predict_prob = []
    f_labels_prob = []
    with torch.no_grad():
        # for features, token_type, bert_mask, labels, length, gat_adj in data_iter:
        # for features, token_type, bert_mask, labels, length in data_iter:
        for features, token_type, bert_mask, labels in data_iter:
        # for features, bert_mask, labels in data_iter:
        # for features, token_type, bert_mask, labels, length, features_summary, token_type_summary, bert_mask_summary in data_iter:
        # for features, labels in data_iter:
        # for features, bert_mask, labels in data_iter:

            # plus1
            # features_summary = features_summary.to(DEVICE)
            # token_type_summary = token_type_summary.to(DEVICE)
            # bert_mask_summary = bert_mask_summary.to(DEVICE)
            # plus1end

            features = features.to(DEVICE)
            # features = features
            token_type = token_type.to(DEVICE)
            # token_type = token_type
            bert_mask = bert_mask.to(DEVICE)
            # bert_mask = bert_mask
            labels = labels.to(DEVICE)
            # labels = labels
            # length = get_length_matrix(length, configs.batch_size).to(DEVICE)
            # length = get_length_matrix(length, configs.batch_size)
            # length = length.to(DEVICE)
            # gat_adj = gat_adj.to(DEVICE)
            # outputs = model(features, bert_mask, token_type, length, gat_adj)
            # outputs = model(features, bert_mask, token_type, length)
            # BERT
            outputs = model(features, bert_mask, token_type)
            # outputs = model(features, bert_mask, token_type, length, features_summary, bert_mask_summary, token_type_summary)
            # outputs = model(features)
            # RoBERTa
            # outputs = model(features, bert_mask)

            entroy = nn.CrossEntropyLoss()
            loss = entroy(outputs, labels)
            # focal = FocalLoss(configs.num_classes, alpha=configs.focal_loss_alpha, gamma=configs.focal_loss_gamma)
            # loss = focal(outputs, labels)
            # loss = model.loss_preds(outputs, labels)
            # loss = prob_cross_entropy_loss(outputs, labels)
            loss_total += loss

            # coarse
            outputs = F.softmax(outputs, dim=1)
            predict_prob = predict_prob + outputs.cpu().numpy().tolist()
            labels_prob = labels_prob + labels.cpu().numpy().tolist()
            # labels = torch.max(labels.data, 1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            # # fine_grained
            # f_predict_prob = f_predict_prob + fine_grained_outputs.cpu().numpy().tolist()
            # f_labels_prob = f_labels_prob + fine_grained_labels.cpu().numpy().tolist()
            #
            # f_labels = torch.max(fine_grained_labels.data, 1)[1].cpu().numpy()
            # # labels = labels.data.cpu().numpy()
            # f_predic = torch.max(fine_grained_outputs.data, 1)[1].cpu().numpy()
            #
            # f_labels_all = np.append(f_labels_all, f_labels)
            # f_predict_all = np.append(f_predict_all, f_predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    # f1 = metrics.f1_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    # precision = metrics.precision_score(labels_all, predict_all)
    precision = metrics.precision_score(labels_all, predict_all, average='macro')
    # recall = metrics.recall_score(labels_all, predict_all)
    recall = metrics.recall_score(labels_all, predict_all, average='macro')
    # roc_auc = metrics.roc_auc_score(labels_all, predict_all)
    try:
        roc_auc = metrics.roc_auc_score(labels_all, predict_all, average='macro')
    except ValueError:
        roc_auc = 0
    ap = pearsonr_correlation_coefficient(predict_prob, labels_prob)
    target_names = ['肿瘤科', '整形美容科', '康复医学科', '心理健康科', '儿科', '妇产科', '传染科', '皮肤性病科', '辅助检查科', '内科', '外科',
                    '五官科', '营养保健科', '中医科']
    train_classification_report = metrics.classification_report(labels_all, predict_all, target_names=target_names,
                                                                digits=4)

    # f_acc = metrics.accuracy_score(f_labels_all, f_predict_all)
    # f_ap = pearsonr_correlation_coefficient(f_predict_prob, f_labels_prob)
    if test:
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, f1, precision, recall, roc_auc, loss_total / len(data_iter), confusion
    return loss_total / len(data_iter), acc, f1, precision, recall, train_classification_report, labels_all, predict_all


if __name__ == '__main__':
    # configs = Config()
    configs = Medical_Config()

    run(configs)
