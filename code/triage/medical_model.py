from transformers import BertModel, BertConfig, RobertaModel,AutoModelForPreTraining,AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import GraphAttentionLayer
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class LSTMClassifier(nn.Module):
    def __init__(self, configs):
        super(LSTMClassifier, self).__init__()
        self.batch_size = configs.batch_size
        self.lstm_layer = nn.LSTM(configs.jieba_embedding_size, configs.LSTM_hidden_size, num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.fc1 = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

    def forward(self, lstm_input):
        lstm_input = lstm_input.float()
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(lstm_input)
        lstm_output = F.leaky_relu(self.fc1(output[:, -1, :]))

        return lstm_output


class SelfAttention(nn.Module):
    def __init__(self, configs):
        super(SelfAttention, self).__init__()
        self.batch_size = configs.batch_size
        self.dropout = 0.8
        self.lstm_layer = nn.LSTM(configs.GloVe_embedding_length, configs.LSTM_hidden_size, dropout=self.dropout,
                                  num_layers=1, batch_first=True, bidirectional=True)  # 双向LSTM
        self.fc1 = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

        self.W_s1 = nn.Linear(2 * configs.LSTM_hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30 * 2 * configs.LSTM_hidden_size, 2000)
        self.label = nn.Linear(2000, configs.num_classes)

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, lstm_input):
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(lstm_input)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        logits = self.label(fc_out)

        return logits


class AttentionModel(nn.Module):
    def __init__(self, configs):
        super(AttentionModel, self).__init__()
        self.batch_size = configs.batch_size
        self.lstm_layer = nn.LSTM(configs.GloVe_embedding_length, configs.LSTM_hidden_size,
                                  batch_first=True)
        self.fc1 = nn.Linear(configs.LSTM_hidden_size, configs.num_classes, bias=False)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        print(hidden.shape)
        print(hidden.unsqueeze(2).shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, lstm_input):
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(lstm_input)
        print(output.shape)
        print(final_hidden_state.shape)
        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.fc1(attn_output)
        return logits


class Self_DD_Attn_model(nn.Module):
    def __init__(self, configs):
        super(Self_DD_Attn_model, self).__init__()
        self.dropout = 0.8

        self.lstm_layer = nn.LSTM(configs.GloVe_embedding_length, configs.LSTM_hidden_size, dropout=self.dropout,
                                  num_layers=1, batch_first=True, bidirectional=True)  # 双向LSTM
        self.fc1 = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

        self.W_s11 = nn.Linear(2 * configs.LSTM_hidden_size, 350, bias=False)
        self.W_s22 = nn.Linear(350, 30, bias=False)
        self.dd11 = nn.Linear(350, 350, bias=False)
        self.dd22 = nn.Linear(30, 30, bias=False)

        self.fc_layer = nn.Linear(30 * 2 * configs.LSTM_hidden_size, 2000, bias=False)
        self.label = nn.Linear(2000, configs.num_classes, bias=False)

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s11(lstm_output)
        attn_weight_matrix_g = self.dd11(attn_weight_matrix)
        attn_weight_matrix = attn_weight_matrix_g * attn_weight_matrix + attn_weight_matrix_g
        attn_weight_matrix = self.W_s22(attn_weight_matrix)
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, lstm_input):
        # 通过self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(lstm_input)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        final_output = self.label(fc_out)

        # 拼接cls向量
        # last_hidden_state = bert_output[1]

        # 加入BERT不同层的embedding

        # 加入DD

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class RNN(nn.Module):
    def __init__(self, configs):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(configs.GloVe_embedding_length, configs.RNN_hidden_dim, num_layers=2, batch_first=True,
                          bidirectional=True)
        self.label = nn.Linear(4 * configs.RNN_hidden_dim, configs.num_classes)

    def forward(self, input_sentences):
        print(input_sentences.size())
        output, h_n = self.rnn(input_sentences)
        print(h_n.size())
        # h_n.size() = (batch_size, 4, hidden_size)
        h_n_1 = h_n.transpose(0, 1).contiguous()
        print(h_n_1.size())
        h_n_1 = h_n_1.contiguous().view(h_n_1.size()[0], h_n_1.size()[1] * h_n_1.size()[2])
        print(h_n_1.size())
        # h_n.size() = (batch_size, 4*hidden_size)
        logits = self.label(h_n_1)

        return logits


class RCNN(nn.Module):
    def __init__(self, configs):
        super(RCNN, self).__init__()
        self.dropout = 0.8
        self.lstm = nn.LSTM(configs.GloVe_embedding_length, configs.LSTM_hidden_size, dropout=self.dropout,
                            batch_first=True, bidirectional=True)
        self.W2 = nn.Linear(2 * configs.LSTM_hidden_size + configs.GloVe_embedding_length, configs.LSTM_hidden_size)
        self.label = nn.Linear(configs.LSTM_hidden_size, configs.num_classes)

    def forward(self, input_sentence, batch_size=None):
        # embedded input of shape = (batch_size, num_sequences, embedding_length)
        output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence)

        final_encoding = torch.cat((output, input_sentence), 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        logits = self.label(y)

        return logits


class RCNN_DD_model(nn.Module):
    def __init__(self, configs):
        super(RCNN_DD_model, self).__init__()

        self.dropout = 0.8
        self.lstm = nn.LSTM(configs.GloVe_embedding_length, configs.LSTM_hidden_size, dropout=self.dropout,
                            batch_first=True, bidirectional=True)
        self.W2 = nn.Linear(2 * configs.LSTM_hidden_size + configs.GloVe_embedding_length, configs.LSTM_hidden_size)
        self.dd = nn.Linear(configs.LSTM_hidden_size, configs.LSTM_hidden_size, bias=False)
        self.label = nn.Linear(configs.LSTM_hidden_size, configs.num_classes, bias=False)

    def forward(self, input_sentence):
        # # 利用BERT得到embedding
        # bert_output = self.bert(input_ids=bert_token,
        #                         attention_mask=bert_mask,
        #                         token_type_ids=token_type)  # not sure
        # pooler_output = bert_output[0]

        # 通过RCNN网络
        print(input_sentence.size())  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence)
        final_encoding = torch.cat((lstm_output, input_sentence), 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)

        # 拼接cls向量
        # last_hidden_state = bert_output[1]
        # y_cat = torch.cat([last_hidden_state, y], dim=1)

        # 加入BERT不同层的embedding

        # 加入DD
        d_out = y

        g1 = self.dd(d_out)
        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2

        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3

        final_output = self.label(d_out)

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.in_channels = 1
        self.out_channels = 8
        self.kernel_heights = [5, 5, 5]
        self.stride = (1, 1)
        self.padding = 0
        self.keep_probab = 0.5

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[0], configs.jieba_embedding_size), self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[1], configs.jieba_embedding_size), self.stride, self.padding)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[2], configs.jieba_embedding_size), self.stride, self.padding)
        self.dropout = nn.Dropout(self.keep_probab)
        self.label = nn.Linear(len(self.kernel_heights) * self.out_channels, configs.num_classes)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))
        # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input_sentences):
        # input_sentences.size() = (batch_size, num_seq, embedding_length)
        input_sentences = input_sentences.float()
        input = input_sentences.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)

        return logits


class bert_model(nn.Module):
    def __init__(self, configs):
        super(bert_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert = BertModel.from_pretrained(configs.pretrain_bert_all_final)
        # self.bert = RobertaModel.from_pretrained(configs.roberta_cache_path_english)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

    def forward(self, bert_token, bert_mask, token_type):
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        # print(bert_output[1].size())

        output = F.relu(self.fc(bert_output[1]))

        return output

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class roberta_model(nn.Module):
    def __init__(self, configs):
        super(roberta_model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(configs.roberta_cache_path_Chinese)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

    def forward(self, bert_token, bert_mask, token_type):
        roberta_output = self.roberta(input_ids=bert_token, attention_mask=bert_mask)  # not sure


        output = F.relu(self.fc(roberta_output[1]))

        return output


class pert_model(nn.Module):
    def __init__(self, configs):
        super(pert_model, self).__init__()
        self.pert = BertModel.from_pretrained(configs.pert_cache_path_Chinese)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

    def forward(self, bert_token, bert_mask, token_type):
        pert_output = self.pert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure

        output = F.relu(self.fc(pert_output[1]))

        return output


class macbert_model(nn.Module):
    def __init__(self, configs):
        super(macbert_model, self).__init__()
        # self.pert = BertModel.from_pretrained(configs.macbert_cache_path_Chinese)
        self.pert = BertModel.from_pretrained(configs.pretrain_macbert_more_41)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

    def forward(self, bert_token, bert_mask, token_type):
        pert_output = self.pert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure

        output = F.relu(self.fc(pert_output[1]))

        return output


class bert_RCNN_model(nn.Module):
    def __init__(self, configs):
        super(bert_RCNN_model, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

        self.dropout = 0.8
        self.lstm = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, dropout=self.dropout,
                            batch_first=True, bidirectional=True)
        self.W2 = nn.Linear(2 * configs.LSTM_hidden_size + configs.bert_output_dim, configs.LSTM_hidden_size)
        self.label = nn.Linear(configs.LSTM_hidden_size, configs.num_classes)

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过RCNN网络
        input_sentence = pooler_output
        print(input_sentence.size())  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence)
        final_encoding = torch.cat((lstm_output, input_sentence), 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        final_output = self.label(y)

        # 拼接cls向量
        # last_hidden_state = bert_output[1]

        # 加入BERT不同层的embedding

        # 加入DD

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_RCNN_DD_model(nn.Module):
    def __init__(self, configs):
        super(bert_RCNN_DD_model, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

        self.dropout = 0.8
        self.lstm = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, dropout=self.dropout,
                            batch_first=True, bidirectional=True)
        self.W2 = nn.Linear(2 * configs.LSTM_hidden_size + configs.bert_output_dim, configs.LSTM_hidden_size)
        self.dd = nn.Linear(configs.LSTM_hidden_size, configs.LSTM_hidden_size, bias=False)
        self.label = nn.Linear(configs.LSTM_hidden_size, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过RCNN网络
        input_sentence = pooler_output
        print(input_sentence.size())  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence)
        final_encoding = torch.cat((lstm_output, input_sentence), 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)

        # 拼接cls向量
        # last_hidden_state = bert_output[1]
        # y_cat = torch.cat([last_hidden_state, y], dim=1)

        # 加入BERT不同层的embedding

        # 加入DD
        d_out = y

        g1 = self.dd(d_out)
        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2

        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3

        final_output = self.label(d_out)

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_RCNN_cat_model(nn.Module):
    def __init__(self, configs):
        super(bert_RCNN_cat_model, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

        self.dropout = 0.8
        self.lstm = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, dropout=self.dropout,
                            batch_first=True, bidirectional=True)
        self.W2 = nn.Linear(2 * configs.LSTM_hidden_size + configs.bert_output_dim, configs.LSTM_hidden_size)
        self.label = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes)
        self.fc_bert = nn.Linear(configs.bert_output_dim, configs.LSTM_hidden_size)

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过RCNN网络
        input_sentence = pooler_output
        print(input_sentence.size())  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence)
        final_encoding = torch.cat((lstm_output, input_sentence), 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)

        # 拼接cls向量
        last_hidden_state = self.fc_bert(bert_output[1])
        y_cat = torch.cat([last_hidden_state, y], dim=1)
        final_output = self.label(y_cat)

        # 加入BERT不同层的embedding

        # 加入DD

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_RCNN_cat_DD_model(nn.Module):
    def __init__(self, configs):
        super(bert_RCNN_cat_DD_model, self).__init__()
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert = BertModel.from_pretrained(configs.pretrain_bert_all_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_all_final)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

        self.dropout = 0.8
        self.lstm = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, dropout=self.dropout,
                            batch_first=True, bidirectional=True)
        self.fc_bert = nn.Linear(configs.bert_output_dim, configs.LSTM_hidden_size)
        self.W2 = nn.Linear(2 * configs.LSTM_hidden_size + configs.bert_output_dim, configs.LSTM_hidden_size)
        self.dd = nn.Linear(configs.LSTM_hidden_size * 2, configs.LSTM_hidden_size * 2, bias=False)
        self.label = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过RCNN网络
        input_sentence = pooler_output
        print(input_sentence.size())  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence)
        final_encoding = torch.cat((lstm_output, input_sentence), 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)

        # 拼接cls向量
        last_hidden_state = self.fc_bert(bert_output[1])
        print('last_hidden_state.shape', last_hidden_state.shape)
        y_cat = torch.cat([last_hidden_state, y], dim=1)
        print('y_cat.shape', y_cat.shape)

        # 加入BERT不同层的embedding

        # 加入DD
        d_out = y_cat
        print('d_out.shape', d_out.shape)
        g1 = self.dd(d_out)
        print('g1.shape', g1.shape)

        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2

        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3

        final_output = self.label(d_out)

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_Self_Attn_cat_DD_model(nn.Module):
    def __init__(self, configs):
        super(bert_Self_Attn_cat_DD_model, self).__init__()
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        # self.bert = BertModel.from_pretrained(configs.all_test_bert_cache_path_Chinese)
        self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph2_qa_ckpt)
        self.dropout = 0.8
        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, dropout=self.dropout,
                                  num_layers=1, batch_first=True, bidirectional=True)  # 双向LSTM
        self.fc1 = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

        self.W_s1 = nn.Linear(2 * configs.LSTM_hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)

        self.fc_layer = nn.Linear(30 * 2 * configs.LSTM_hidden_size, 2000)
        self.label_att = nn.Linear(2000, configs.LSTM_hidden_size)

        self.fc_bert = nn.Linear(configs.bert_output_dim, configs.LSTM_hidden_size)
        self.dd = nn.Linear(configs.LSTM_hidden_size * 2, configs.LSTM_hidden_size * 2, bias=False)
        self.label = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pooler_output)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        att_output = self.label_att(fc_out)
        # att_output = fc_out

        # 拼接cls向量
        last_hidden_state = self.fc_bert(bert_output[1])
        print('last_hidden_state.shape', last_hidden_state.shape)
        y_cat = torch.cat([last_hidden_state, att_output], dim=1)
        print('y_cat.shape', y_cat.shape)

        # 加入DD
        d_out = y_cat
        print('d_out.shape', d_out.shape)
        g1 = self.dd(d_out)
        print('g1.shape', g1.shape)

        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2

        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3

        final_output = self.label(d_out)

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_LSTM_Attn_model(nn.Module):
    def __init__(self, configs):
        super(bert_LSTM_Attn_model, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size,
                                  batch_first=True)
        self.fc1 = nn.Linear(configs.LSTM_hidden_size, configs.num_classes, bias=False)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        print(hidden.shape)
        print(hidden.unsqueeze(2).shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pooler_output)
        print(output.shape)
        print(final_hidden_state.shape)
        attn_output = self.attention_net(output, final_hidden_state)
        final_output = self.fc1(attn_output)

        # 拼接cls向量
        # last_hidden_state = bert_output[1]

        # 加入BERT不同层的embedding

        # 加入DD

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return


class bert_Self_Attn_model(nn.Module):
    def __init__(self, configs):
        super(bert_Self_Attn_model, self).__init__()
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert = BertModel.from_pretrained(configs.pretrain_bert_all_final)

        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph_qa_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph_qa_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph2_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph2_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph2_qa_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_graph2_qa_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_qa_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_qa_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_qa_graph_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_qa_graph_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_macbert_qa_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_macbert_qa_final)
        # self.bert = BertModel.from_pretrained(configs.pretrain_macbert_graph_ckpt)
        # self.bert = BertModel.from_pretrained(configs.pretrain_macbert_graph_final)

        self.dropout = 0.8

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, dropout=self.dropout,
                                  num_layers=1, batch_first=True, bidirectional=True)  # 双向LSTM
        self.fc1 = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

        self.W_s1 = nn.Linear(2 * configs.LSTM_hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)

        self.fc_layer = nn.Linear(30 * 2 * configs.LSTM_hidden_size, 2000)
        self.label = nn.Linear(2000, configs.num_classes)

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pooler_output)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        final_output = self.label(fc_out)

        # 拼接cls向量
        # last_hidden_state = bert_output[1]

        # 加入BERT不同层的embedding

        # 加入DD

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_Self_DD_Attn_model(nn.Module):
    def __init__(self, configs):
        super(bert_Self_DD_Attn_model, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)

        self.dropout = 0.8

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, dropout=self.dropout,
                                  num_layers=1, batch_first=True, bidirectional=True)  # 双向LSTM
        self.fc1 = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

        self.W_s11 = nn.Linear(2 * configs.LSTM_hidden_size, 350, bias=False)
        self.W_s22 = nn.Linear(350, 30, bias=False)
        self.dd11 = nn.Linear(350, 350, bias=False)
        self.dd22 = nn.Linear(30, 30, bias=False)

        self.fc_layer = nn.Linear(30 * 2 * configs.LSTM_hidden_size, 2000, bias=False)
        self.label = nn.Linear(2000, configs.num_classes, bias=False)

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s11(lstm_output)
        attn_weight_matrix_g = self.dd11(attn_weight_matrix)
        attn_weight_matrix = attn_weight_matrix_g * attn_weight_matrix + attn_weight_matrix_g
        attn_weight_matrix = self.W_s22(attn_weight_matrix)
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pooler_output)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        final_output = self.label(fc_out)

        # 拼接cls向量
        # last_hidden_state = bert_output[1]

        # 加入BERT不同层的embedding

        # 加入DD

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_LSTM_Attn_cat_DD_model(nn.Module):
    def __init__(self, configs):
        super(bert_LSTM_Attn_cat_DD_model, self).__init__()
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert = BertModel.from_pretrained(configs.pretrain_bert_all_final)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size,
                                  batch_first=True)
        self.fc_bert = nn.Linear(configs.bert_output_dim, configs.LSTM_hidden_size)

        self.dd = nn.Linear(configs.LSTM_hidden_size * 2, configs.LSTM_hidden_size * 2, bias=False)
        self.label = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)
        # self.fc1 = nn.Linear(configs.LSTM_hidden_size, configs.num_classes, bias=False)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        print(hidden.shape)
        print(hidden.unsqueeze(2).shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]

        # 通过LSTM self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pooler_output)
        print(output.shape)
        print(final_hidden_state.shape)
        attn_output = self.attention_net(output, final_hidden_state)
        # att_output = self.fc1(attn_output)

        # 拼接cls向量
        cls_outpot = self.fc_bert(bert_output[1])
        print('last_hidden_state.shape', cls_outpot.shape)
        y_cat = torch.cat([cls_outpot, attn_output], dim=1)
        print('y_cat.shape', y_cat.shape)

        # 加入DD
        d_out = y_cat
        print('d_out.shape', d_out.shape)
        g1 = self.dd(d_out)
        print('g1.shape', g1.shape)

        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2

        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3

        final_output = self.label(d_out)

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class bert_LSTM_cat_DD_model(nn.Module):
    def __init__(self, configs):
        super(bert_LSTM_cat_DD_model, self).__init__()
        self.bert = AutoModel.from_pretrained(configs.pretrain_macbert_all)
        # self.bert = BertModel.from_pretrained(configs.pretrain_bert_all_more_test)
        # self.bert = BertModel.from_pretrained(configs.macbert_cache_path_Chinese)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, num_layers=2, batch_first=True,
                                  bidirectional=True)

        self.fc_bert = nn.Linear(configs.bert_output_dim, configs.LSTM_hidden_size * 2)

        self.dd = nn.Linear(configs.LSTM_hidden_size * 4, configs.LSTM_hidden_size * 4, bias=False)
        self.label = nn.Linear(configs.LSTM_hidden_size * 4, configs.num_classes, bias=False)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        print(hidden.shape)
        print(hidden.unsqueeze(2).shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = bert_output[0]
        print(bert_output[0].shape)

        # 通过LSTM self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pooler_output)
        print('output.shape', output.shape)
        print('final_hidden_state.shape', final_hidden_state.shape)
        # attn_output = self.attention_net(output, final_hidden_state)
        # att_output = self.fc1(attn_output)

        # 拼接cls向量
        cls_outpot = self.fc_bert(bert_output[0][:, 0, :])
        print('last_hidden_state.shape', cls_outpot.shape)
        y_cat = torch.cat([cls_outpot, output[:, -1, :]], dim=1)
        print('y_cat.shape', y_cat.shape)

        # 加入DD
        d_out = y_cat
        print('d_out.shape', d_out.shape)
        g1 = self.dd(d_out)
        print('g1.shape', g1.shape)

        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2
        #
        # g3 = self.dd(d_out)
        # d_out = g3 * d_out + g3

        # g4 = self.dd(d_out)
        # d_out = g4 * d_out + g4
        #
        # g5 = self.dd(d_out)
        # d_out = g5 * d_out + g5

        final_output = self.label(d_out)

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class RoBERTa_LSTM_cat_DD_model(nn.Module):
    def __init__(self, configs):
        super(RoBERTa_LSTM_cat_DD_model, self).__init__()
        # self.roberta = RobertaModel.from_pretrained(configs.pretrain_roberta_all)
        self.roberta = RobertaModel.from_pretrained(configs.roberta_cache_path_Chinese)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, num_layers=2, batch_first=True,
                                  bidirectional=True)
        self.fc_bert = nn.Linear(configs.bert_output_dim, configs.LSTM_hidden_size * 2)

        self.dd = nn.Linear(configs.LSTM_hidden_size * 4, configs.LSTM_hidden_size * 4, bias=False)
        self.label = nn.Linear(configs.LSTM_hidden_size * 4, configs.num_classes, bias=False)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        print(hidden.shape)
        print(hidden.unsqueeze(2).shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, bert_token, bert_mask):
        # 利用BERT得到embedding
        roberta_output = self.roberta(input_ids=bert_token, attention_mask=bert_mask)  # not sure
        pooler_output = roberta_output[0]

        # 通过LSTM self attention网络
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(pooler_output)
        print('output.shape', output.shape)
        print('final_hidden_state.shape', final_hidden_state.shape)
        # attn_output = self.attention_net(output, final_hidden_state)
        # att_output = self.fc1(attn_output)

        # 拼接cls向量
        cls_outpot = self.fc_bert(roberta_output[1])
        print('last_hidden_state.shape', cls_outpot.shape)
        y_cat = torch.cat([cls_outpot, output[:, -1, :]], dim=1)
        print('y_cat.shape', y_cat.shape)

        # 加入DD
        d_out = y_cat
        print('d_out.shape', d_out.shape)
        g1 = self.dd(d_out)
        print('g1.shape', g1.shape)

        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2

        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3

        final_output = self.label(d_out)

        # 与glove的embedding进行对比

        # 在embedding基础上换不同的网络

        return final_output


class electra_LSTM_cat_DD_model(nn.Module):
    def __init__(self, configs):
        super(electra_LSTM_cat_DD_model, self).__init__()
        self.electra = AutoModelForPreTraining.from_pretrained(configs.electra_cache_path_Chinese)
        # self.bert = RobertaModel.from_pretrained(configs.pretrain_macbert_all)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.LSTM_hidden_size, num_layers=2, batch_first=True,
                                  bidirectional=True)
        self.fc_bert = nn.Linear(256, 256)

        self.dd = nn.Linear(256, 256, bias=False)
        self.label = nn.Linear(256, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type):
        # 利用BERT得到embedding
        electra_output = self.electra(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure
        pooler_output = electra_output[0]


        # 拼接cls向量
        cls_outpot = self.fc_bert(pooler_output)
        print('last_hidden_state.shape', cls_outpot.shape)


        # 加入DD
        d_out = cls_outpot
        print('d_out.shape', d_out.shape)
        g1 = self.dd(d_out)
        print('g1.shape', g1.shape)

        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2
        #
        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3
        #
        # g4 = self.dd(d_out)
        # d_out = g4 * d_out + g4
        # #
        # g5 = self.dd(d_out)
        # d_out = g5 * d_out + g5

        final_output = self.label(d_out)

        return final_output


class f_model(nn.Module):
    def __init__(self, configs):
        super(f_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.fc1 = nn.ReLU()
        self.fc2 = nn.ReLU()
        self.fc4 = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())

    def forward(self, bert_token, bert_mask, token_type, touch, anger, amuse, sad, novel,
                shock, p, n):
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure

        finegrained = F.softmax(F.relu(self.fc(bert_output[1])))

        touch_output = self.bert(touch)
        anger_output = self.bert(anger)
        amuse_output = self.bert(amuse)
        sad_output = self.bert(sad)
        novel_output = self.bert(novel)
        shock_output = self.bert(shock)

        label_emb = torch.stack([touch_output[1], anger_output[1], amuse_output[1],
                                 sad_output[1], novel_output[1], shock_output[1]], dim=1)

        label_emb = label_emb.permute(0, 2, 1)
        pos_scores = self.fc1(torch.matmul(bert_output[1].unsqueeze(1), label_emb))

        neg_scores = self.fc2(torch.matmul(bert_output[1].unsqueeze(1), label_emb))

        pos_prob = masked_softmax(pos_scores.squeeze(1), p)
        neg_prob = masked_softmax(neg_scores.squeeze(1), n)

        # pos_prob = F.softmax(pos_scores)
        # neg_prob = F.softmax(neg_scores)

        pos_score = torch.Tensor.sum(finegrained * pos_prob, dim=1, keepdim=True)
        neg_score = torch.Tensor.sum(finegrained * neg_prob, dim=1, keepdim=True)

        prob = torch.cat((pos_score, neg_score), 1)

        out = self.fc4(prob)

        return out, finegrained

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class f_model2(nn.Module):
    def __init__(self, configs):
        super(f_model2, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.fc1 = nn.Sequential(nn.Linear(512, 1), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 1), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(2, 64), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, bert_token, bert_mask, token_type, touch, anger, amuse, sad, novel,
                shock, p, n):
        bert_output = self.bert(input_ids=bert_token,
                                attention_mask=bert_mask,
                                token_type_ids=token_type)  # not sure [b, 512, 768]

        finegrained = F.softmax(F.relu(self.fc(bert_output[1])))

        touch_output = self.bert(touch)  # [b, 4, 768]
        anger_output = self.bert(anger)
        amuse_output = self.bert(amuse)
        sad_output = self.bert(sad)
        novel_output = self.bert(novel)
        shock_output = self.bert(shock)

        label_emb = torch.stack([touch_output[0], anger_output[0], amuse_output[0],
                                 sad_output[0], novel_output[0], shock_output[0]], dim=1)  # [b,6,4,768]

        label_emb = label_emb.permute(0, 1, 3, 2)  # [b, 6, 768, 4]
        mid = torch.matmul(bert_output[0].unsqueeze(1), label_emb)  # [b, 6, 512, 4]

        mid = torch.mean(mid, dim=3)  # [b, 6, 512]

        pos_scores = self.fc1(mid)  # [b, 6, 1]

        neg_scores = self.fc2(mid)  # [b, 6, 1]

        pos_prob = masked_softmax(pos_scores.squeeze(2), p)
        neg_prob = masked_softmax(neg_scores.squeeze(2), n)

        # pos_prob = F.softmax(pos_scores)
        # neg_prob = F.softmax(neg_scores)

        pos_score = torch.Tensor.sum(finegrained * pos_prob, dim=1, keepdim=True)
        neg_score = torch.Tensor.sum(finegrained * neg_prob, dim=1, keepdim=True)

        prob = torch.cat((pos_score, neg_score), 1)

        out = self.fc3(prob)
        out = self.fc4(out)

        return out, finegrained

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs
