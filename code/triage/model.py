from transformers import BertModel, BertConfig, RobertaModel
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
        self.lstm_layer = nn.LSTM(configs.GloVe_embedding_length, configs.LSTM_hidden_size, num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.fc1 = nn.Linear(configs.LSTM_hidden_size * 2, configs.num_classes, bias=False)

    def forward(self, lstm_input):
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
                               (self.kernel_heights[0], configs.GloVe_embedding_length), self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[1], configs.GloVe_embedding_length), self.stride, self.padding)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                               (self.kernel_heights[2], configs.GloVe_embedding_length), self.stride, self.padding)
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


class coRNN(nn.Module):
    def __init__(self, n_inp = 300, n_hid=128, dt=5.4e-2, gamma=4.9, epsilon=4.8):
        super(coRNN, self).__init__()
        self.n_inp = 300
        self.n_hid = n_hid
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(self.n_inp, self.n_hid)
        self.h2h = nn.Linear(self.n_hid + self.n_hid, self.n_hid, bias=False)

    def forward(self, x):
        hy = Variable(torch.zeros(x.size(1), self.n_hid)).to(DEVICE)
        hz = Variable(torch.zeros(x.size(1), self.n_hid)).to(DEVICE)
        inputs = self.i2h(x)
        for t in range(x.size(0)):
            hz = hz + self.dt * (torch.tanh(self.h2h(torch.cat((hz, hy), dim=1)) + inputs[t])
                                 - self.gamma * hy - self.epsilon * hz)
            # hy = hy + self.dt * hz
            hy_0 = hy.unsqueeze(0)
            hy_final = hy_0 + self.dt * hz
            hy_0 = hy_final
        print(hy_final.shape)
        return hy_final

class RNNModel(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, output_dim=2, dt=5.4e-2, gamma=4.9, epsilon=4.8):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = coRNN(embedding_dim, hidden_dim,dt,gamma,epsilon).to(DEVICE)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # embedded = self.embedding(text)
        hidden = self.rnn(text)
        out = self.readout(hidden)
        print(out.shape)
        return out

class DPCNN(nn.Module):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, 300), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, 2)

    def forward(self, x):
        batch = x.shape[0]

        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, 2*self.channel_size)
        x = self.linear_out(x)

        return x


class our_model(nn.Module):
    def __init__(self, configs):
        super(our_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        # self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.bert = RobertaModel.from_pretrained(configs.roberta_cache_path_english)

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
        self.roberta = RobertaModel.from_pretrained(configs.roberta_cache_path_english)

        self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)
        self.embedding_size = configs.embedding_size

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # print(bert_output[1].size())

        output = F.relu(self.fc(roberta_output[1]))

        return output


class bert_lstm_model(nn.Module):
    def __init__(self, configs):
        super(bert_lstm_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        # self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        # self.bert_0 = BertModel.from_pretrained(configs.roberta_cache_path_Chinese)
        self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_1 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_2 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_3 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_4 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_5 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_6 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_7 = BertModel.from_pretrained(configs.bert_cache_path)
        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.fc = nn.Linear(configs.lstm_hidden_dim, configs.num_classes, bias=False)
        self.fc1 = nn.Linear(configs.lstm_hidden_dim * 2, configs.num_classes, bias=False)
        self.fc2 = torch.nn.Linear(configs.bert_output_dim, configs.num_classes, bias=False)
        # self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)

    def forward(self, bert_token, bert_mask, token_type, length):
        # print(bert_token.shape)
        # print(bert_mask.shape)
        # print(token_type.shape)
        # print(length.shape)
        # lens = torch.split(self.get_length_matrix(length), 1, dim=1)
        # lens = torch.split(length, 1, dim=2)
        # bert_tokens = torch.split(bert_token, 1, dim=2)
        # bert_masks = torch.split(bert_mask, 1, dim=2)
        # token_types = torch.split(token_type, 1, dim=2)
        '''
        lens = list(torch.split(length, 1, dim=2))
        bert_tokens = list(torch.split(bert_token, 1, dim=2))
        bert_masks = list(torch.split(bert_mask, 1, dim=2))
        token_types = list(torch.split(token_type, 1, dim=2))
        '''

        # bert_token_0 = bert_token[0].view([25000, 350])
        '''
        for i in range(self.pieces_size):
            bert_tokens[i] = bert_tokens[i].view([-1, self.embedding_size])
            bert_masks[i] = bert_masks[i].view([-1, self.embedding_size])
            token_types[i] = token_types[i].view([-1, self.embedding_size])
            lens[i] = lens[i].view([-1, self.embedding_size])
        '''
        # for i in range(self.pieces_size):
        #     bert_tokens[i] = bert_tokens[i].view([self.batch_size, self.embedding_size])
        #     bert_masks[i] = bert_masks[i].view([self.batch_size, self.embedding_size])
        #     token_types[i] = token_types[i].view([self.batch_size, self.embedding_size])
        #     lens[i] = lens[i].view([self.batch_size, self.embedding_size])
        # bert_tokens_0 = bert_tokens[0].view([self.batch_size, self.embedding_size])
        # bert_masks_0 = bert_masks[0].view([self.batch_size, self.embedding_size])
        # token_types_0 = token_types[0].view([self.batch_size, self.embedding_size])
        # lens_0 = lens[0].view([self.batch_size, self.embedding_size])
        # print(bert_tokens_0.shape)
        # print(bert_masks_0.shape)
        # print(token_types_0.shape)
        # print(lens_0.shape)
        # bert_output_0 = self.bert_0(input_ids=(bert_tokens_0 * lens_0).long(),
        #                             attention_mask=(bert_masks_0 * lens_0).long(),
        #                             token_type_ids=(token_types_0 * lens_0).long())  # not sure

        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        # print(bert_output_0[1].size())
        # print(bert_output_0[1])
        len_0 = length[:, 0, 0].view([-1, 1])
        lstm_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])

        # mean_input = bert_output_0[1] * len_0
        # mean_input = bert_output_0[1]
        # print('mean_input')
        # print(mean_input)
        # print(mean_input.size)

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)
        # mean_input = bert_output_1[1] * len_1 + mean_input

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)

        print('lstm part')
        print(lstm_input.shape)
        # print(lstm_input[:,:,7])

        lstm_input = lstm_input.transpose(1, 2).contiguous()
        # len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        # len_all = len_0 + len_1 + len_2 + len_3
        # print(mean_input)
        # print(len_all)
        # if sum(sum(len_all)) == 0:
        #     print('error')

        # mean_input = mean_input / len_all
        # print(lstm_input.shape)

        output, (hn, cn) = self.lstm_layer(lstm_input)
        # print('mean_input')
        # print(mean_input)
        # axa = self.fc2(mean_input)
        # print('axa')
        # print(axa)
        # mean_output = F.leaky_relu(self.fc2(mean_input))
        print('output')
        # print(mean_output)
        print(output.shape)
        print(output[:, -1, :].shape)

        # hidden = torch.cat([hn[0], hn[-1]], dim=1)

        # output = F.relu(self.fc1(lstm_output))
        lstm_output = F.leaky_relu(self.fc1(output[:, -1, :]))
        # print(output.shape)

        # return mean_output
        return lstm_output

    # def get_length_matrix(self, length):
    #     matrix = torch.zeros([self.batch_size, 8])
    #     for i in range(self.batch_size):
    #         if length[i] > 8:
    #             lens = 8
    #         else:
    #             lens = length[i]
    #         for j in range(lens):
    #             matrix[i][j] = 1
    #     return matrix

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_lstm_plus1_model(nn.Module):
    def __init__(self, configs):
        super(bert_lstm_plus1_model, self).__init__()
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        # self.bert_1 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert_0 = BertModel.from_pretrained(configs.roberta_cache_path_Chinese)
        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.fc = nn.Linear(configs.lstm_hidden_dim, configs.num_classes, bias=False)
        self.fc1 = nn.Linear(configs.lstm_hidden_dim * 2, configs.num_classes, bias=False)
        # self.fc2 = nn.Linear(configs.bert_output_dim, configs.num_classes, bias=False)
        self.fc3 = nn.Linear(configs.lstm_hidden_dim * 2 + configs.bert_output_dim, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type, length, bert_token_summary, bert_mask_summary,
                token_type_summary):
        """
        lens = list(torch.split(length, 1, dim=2))
        bert_tokens = list(torch.split(bert_token, 1, dim=2))
        bert_masks = list(torch.split(bert_mask, 1, dim=2))
        token_types = list(torch.split(token_type, 1, dim=2))
        """

        '''
        for i in range(self.pieces_size):
            bert_tokens[i] = bert_tokens[i].view([-1, self.embedding_size])
            bert_masks[i] = bert_masks[i].view([-1, self.embedding_size])
            token_types[i] = token_types[i].view([-1, self.embedding_size])
            lens[i] = lens[i].view([-1, self.embedding_size])
        '''
        bert_output_summary = self.bert_0(
            input_ids=bert_token_summary.int(),
            attention_mask=bert_mask_summary.int(),
            token_type_ids=token_type_summary.int())
        lstm_input = bert_output_summary[1].view([-1, self.hidden_size, 1])

        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        # lstm_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])), 2)

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)

        print('lstm part')
        print(lstm_input.shape)
        # print(lstm_input[:,:,7])

        lstm_input = lstm_input.transpose(1, 2).contiguous()

        output, (hn, cn) = self.lstm_layer(lstm_input)

        print('output')
        print(output.shape)
        print(output[:, -1, :].shape)

        # dense_input = torch.cat([bert_output_summary[1], output[:, -1, :]], dim=1)
        # output = F.relu(self.fc1(lstm_output))
        lstm_output = F.leaky_relu(self.fc1(output[:, -1, :]))
        # dense_output = F.leaky_relu(self.fc3(dense_input))
        # print(output.shape)

        # return mean_output
        return lstm_output

    # def get_length_matrix(self, length):
    #     matrix = torch.zeros([self.batch_size, 8])
    #     for i in range(self.batch_size):
    #         if length[i] > 8:
    #             lens = 8
    #         else:
    #             lens = length[i]
    #         for j in range(lens):
    #             matrix[i][j] = 1
    #     return matrix

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_transformer_LSTM_plus1_model(nn.Module):
    def __init__(self, configs):
        super(bert_transformer_LSTM_plus1_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        # self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        self.bert_0 = BertModel.from_pretrained(configs.roberta_cache_path_Chinese)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.bert_output_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=configs.bert_output_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.fc = nn.Linear(configs.lstm_hidden_dim, configs.num_classes, bias=False)
        self.fc1 = nn.Linear(configs.lstm_hidden_dim * 2, configs.num_classes, bias=False)
        self.fc_transformer = nn.Linear(configs.bert_output_dim * (1 + self.pieces_size), configs.lstm_hidden_dim * 2,
                                        bias=False)
        # self.fc2 = torch.nn.Linear(configs.bert_output_dimdim * (1 + self.pieces_size), configs.num_classes, bias=False)
        self.fc_final = torch.nn.Linear(configs.lstm_hidden_dim * 4, configs.num_classes, bias=False)
        # self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        print(hidden.shape)
        print(hidden.unsqueeze(2).shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, bert_token, bert_mask, token_type, length, bert_token_summary, bert_mask_summary,
                token_type_summary):
        # print(bert_token.shape)
        # print(bert_mask.shape)
        # print(token_type.shape)
        # print(length.shape)
        # lens = torch.split(self.get_length_matrix(length), 1, dim=1)
        # lens = torch.split(length, 1, dim=2)
        # bert_tokens = torch.split(bert_token, 1, dim=2)
        # bert_masks = torch.split(bert_mask, 1, dim=2)
        # token_types = torch.split(token_type, 1, dim=2)
        '''
        lens = list(torch.split(length, 1, dim=2))
        bert_tokens = list(torch.split(bert_token, 1, dim=2))
        bert_masks = list(torch.split(bert_mask, 1, dim=2))
        token_types = list(torch.split(token_type, 1, dim=2))
        '''

        # bert_token_0 = bert_token[0].view([25000, 350])
        '''
        for i in range(self.pieces_size):
            bert_tokens[i] = bert_tokens[i].view([-1, self.embedding_size])
            bert_masks[i] = bert_masks[i].view([-1, self.embedding_size])
            token_types[i] = token_types[i].view([-1, self.embedding_size])
            lens[i] = lens[i].view([-1, self.embedding_size])
        '''
        # for i in range(self.pieces_size):
        #     bert_tokens[i] = bert_tokens[i].view([self.batch_size, self.embedding_size])
        #     bert_masks[i] = bert_masks[i].view([self.batch_size, self.embedding_size])
        #     token_types[i] = token_types[i].view([self.batch_size, self.embedding_size])
        #     lens[i] = lens[i].view([self.batch_size, self.embedding_size])
        # bert_tokens_0 = bert_tokens[0].view([self.batch_size, self.embedding_size])
        # bert_masks_0 = bert_masks[0].view([self.batch_size, self.embedding_size])
        # token_types_0 = token_types[0].view([self.batch_size, self.embedding_size])
        # lens_0 = lens[0].view([self.batch_size, self.embedding_size])
        # print(bert_tokens_0.shape)
        # print(bert_masks_0.shape)
        # print(token_types_0.shape)
        # print(lens_0.shape)
        # bert_output_0 = self.bert_0(input_ids=(bert_tokens_0 * lens_0).long(),
        #                             attention_mask=(bert_masks_0 * lens_0).long(),
        #                             token_type_ids=(token_types_0 * lens_0).long())  # not sure

        bert_output_summary = self.bert_0(
            input_ids=bert_token_summary.int(),
            attention_mask=bert_mask_summary.int(),
            token_type_ids=token_type_summary.int())
        lstm_input = bert_output_summary[1].view([-1, self.hidden_size, 1])

        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        # lstm_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])), 2)

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)
        # mean_input = bert_output_1[1] * len_1 + mean_input

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)

        print('lstm part')
        print(lstm_input.shape)
        # print(lstm_input[:,:,7])

        lstm_input = lstm_input.transpose(1, 2).contiguous()
        # len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        # len_all = len_0 + len_1 + len_2 + len_3
        # print(mean_input)
        # print(len_all)
        # if sum(sum(len_all)) == 0:
        #     print('error')

        # mean_input = mean_input / len_all
        # print(lstm_input.shape)

        output, (hn, cn) = self.lstm_layer(lstm_input)
        # lstm_output = F.leaky_relu(self.fc1(output[:, -1, :]))

        out = self.transformer_encoder(lstm_input)
        print('transformer_input')
        print(out.shape)
        transformer_output = self.fc_transformer(out.flatten(start_dim=1))

        print('transformer_output')
        print(transformer_output.shape)
        # axa = self.fc2(mean_input)
        # print('axa')
        # print(axa)
        # mean_output = F.leaky_relu(self.fc2(mean_input))
        # print('output')
        # print(mean_output)
        # print(output.shape)
        print(output[:, -1, :].shape)

        hidden = torch.cat([output[:, -1, :], transformer_output], dim=1)
        print('final_output')
        print(hidden.shape)
        # output = F.relu(self.fc1(lstm_output))
        # transformer_output = F.leaky_relu(self.fc2(out.flatten(start_dim=1)))
        transformer_output = F.leaky_relu(self.fc_final(hidden))
        # print('transformer_output')
        # print(transformer_output.shape)
        # print(output.shape)

        # return mean_output
        return transformer_output

    # def get_length_matrix(self, length):
    #     matrix = torch.zeros([self.batch_size, 8])
    #     for i in range(self.batch_size):
    #         if length[i] > 8:
    #             lens = 8
    #         else:
    #             lens = length[i]
    #         for j in range(lens):
    #             matrix[i][j] = 1
    #     return matrix

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_dd_LSTM_plus1_model(nn.Module):
    def __init__(self, configs):
        super(bert_dd_LSTM_plus1_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        # self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        self.bert_0 = BertModel.from_pretrained(configs.roberta_cache_path_Chinese)
        self.dd = nn.Linear(configs.lstm_hidden_dim * 2, configs.lstm_hidden_dim * 2, bias=False)
        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM

        self.fc0 = nn.Linear(configs.bert_output_dim * (1 + self.pieces_size), configs.lstm_hidden_dim * 2, bias=False)
        self.fc = nn.Linear(configs.lstm_hidden_dim, configs.num_classes, bias=False)
        self.fc1 = nn.Linear(configs.lstm_hidden_dim * 2, configs.num_classes, bias=False)
        self.fc_transformer = nn.Linear(configs.bert_output_dim * (1 + self.pieces_size), configs.lstm_hidden_dim * 2,
                                        bias=False)
        # self.fc2 = torch.nn.Linear(configs.bert_output_dim * (1 + self.pieces_size), configs.num_classes, bias=False)
        self.fc_final1 = torch.nn.Linear(configs.lstm_hidden_dim * 4, 512, bias=False)
        self.fc_final2 = torch.nn.Linear(512, configs.num_classes, bias=False)
        # self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)

    def attention_net(self, lstm_output, final_state):
        print('lstm_output.shape:', lstm_output.shape)
        print('final_state.shape:', final_state.shape)
        # hidden = final_state.squeeze(0)
        hidden = final_state.transpose(0, 1).contiguous().view([-1, 2048])

        print('hidden.shape:', hidden.shape)
        print('hidden.unsqueeze(2).shape:', hidden.unsqueeze(2).shape)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        print('attn_weights.shape:', attn_weights.shape)
        soft_attn_weights = F.softmax(attn_weights, 1)
        print('soft_attn_weights.shape:', soft_attn_weights.shape)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        print('new_hidden_state.shape:', new_hidden_state.shape)

        return new_hidden_state

    def forward(self, bert_token, bert_mask, token_type, length, bert_token_summary, bert_mask_summary,
                token_type_summary):
        bert_output_summary = self.bert_0(
            input_ids=bert_token_summary.int(),
            attention_mask=bert_mask_summary.int(),
            token_type_ids=token_type_summary.int())
        lstm_input = bert_output_summary[1].view([-1, self.hidden_size, 1])

        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])), 2)

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)

        print('lstm part')
        print(lstm_input.shape)

        lstm_input = lstm_input.transpose(1, 2).contiguous()

        output, (hn, cn) = self.lstm_layer(lstm_input)
        attn_output = self.attention_net(output, hn)

        # c = lstm_input
        d_out = self.fc0(lstm_input.flatten(start_dim=1))

        g1 = self.dd(d_out)
        d_out = g1 * d_out + g1

        g2 = self.dd(d_out)
        d_out = g2 * d_out + g2

        g3 = self.dd(d_out)
        d_out = g3 * d_out + g3

        # transformer_output = self.fc_transformer(out.flatten(start_dim=1))

        # print('transformer_output')
        # print(transformer_output.shape)
        # print(output[:, -1, :].shape)

        hidden = torch.cat([output[:, -1, :], d_out], dim=1)
        print('final_output')
        print(hidden.shape)
        transformer_output = F.leaky_relu(self.fc_final2(self.fc_final1(hidden)))

        return transformer_output

    # def get_length_matrix(self, length):
    #     matrix = torch.zeros([self.batch_size, 8])
    #     for i in range(self.batch_size):
    #         if length[i] > 8:
    #             lens = 8
    #         else:
    #             lens = length[i]
    #         for j in range(lens):
    #             matrix[i][j] = 1
    #     return matrix

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_transformer_plus1_model(nn.Module):
    def __init__(self, configs):
        super(bert_transformer_plus1_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        # self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        self.bert_0 = BertModel.from_pretrained(configs.roberta_cache_path_Chinese)

        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.bert_output_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=configs.bert_output_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.fc = nn.Linear(configs.lstm_hidden_dim, configs.num_classes, bias=False)
        self.fc1 = nn.Linear(configs.lstm_hidden_dim * 2, configs.num_classes, bias=False)
        self.fc2 = torch.nn.Linear(configs.bert_output_dim * (1 + self.pieces_size), configs.num_classes, bias=False)
        # self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)

    def forward(self, bert_token, bert_mask, token_type, length, bert_token_summary, bert_mask_summary,
                token_type_summary):
        # print(bert_token.shape)
        # print(bert_mask.shape)
        # print(token_type.shape)
        # print(length.shape)
        # lens = torch.split(self.get_length_matrix(length), 1, dim=1)
        # lens = torch.split(length, 1, dim=2)
        # bert_tokens = torch.split(bert_token, 1, dim=2)
        # bert_masks = torch.split(bert_mask, 1, dim=2)
        # token_types = torch.split(token_type, 1, dim=2)
        '''
        lens = list(torch.split(length, 1, dim=2))
        bert_tokens = list(torch.split(bert_token, 1, dim=2))
        bert_masks = list(torch.split(bert_mask, 1, dim=2))
        token_types = list(torch.split(token_type, 1, dim=2))
        '''

        # bert_token_0 = bert_token[0].view([25000, 350])
        '''
        for i in range(self.pieces_size):
            bert_tokens[i] = bert_tokens[i].view([-1, self.embedding_size])
            bert_masks[i] = bert_masks[i].view([-1, self.embedding_size])
            token_types[i] = token_types[i].view([-1, self.embedding_size])
            lens[i] = lens[i].view([-1, self.embedding_size])
        '''
        # for i in range(self.pieces_size):
        #     bert_tokens[i] = bert_tokens[i].view([self.batch_size, self.embedding_size])
        #     bert_masks[i] = bert_masks[i].view([self.batch_size, self.embedding_size])
        #     token_types[i] = token_types[i].view([self.batch_size, self.embedding_size])
        #     lens[i] = lens[i].view([self.batch_size, self.embedding_size])
        # bert_tokens_0 = bert_tokens[0].view([self.batch_size, self.embedding_size])
        # bert_masks_0 = bert_masks[0].view([self.batch_size, self.embedding_size])
        # token_types_0 = token_types[0].view([self.batch_size, self.embedding_size])
        # lens_0 = lens[0].view([self.batch_size, self.embedding_size])
        # print(bert_tokens_0.shape)
        # print(bert_masks_0.shape)
        # print(token_types_0.shape)
        # print(lens_0.shape)
        # bert_output_0 = self.bert_0(input_ids=(bert_tokens_0 * lens_0).long(),
        #                             attention_mask=(bert_masks_0 * lens_0).long(),
        #                             token_type_ids=(token_types_0 * lens_0).long())  # not sure

        bert_output_summary = self.bert_0(
            input_ids=bert_token_summary.int(),
            attention_mask=bert_mask_summary.int(),
            token_type_ids=token_type_summary.int())
        lstm_input = bert_output_summary[1].view([-1, self.hidden_size, 1])

        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        # lstm_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])), 2)

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)
        # mean_input = bert_output_1[1] * len_1 + mean_input

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)

        print('lstm part')
        print(lstm_input.shape)
        # print(lstm_input[:,:,7])

        lstm_input = lstm_input.transpose(1, 2).contiguous()
        # len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        # len_all = len_0 + len_1 + len_2 + len_3
        # print(mean_input)
        # print(len_all)
        # if sum(sum(len_all)) == 0:
        #     print('error')

        # mean_input = mean_input / len_all
        # print(lstm_input.shape)

        # output, (hn, cn) = self.lstm_layer(lstm_input)
        out = self.transformer_encoder(lstm_input)
        print('transformer_input')
        print(out.shape)
        # axa = self.fc2(mean_input)
        # print('axa')
        # print(axa)
        # mean_output = F.leaky_relu(self.fc2(mean_input))
        # print('output')
        # print(mean_output)
        # print(output.shape)
        # print(output[:, -1, :].shape)

        # hidden = torch.cat([hn[0], hn[-1]], dim=1)

        # output = F.relu(self.fc1(lstm_output))
        transformer_output = F.leaky_relu(self.fc2(out.flatten(start_dim=1)))
        print('transformer_output')
        print(transformer_output.shape)
        # print(output.shape)

        # return mean_output
        return transformer_output

    # def get_length_matrix(self, length):
    #     matrix = torch.zeros([self.batch_size, 8])
    #     for i in range(self.batch_size):
    #         if length[i] > 8:
    #             lens = 8
    #         else:
    #             lens = length[i]
    #         for j in range(lens):
    #             matrix[i][j] = 1
    #     return matrix

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_transformer_model(nn.Module):
    def __init__(self, configs):
        super(bert_transformer_model, self).__init__()
        # self.bert_config = BertConfig(configs.bert_cache_path + 'bert_config.json')
        # self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_1 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_2 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_3 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_4 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_5 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_6 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_7 = BertModel.from_pretrained(configs.bert_cache_path)
        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.bert_output_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=configs.bert_output_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.fc = nn.Linear(configs.lstm_hidden_dim, configs.num_classes, bias=False)
        self.fc1 = nn.Linear(configs.lstm_hidden_dim * 2, configs.num_classes, bias=False)
        self.fc2 = torch.nn.Linear(configs.bert_output_dim * self.pieces_size, configs.num_classes, bias=False)
        # self.fc = torch.nn.Linear(configs.bert_output_dim, configs.num_classes)

    def forward(self, bert_token, bert_mask, token_type, length):
        # print(bert_token.shape)
        # print(bert_mask.shape)
        # print(token_type.shape)
        # print(length.shape)
        # lens = torch.split(self.get_length_matrix(length), 1, dim=1)
        # lens = torch.split(length, 1, dim=2)
        # bert_tokens = torch.split(bert_token, 1, dim=2)
        # bert_masks = torch.split(bert_mask, 1, dim=2)
        # token_types = torch.split(token_type, 1, dim=2)
        '''
        lens = list(torch.split(length, 1, dim=2))
        bert_tokens = list(torch.split(bert_token, 1, dim=2))
        bert_masks = list(torch.split(bert_mask, 1, dim=2))
        token_types = list(torch.split(token_type, 1, dim=2))
        '''

        # bert_token_0 = bert_token[0].view([25000, 350])
        '''
        for i in range(self.pieces_size):
            bert_tokens[i] = bert_tokens[i].view([-1, self.embedding_size])
            bert_masks[i] = bert_masks[i].view([-1, self.embedding_size])
            token_types[i] = token_types[i].view([-1, self.embedding_size])
            lens[i] = lens[i].view([-1, self.embedding_size])
        '''
        # for i in range(self.pieces_size):
        #     bert_tokens[i] = bert_tokens[i].view([self.batch_size, self.embedding_size])
        #     bert_masks[i] = bert_masks[i].view([self.batch_size, self.embedding_size])
        #     token_types[i] = token_types[i].view([self.batch_size, self.embedding_size])
        #     lens[i] = lens[i].view([self.batch_size, self.embedding_size])
        # bert_tokens_0 = bert_tokens[0].view([self.batch_size, self.embedding_size])
        # bert_masks_0 = bert_masks[0].view([self.batch_size, self.embedding_size])
        # token_types_0 = token_types[0].view([self.batch_size, self.embedding_size])
        # lens_0 = lens[0].view([self.batch_size, self.embedding_size])
        # print(bert_tokens_0.shape)
        # print(bert_masks_0.shape)
        # print(token_types_0.shape)
        # print(lens_0.shape)
        # bert_output_0 = self.bert_0(input_ids=(bert_tokens_0 * lens_0).long(),
        #                             attention_mask=(bert_masks_0 * lens_0).long(),
        #                             token_type_ids=(token_types_0 * lens_0).long())  # not sure

        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        # print(bert_output_0[1].size())
        # print(bert_output_0[1])
        len_0 = length[:, 0, 0].view([-1, 1])
        lstm_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])

        # mean_input = bert_output_0[1] * len_0
        # mean_input = bert_output_0[1]
        # print('mean_input')
        # print(mean_input)
        # print(mean_input.size)

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)
        # mean_input = bert_output_1[1] * len_1 + mean_input

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)

        print('lstm part')
        print(lstm_input.shape)
        # print(lstm_input[:,:,7])

        lstm_input = lstm_input.transpose(1, 2).contiguous()
        # len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        # len_all = len_0 + len_1 + len_2 + len_3
        # print(mean_input)
        # print(len_all)
        # if sum(sum(len_all)) == 0:
        #     print('error')

        # mean_input = mean_input / len_all
        # print(lstm_input.shape)

        # output, (hn, cn) = self.lstm_layer(lstm_input)
        out = self.transformer_encoder(lstm_input)
        print('transformer_input')
        print(out.shape)
        # axa = self.fc2(mean_input)
        # print('axa')
        # print(axa)
        # mean_output = F.leaky_relu(self.fc2(mean_input))
        # print('output')
        # print(mean_output)
        # print(output.shape)
        # print(output[:, -1, :].shape)

        # hidden = torch.cat([hn[0], hn[-1]], dim=1)

        # output = F.relu(self.fc1(lstm_output))
        transformer_output = F.leaky_relu(self.fc2(out.flatten(start_dim=1)))
        print('transformer_output')
        print(transformer_output.shape)
        # print(output.shape)

        # return mean_output
        return transformer_output

    # def get_length_matrix(self, length):
    #     matrix = torch.zeros([self.batch_size, 8])
    #     for i in range(self.batch_size):
    #         if length[i] > 8:
    #             lens = 8
    #         else:
    #             lens = length[i]
    #         for j in range(lens):
    #             matrix[i][j] = 1
    #     return matrix

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_mean_model(nn.Module):
    def __init__(self, configs):
        super(bert_mean_model, self).__init__()
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_1 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_2 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_3 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_4 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_5 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_6 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_7 = BertModel.from_pretrained(configs.bert_cache_path)
        self.fc2 = torch.nn.Linear(configs.bert_output_dim, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type, length):
        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        mean_input = bert_output_0[1] * len_0

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        mean_input = bert_output_1[1] * len_1 + mean_input

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        mean_input = bert_output_2[1] * len_2 + mean_input

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        mean_input = bert_output_3[1] * len_3 + mean_input

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        mean_input = bert_output_4[1] * len_4 + mean_input

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        mean_input = bert_output_5[1] * len_5 + mean_input

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        mean_input = bert_output_6[1] * len_6 + mean_input

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        mean_input = bert_output_7[1] * len_7 + mean_input

        len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        if sum(sum(len_all)) == 0:
            print('error')
        mean_input = mean_input / len_all
        mean_output = F.leaky_relu(self.fc2(mean_input))
        return mean_output

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_gat_model(nn.Module):
    def __init__(self, configs):
        super(bert_gat_model, self).__init__()
        self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.dropout = configs.gat_dropout
        self.alpha = configs.gat_alpha
        self.n_heads = configs.gat_head
        self.attentions = [GraphAttentionLayer(self.hidden_size, self.hidden_size, self.dropout, self.alpha) for _ in
                           range(self.n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.GAL = GraphAttentionLayer(self.n_heads * self.hidden_size, self.hidden_size, self.dropout, self.alpha)

        # self.bert_1 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_2 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_3 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_4 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_5 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_6 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_7 = BertModel.from_pretrained(configs.bert_cache_path)
        self.fc1 = torch.nn.Linear(configs.bert_output_dim, configs.num_classes, bias=False)
        self.fc2 = torch.nn.Linear(self.pieces_size * configs.bert_output_dim, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type, length, graph_adj):
        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        gat_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        gat_input = torch.cat((gat_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        gat_input = torch.cat((gat_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        gat_input = torch.cat((gat_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        gat_input = torch.cat((gat_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        gat_input = torch.cat((gat_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        gat_input = torch.cat((gat_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        gat_input = torch.cat((gat_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)
        gat_input = gat_input.transpose(1, 2).contiguous()  # [batch size, 8, 768]
        for i in range(gat_input.shape[0]):
            print(i)
            gat_input_one = F.dropout(gat_input[i], self.dropout, training=self.training)
            gat_input_one = torch.cat([att(gat_input_one, graph_adj[i]) for att in self.attentions], dim=1)
            gat_input_one = F.dropout(gat_input_one, self.dropout, training=self.training)
            gat_output_one = self.GAL(gat_input_one, graph_adj[i])
            for j in range(self.pieces_size):
                output_one = gat_output_one[j] * length[i, 0, j]
                # output_one = gat_output_one[j]
                if j != 0:
                    # output_one = torch.cat((output_old_one, output_one), 0)  # 拼接，可能会有很多冗余，并且会干扰内容
                    output_one = output_old_one + output_one  # 加起来试试
                output_old_one = output_one
                print(output_old_one.shape)

            gat_output = output_old_one.view(-1, 1)
            if i != 0:
                gat_output = torch.cat((gat_output_old, gat_output), 1)
            gat_output_old = gat_output
            print(gat_output_old.shape)
        gat_output_old = gat_output_old.transpose(0, 1).contiguous()
        print(gat_output_old.shape)

        len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        gat_output_old = gat_output_old / len_all

        gat_output_all = F.leaky_relu(self.fc1(gat_output_old))
        print(gat_output_all.shape)
        return gat_output_all

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_lstm_gat_model(nn.Module):
    def __init__(self, configs):
        super(bert_lstm_gat_model, self).__init__()
        self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        self.dropout = configs.gat_dropout
        self.alpha = configs.gat_alpha
        self.n_heads = configs.gat_head
        self.attentions = [GraphAttentionLayer(self.hidden_size, self.hidden_size, self.dropout, self.alpha) for _ in
                           range(self.n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.GAL = GraphAttentionLayer(self.n_heads * self.hidden_size, self.hidden_size, self.dropout, self.alpha)

        # self.bert_1 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_2 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_3 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_4 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_5 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_6 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_7 = BertModel.from_pretrained(configs.bert_cache_path)
        self.lstm_layer = nn.LSTM(configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True)  # 双向LSTM
        self.fc = nn.Linear(configs.lstm_hidden_dim * 2, self.hidden_size, bias=False)

        self.fc1 = torch.nn.Linear(self.hidden_size, configs.num_classes, bias=False)
        self.fc2 = torch.nn.Linear(self.pieces_size * configs.bert_output_dim, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type, length, graph_adj):
        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        lstm_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])), 2)

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        lstm_input = torch.cat((lstm_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])), 2)
        lstm_input = lstm_input.transpose(1, 2).contiguous()  # [batch size, 8, 768]
        # print(lstm_input.shape)

        lstm_output, (hn, cn) = self.lstm_layer(lstm_input)
        gat_input = F.leaky_relu(self.fc(lstm_output))

        # print(gat_input.shape)
        # print(gat_input[:, -1, :].shape)

        for i in range(gat_input.shape[0]):
            # print(i)
            gat_input_one = F.dropout(gat_input[i], self.dropout, training=self.training)
            gat_input_one = torch.cat([att(gat_input_one, graph_adj[i]) for att in self.attentions], dim=1)
            gat_input_one = F.dropout(gat_input_one, self.dropout, training=self.training)
            gat_output_one = self.GAL(gat_input_one, graph_adj[i])
            for j in range(self.pieces_size):
                # output_one = gat_output_one[j] * length[i, 0, j]
                output_one = gat_output_one[j]
                if j != 0:
                    # output_one = torch.cat((output_old_one, output_one), 0)  # 拼接，可能会有很多冗余，并且会干扰内容
                    output_one = output_old_one + output_one  # 加起来试试
                output_old_one = output_one
                # print(output_old_one.shape)

            gat_output = output_old_one.view(-1, 1)
            if i != 0:
                gat_output = torch.cat((gat_output_old, gat_output), 1)
            gat_output_old = gat_output
            # print(gat_output_old.shape)
        gat_output_old = gat_output_old.transpose(0, 1).contiguous()
        # print(gat_output_old.shape)

        # len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        # gat_output_old = gat_output_old / len_all

        gat_output_all = F.leaky_relu(self.fc1(gat_output_old))
        print(gat_output_all.shape)
        return gat_output_all

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_gru_gat_model(nn.Module):
    def __init__(self, configs):
        super(bert_gru_gat_model, self).__init__()
        self.batch_size = configs.batch_size
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path_Chinese)
        self.dropout = configs.gat_dropout
        self.alpha = configs.gat_alpha
        self.n_heads = configs.gat_head
        self.attentions = [GraphAttentionLayer(self.hidden_size, self.hidden_size, self.dropout, self.alpha) for _ in
                           range(self.n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.GAL = GraphAttentionLayer(self.n_heads * self.hidden_size, self.hidden_size, self.dropout, self.alpha)

        # self.bert_1 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_2 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_3 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_4 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_5 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_6 = BertModel.from_pretrained(configs.bert_cache_path)
        # self.bert_7 = BertModel.from_pretrained(configs.bert_cache_path)
        self.gru_layer_bert = nn.GRU(configs.bert_output_dim, configs.bert_output_dim, num_layers=1, batch_first=True,
                                     bidirectional=True)
        self.gru_layer = nn.GRU(2 * configs.bert_output_dim, configs.lstm_hidden_dim, num_layers=1, batch_first=True,
                                bidirectional=True)  # 双向GRU
        self.fc = nn.Linear(configs.lstm_hidden_dim * 2, self.hidden_size, bias=False)

        self.fc1 = torch.nn.Linear(self.hidden_size, configs.num_classes, bias=False)
        self.fc2 = torch.nn.Linear(self.pieces_size * configs.bert_output_dim, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type, length, graph_adj):
        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        print('part section#########################################################################################')
        print(bert_output_0[0].shape)
        len_0 = length[:, 0, 0].view([-1, 1])
        print(len_0.shape)
        _, hn = self.gru_layer_bert(bert_output_0[0])
        h_0 = hn.view([-1, 2 * self.hidden_size])
        print(h_0.shape)
        gru_input = (h_0 * len_0).view([-1, 2 * self.hidden_size, 1])
        print(gru_input.shape)
        print('part section#########################################################################################')

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        _, hn = self.gru_layer_bert(bert_output_1[0])
        h_1 = hn.view([-1, 2 * self.hidden_size])
        gru_input = torch.cat((gru_input, (h_1 * len_1).view([-1, 2 * self.hidden_size, 1])), 2)

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        _, hn = self.gru_layer_bert(bert_output_2[0])
        h_2 = hn.view([-1, 2 * self.hidden_size])
        gru_input = torch.cat((gru_input, (h_2 * len_2).view([-1, 2 * self.hidden_size, 1])), 2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        _, hn = self.gru_layer_bert(bert_output_3[0])
        h_3 = hn.view([-1, 2 * self.hidden_size])
        gru_input = torch.cat((gru_input, (h_3 * len_3).view([-1, 2 * self.hidden_size, 1])), 2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        _, hn = self.gru_layer_bert(bert_output_4[0])
        h_4 = hn.view([-1, 2 * self.hidden_size])
        gru_input = torch.cat((gru_input, (h_4 * len_4).view([-1, 2 * self.hidden_size, 1])), 2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        _, hn = self.gru_layer_bert(bert_output_5[0])
        h_5 = hn.view([-1, 2 * self.hidden_size])
        gru_input = torch.cat((gru_input, (h_5 * len_5).view([-1, 2 * self.hidden_size, 1])), 2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        _, hn = self.gru_layer_bert(bert_output_6[0])
        h_6 = hn.view([-1, 2 * self.hidden_size])
        gru_input = torch.cat((gru_input, (h_6 * len_6).view([-1, 2 * self.hidden_size, 1])), 2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        _, hn = self.gru_layer_bert(bert_output_7[0])
        h_7 = hn.view([-1, 2 * self.hidden_size])
        gru_input = torch.cat((gru_input, (h_7 * len_7).view([-1, 2 * self.hidden_size, 1])), 2)
        print(gru_input.shape)

        gru_input = gru_input.transpose(1, 2).contiguous()  # [batch size, 8, 768]
        print(gru_input.shape)

        gru_output, (hn, cn) = self.gru_layer(gru_input)
        print(gru_output.shape)

        gat_input = F.leaky_relu(self.fc(gru_output))
        print(gat_input.shape)

        for i in range(gat_input.shape[0]):
            # print(i)
            gat_input_one = F.dropout(gat_input[i], self.dropout, training=self.training)
            gat_input_one = torch.cat([att(gat_input_one, graph_adj[i]) for att in self.attentions], dim=1)
            gat_input_one = F.dropout(gat_input_one, self.dropout, training=self.training)
            gat_output_one = self.GAL(gat_input_one, graph_adj[i])
            for j in range(self.pieces_size):
                # output_one = gat_output_one[j] * length[i, 0, j]
                output_one = gat_output_one[j]
                if j != 0:
                    # output_one = torch.cat((output_old_one, output_one), 0)  # 拼接，可能会有很多冗余，并且会干扰内容
                    output_one = output_old_one + output_one  # 加起来试试
                output_old_one = output_one
                # print(output_old_one.shape)

            gat_output = output_old_one.view(-1, 1)
            if i != 0:
                gat_output = torch.cat((gat_output_old, gat_output), 1)
            gat_output_old = gat_output
            # print(gat_output_old.shape)
        gat_output_old = gat_output_old.transpose(0, 1).contiguous()
        # print(gat_output_old.shape)

        # len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        # gat_output_old = gat_output_old / len_all

        gat_output_all = F.leaky_relu(self.fc1(gat_output_old))
        print(gat_output_all.shape)
        return gat_output_all

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


class bert_transformer_GAT_model(nn.Module):
    def __init__(self, configs):
        super(bert_transformer_GAT_model, self).__init__()
        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.hidden_size = configs.bert_output_dim
        self.bert_0 = BertModel.from_pretrained(configs.bert_cache_path)
        self.dropout = configs.gat_dropout
        self.alpha = configs.gat_alpha
        self.n_heads = configs.gat_head
        self.attentions = [GraphAttentionLayer(self.hidden_size, self.hidden_size, self.dropout, self.alpha) for _ in
                           range(self.n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.GAL = GraphAttentionLayer(self.n_heads * self.hidden_size, self.hidden_size, self.dropout, self.alpha)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc1 = nn.Linear(self.hidden_size, configs.num_classes, bias=False)
        self.fc2 = torch.nn.Linear(self.hidden_size * self.pieces_size, configs.num_classes, bias=False)

    def forward(self, bert_token, bert_mask, token_type, length, graph_adj):
        bert_output_0 = self.bert_0(
            input_ids=(bert_token[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 0] * length[:, :, 0]).view([-1, self.embedding_size]).int())  # not sure
        len_0 = length[:, 0, 0].view([-1, 1])
        transformer_input = (bert_output_0[1] * len_0).view([-1, self.hidden_size, 1])

        bert_output_1 = self.bert_0(
            input_ids=(bert_token[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 1] * length[:, :, 1]).view([-1, self.embedding_size]).int())
        len_1 = length[:, 0, 1].view([-1, 1])
        transformer_input = torch.cat((transformer_input, (bert_output_1[1] * len_1).view([-1, self.hidden_size, 1])),
                                      2)

        bert_output_2 = self.bert_0(
            input_ids=(bert_token[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 2] * length[:, :, 2]).view([-1, self.embedding_size]).int())
        len_2 = length[:, 0, 2].view([-1, 1])
        transformer_input = torch.cat((transformer_input, (bert_output_2[1] * len_2).view([-1, self.hidden_size, 1])),
                                      2)

        bert_output_3 = self.bert_0(
            input_ids=(bert_token[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 3] * length[:, :, 3]).view([-1, self.embedding_size]).int())
        len_3 = length[:, 0, 3].view([-1, 1])
        transformer_input = torch.cat((transformer_input, (bert_output_3[1] * len_3).view([-1, self.hidden_size, 1])),
                                      2)

        bert_output_4 = self.bert_0(
            input_ids=(bert_token[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 4] * length[:, :, 4]).view([-1, self.embedding_size]).int())
        len_4 = length[:, 0, 4].view([-1, 1])
        transformer_input = torch.cat((transformer_input, (bert_output_4[1] * len_4).view([-1, self.hidden_size, 1])),
                                      2)

        bert_output_5 = self.bert_0(
            input_ids=(bert_token[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 5] * length[:, :, 5]).view([-1, self.embedding_size]).int())
        len_5 = length[:, 0, 5].view([-1, 1])
        transformer_input = torch.cat((transformer_input, (bert_output_5[1] * len_5).view([-1, self.hidden_size, 1])),
                                      2)

        bert_output_6 = self.bert_0(
            input_ids=(bert_token[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 6] * length[:, :, 6]).view([-1, self.embedding_size]).int())
        len_6 = length[:, 0, 6].view([-1, 1])
        transformer_input = torch.cat((transformer_input, (bert_output_6[1] * len_6).view([-1, self.hidden_size, 1])),
                                      2)

        bert_output_7 = self.bert_0(
            input_ids=(bert_token[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            attention_mask=(bert_mask[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int(),
            token_type_ids=(token_type[:, :, 7] * length[:, :, 7]).view([-1, self.embedding_size]).int())
        len_7 = length[:, 0, 7].view([-1, 1])
        transformer_input = torch.cat((transformer_input, (bert_output_7[1] * len_7).view([-1, self.hidden_size, 1])),
                                      2)
        transformer_input = transformer_input.transpose(1, 2).contiguous()
        transformer_output = self.transformer_encoder(transformer_input)
        gat_input = F.leaky_relu(transformer_output)

        for i in range(gat_input.shape[0]):
            gat_input_one = F.dropout(gat_input[i], self.dropout, training=self.training)
            gat_input_one = torch.cat([att(gat_input_one, graph_adj[i]) for att in self.attentions], dim=1)
            gat_input_one = F.dropout(gat_input_one, self.dropout, training=self.training)
            gat_output_one = self.GAL(gat_input_one, graph_adj[i])
            for j in range(self.pieces_size):
                # output_one = gat_output_one[j] * length[i, 0, j]
                output_one = gat_output_one[j]
                if j != 0:
                    # output_one = torch.cat((output_old_one, output_one), 0)  # 拼接，可能会有很多冗余，并且会干扰内容
                    output_one = output_old_one + output_one  # 加起来试试
                output_old_one = output_one

            gat_output = output_old_one.view(-1, 1)
            if i != 0:
                gat_output = torch.cat((gat_output_old, gat_output), 1)
            gat_output_old = gat_output
        gat_output_old = gat_output_old.transpose(0, 1).contiguous()

        # len_all = len_0 + len_1 + len_2 + len_3 + len_4 + len_5 + len_6 + len_7
        # gat_output = gat_output_old / len_all
        # transformer_GAT_output = F.leaky_relu(self.fc1(gat_output))

        transformer_GAT_output = F.leaky_relu(self.fc1(gat_output_old))

        # transformer_GAT_output = F.leaky_relu(self.fc2(gat_input.flatten(start_dim=1)))

        return transformer_GAT_output

    # def get_length_matrix(self, length):
    #     matrix = torch.zeros([self.batch_size, 8])
    #     for i in range(self.batch_size):
    #         if length[i] > 8:
    #             lens = 8
    #         else:
    #             lens = length[i]
    #         for j in range(lens):
    #             matrix[i][j] = 1
    #     return matrix

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.focal_loss_gamma = configs.focal_loss_gamma
#         self.focal_loss_alpha = configs.focal_loss_alpha
#         self.num_classes = configs.num_classes
#
#     def forward(self, inputs, targets):
#         targets = F.one_hot(targets, num_classes=self.num_classes)
#         # 计算正负样本权重
#         alpha_factor = torch.ones(targets.shape) * self.focal_loss_alpha
#         alpha_factor = torch.where(torch.eq(targets, 1), alpha_factor, 1. - alpha_factor)
#         # 计算因子项
#         focal_weight = torch.where(torch.eq(targets, 1), 1. - inputs, inputs)
#         # 得到最终的权重
#         focal_weight = alpha_factor * torch.pow(focal_weight, self.focal_loss_gamma)
#         targets = targets.type(torch.FloatTensor)
#         # 计算标准交叉熵
#         bce = -(targets * torch.log(inputs) + (1. - targets) * torch.log(1. - inputs))
#         # focal loss
#         cls_loss = focal_weight * bce
#         return cls_loss.sum()


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
