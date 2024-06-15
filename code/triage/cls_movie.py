import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers.data.processors.glue import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample

logger = logging.getLogger(__name__)

# 初始化参数
parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])  # 在jupyter notebook中，args不为空
args.data_dir = "./data/"
args.model_type = "bert"
args.task_name = "sst-2"
args.output_dir = "./outputs2/"
args.max_seq_length = 128
args.do_train = True
args.do_eval = True
args.warmup_steps = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.seed = 1234
args.batch_size = 48
args.n_epochs=3
args.lr = 5e-5

print('args: ', args)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(args.seed)


set_seed(args)  # Added here for reproductibility


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev/test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def load_and_cache_examples(args, processor, tokenizer, set_type):
    # Load data features from cache or dataset file
    print("Creating features from dataset file at {}".format(args.data_dir))
    label_list = processor.get_labels()

    if set_type == 'train':
        examples = (
            processor.get_train_examples(args.data_dir)
        )
    if set_type == 'dev':
        examples = (
            processor.get_dev_examples(args.data_dir)
        )
    if set_type == 'test':
        examples = (
            processor.get_test_examples(args.data_dir)
        )

    features = convert_examples_to_features(
        examples,  # 原始数据
        tokenizer,  #
        label_list=label_list,
        max_length=args.max_seq_length,  # 设置每个batch 最大句子长度
        output_mode='classification',  # 设置分类标记
        pad_on_left=False,  # 右侧进行padding
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,  # bert 分类设置0
        mask_padding_with_zero=True  # the attention mask will be filled by ``1``
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, data_loader, criterion, optimizer):
    epoch_acc = 0.
    epoch_loss = 0.
    total_batch = 0
    model.train()

    for batch in data_loader:
        #
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch
        # 预测
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)[0]
        # 计算loss和acc
        loss = criterion(outputs, labels)
        _, y = torch.max(outputs, dim=1)
        acc = (y == labels).float().mean()

        #
        if total_batch % 100==0:
            print('Iter_batch[{}/{}]:'.format(total_batch, len(data_loader)),
                  'Train Loss: ', "%.3f" % loss.item(), 'Train Acc:', "%.3f" % acc.item())

        # 计算批次下总的acc和loss
        epoch_acc += acc.item()  # 当前批次准确率
        epoch_loss += loss.item()  # 当前批次loss

        # 剃度下降
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()  # Update learning rate schedule
        model.zero_grad()

        total_batch += 1
        # break
    return epoch_acc / len(data_loader), epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    epoch_acc = 0.
    epoch_loss = 0.

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            #
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, token_type_ids, labels = batch
            # 预测
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)[0]
            # 计算loss和acc
            loss = criterion(outputs, labels)
            _, y = torch.max(outputs, dim=1)
            acc = (y == labels).float().mean()

            # 计算批次下总的acc和loss
            epoch_acc += acc.item()  # 当前批次准确率
            epoch_loss += loss.item()  # 当前批次loss
    return epoch_acc / len(data_loader), epoch_loss / len(data_loader)


def main():
    # 1. 定义数据处理器
    processor = Sst2Processor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    print('label_list: ', label_list)
    print('num_label: ', num_labels)
    print('*' * 60)
    # 2. 加载数据集
    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)  # 每条记录封装InputExample 类实例对象
    test_examples = processor.get_test_examples(args.data_dir)  # 每条记录封装InputExample 类实例对象

    print('训练集记录数:', len(train_examples))
    print('验证集数据记录数：', len(dev_examples))
    print('测试数据记录数：', len(test_examples))
    print('训练数据数据举例：\n', train_examples[0])
    print('验证集数据数据举例：\n', dev_examples[0])
    print('测试集数据数据举例：\n', test_examples[0])

    # 3.加载本地 词表 模型
    bert_path = './bert_pretrain/'
    tokenizer = BertTokenizer.from_pretrained( os.path.join(bert_path, 'vocab.txt') )
    train_dataset = load_and_cache_examples(args, processor, tokenizer, 'train')
    dev_dataset = load_and_cache_examples(args, processor, tokenizer, 'dev')
    test_dataset = load_and_cache_examples(args, processor, tokenizer, 'test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. 模型定义
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(bert_path, 'pytorch_model.bin'),
        config=os.path.join(bert_path, 'config.json'))
    model.to(device)

    N_EPOCHS = args.n_epochs
    # 梯度更新算法AdamW
    t_total = len(train_dataloader) // N_EPOCHS
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      correct_bias=False)
    # loss_func
    criterion = nn.CrossEntropyLoss()

    # 5. 模型训练
    print('模型训练开始： ')
    logger.info("***** Running training *****")
    logger.info("  train num examples = %d", len(train_dataloader))
    logger.info("  dev num examples = %d", len(dev_dataloader))
    logger.info("  test num examples = %d", len(test_dataloader))
    logger.info("  Num Epochs = %d", args.n_epochs)

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_acc, train_loss = train(model, train_dataloader, criterion, optimizer)
        val_acc, val_loss = evaluate(model, dev_dataloader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_valid_loss:
            print('loss increasing->')
            best_valid_loss = val_loss
            torch.save(model.state_dict(), 'bert-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.3f}%')

    # evaluate
    model.load_state_dict(torch.load('bert-model.pt'))
    test_acc, test_loss = evaluate(model, test_dataloader, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


if __name__ == '__main__':
    main()
