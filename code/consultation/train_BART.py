from ipywidgets import IntProgress
import tqdm 
from datasets import load_dataset
import lawrouge
import datasets
import random
import pandas as pd
from datasets import dataset_dict
import datasets
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import warnings
from pathlib import Path
from typing import List, Tuple, Union

from torch import nn

import numpy as np
import lawrouge
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging
import json
from datasets import load_dataset,Dataset,DatasetDict,load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer,BertConfig
import os
import lawrouge
from transformers.utils import logging
import numpy as np
from torchinfo import summary

os.environ['CUDA_VISIBLE_DEVICES']="1,2"

model_name = '/data0/wxy/gpt2/model/bart_base_Chinese'
#model_name = '/data0/lhz/experiment/bart_fin/results_new_large/fin_data/1-checkpoint-280000'
metric = load_metric('/data0/wxy/gpt2/rouge.py')

TokenModel = "/data0/wxy/bert/medical/pretrain/macbert/all"
data = load_dataset('json',data_files='/data0/wxy/gpt2/data/med_qa_train.json')
data1 = load_dataset('json',data_files='/data0/wxy/gpt2/data/med_qa_test.json')

def preprocess_data(example):
	return {
		"document":example["med_question"],
		"answer":example["med_answer"],
		"id":"0"
	}

dataset = data["train"].map(preprocess_data, remove_columns=["med_question", "med_answer"])
dataset = dataset.shuffle()

dataset_test = data1["train"].map(preprocess_data, remove_columns=["med_question", "med_answer"])


tokenizer = AutoTokenizer.from_pretrained(TokenModel)
config = BertConfig.from_pretrained(TokenModel)

max_input_length = 256
max_target_length = 128

def preprocess_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["answer"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# raw_datasets = dataset
# tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
 
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.01,shuffle=True,seed=42).values()
test_data_tex = dataset_test
#train_data_txt, test_data_tex = train_data_txt.train_test_split(test_size=0.01,shuffle=True,seed=42).values()
# 装载数据
dd = DatasetDict({"train":train_data_txt,"validation": validation_data_txt,"test":test_data_tex }) 
 
raw_datasets = dd
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

print(tokenized_datasets)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
 

logger = logging.get_logger(__name__)

batch_size = 8
args = Seq2SeqTrainingArguments(
    output_dir="/data0/wxy/gpt2/bart_output",
    num_train_epochs= 8,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,  # demo
    per_device_eval_batch_size=batch_size,
    learning_rate=5e-05,
    warmup_steps=5000,
    weight_decay=0.001,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=20000,
    evaluation_strategy="steps",
    metric_for_best_model="eval_rouge1",
    eval_steps = 20000,
    save_total_limit=2,
    save_steps=20000,
    # generation_max_length最大生成长度，系统默认20 generation_num_beams=1表示贪心解码，大于1为树搜索
    generation_max_length=128,
    generation_num_beams=5,
)
 
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # print(labels)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(pred.replace(" ", "")) for pred in decoded_preds]

    decoded_labels = ["\n".join(label.replace(" ", "")) for label in decoded_labels]
    
    # output summaries on test set
    with open("/data0/wxy/gpt2/bart_output/result/test_output.txt","w") as f: 
        for i in decoded_preds:
            # print(i)
            f.write(i.replace("\n","")+"\n")

    with open("/data0/wxy/gpt2/bart_output/result/real.txt","w") as f: 
        for i in decoded_labels:
            # print(i)
            f.write(i.replace("\n","")+"\n")

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}
# # 这里用的是中文lawrouge 至于字符级还是词级计算看自己调整 这里是字符级
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
#     decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]
#     # Rouge with jieba cut
#     # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
#     # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
 
#     labels_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in labels]
#     # length = len(prediction_lens)
 
#     # print(decoded_preds)
#     # print(decoded_labels)
#     rouge = lawrouge.Rouge()
 
#     result = rouge.get_scores(decoded_preds, decoded_labels,avg=True)
#     # print(result)
#     print(result)
#     result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}
 
#     result = {key: value * 100 for key, value in result.items()}
#     return result
 
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

summary(model)
train_result = trainer.train()
print(train_result)

out = trainer.predict(tokenized_datasets["test"],num_beams=5)

predictions, labels ,metric= out

print(metric)

trainer.save_model()
metrics = train_result.metrics
trainer.log_metrics("train",metrics)
trainer.save_metrics("train",metrics)
trainer.save_state()

decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after e ach sentence
decoded_preds = [" ".join(pred.replace(" ", "")) for pred in decoded_preds]
decoded_labels = [" ".join(pred.replace(" ", "")) for label in decoded_labels]


# output summaries on test set
with open("/data0/wxy/gpt2/bart_output/result/test_output.txt","w") as f: 
    for i in decoded_preds:
        # print(i)
        f.write(i.replace("\n","")+"\n")

with open("/data0/wxy/gpt2/bart_output/result/real.txt","w") as f: 
    for i in decoded_labels:
        # print(i)
        f.write(i.replace("\n","")+"\n")