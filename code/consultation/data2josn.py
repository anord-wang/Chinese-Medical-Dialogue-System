#融合两个txt成一个json 
#path1 = 'E:/data/1-2021fin_data/lcsts_fin/new/lcsts_part3_act.txt'
#path2 = 'E:/data/1-2021fin_data/lcsts_fin/new/lcsts_part3_sum.txt'

#f = open(path1,'r',encoding='utf-8')
#s = open(path2,'r',encoding='utf-8')
import pandas as pd
import random
import json

data_file = '/data0/wxy/gpt2/data/data_all_raw.csv'
df_all = pd.read_csv(data_file, encoding='gbk', header=0, low_memory=False)

info_sum = 0
qasum = 0
question_list = []
answer_list = []
for i in range(int(df_all.shape[0])):
    question = str(df_all.iloc[i, 6])
    question = question.replace('健康咨询描述：', '').strip()
    if question != '':
        info_sum = info_sum + 1
        answer = str(df_all.iloc[i, 7])
        if answer != 'NO_answer':
            answer = answer.replace('病情分析：', '')
            answer = answer.replace('指导意见：', '')
            answer.strip()
            question_list.append(question)
            answer_list.append(answer)          
            qasum = qasum + 1
print(qasum)

list_all = random.sample(range(0, qasum), 5000)

train_question_list = []
train_answer_list = []  
test_question_list = []
test_answer_list = [] 
for i in range(qasum):
    if i in list_all:
        test_question_list.append(question_list[i])
        test_answer_list.append(answer_list[i])
    else:
        train_question_list.append(question_list[i])
        train_answer_list.append(answer_list[i])
        
print('test_question_list:',len(test_question_list))
print('test_answer_list:',len(test_answer_list))
print('train_question_list:',len(train_question_list))
print('train_answer_list:',len(train_answer_list))
    
for i,j in zip(train_question_list,train_answer_list):
    dic ={}
    dic["med_question"] = i
    dic["med_answer"] = j
    with open('/data0/wxy/gpt2/data/med_qa_train.json','a',encoding='utf-8') as file:
        json.dump(dic,file,ensure_ascii=False)
        file.write('\n')
    
for i,j in zip(test_question_list,test_answer_list):
    dic ={}
    dic["med_question"] = i
    dic["med_answer"] = j  
    with open('/data0/wxy/gpt2/data/med_qa_test.json','a',encoding='utf-8') as file:
        json.dump(dic,file,ensure_ascii=False)
        file.write('\n')