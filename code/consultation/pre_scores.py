#-*- codeing = utf-8 -*-
import os
import re
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

dir_raw = "/data0/wxy/gpt2/sample/med_qa_test/0_raw_gpt2/"
dir_none = "/data0/wxy/gpt2/sample/med_qa_test/1_none/"
dir_graph_wo = "/data0/wxy/gpt2/sample/med_qa_test/2_only_graph/without_supply/"
dir_graph_w = "/data0/wxy/gpt2/sample/med_qa_test/2_only_graph/with_supply/"
dir_pretrain = "/data0/wxy/gpt2/sample/med_qa_test/3_only_pretrain/"
dir_pretrain_graph_wo = "/data0/wxy/gpt2/sample/med_qa_test/4_pretrain_graph/without_supply"
dir_pretrain_graph_w = "/data0/wxy/gpt2/sample/med_qa_test/4_pretrain_graph/with_supply"

dir_list = []
dir_title = "/data0/wxy/gpt2/sample/med_qa_test/"
dir_name_1 = ['1_none', '3_only_pretrain']
dir_name_2 = ['2_only_graph', '4_pretrain_graph']
for name1 in dir_name_1:
    dir_list.append(dir_title + name1 + '/')
for name2 in dir_name_2:
    dir_list.append(dir_title + name2 + '/without_supply/')
    dir_list.append(dir_title + name2 + '/with_supply/')
# 此时dir_list共应有6个
print('dir_list', len(dir_list))

file_list = ['/data0/wxy/gpt2/sample/med_qa_test/0_raw_gpt2/']


for dirs in dir_list:
    file_list.append(dirs + 'epoch1/')
    file_list.append(dirs + 'epoch5/')
    file_list.append(dirs + 'epoch10/')
    file_list.append(dirs + 'epoch20/')
    file_list.append(dirs + 'min_ppl_model/')
# 此时file_list共应有31个
print('file_list', len(file_list))


# 共计31个文件夹，改写31次
for file_dir in file_list:  
    # 依次改写1-5的文件，首先构建文件列表
    for i_gen in range(5):
        remote_file = file_dir + 'med_qa_test_' + str(i_gen+1) + '.txt'
        f_original = open(remote_file, "r", encoding="utf8")
        data_original = f_original.readlines()
        
        new_file = file_dir + 'med_dialogue_test_' + str(i_gen+1) + '.txt'
        #f_new = open(new_file, 'a', encoding='utf8')
        f_new = open(new_file, 'w', encoding='utf8')
    
        for j in range(len(data_original)):
            text = data_original[j].strip()
            if u'\u4e00' <= text <= u'\u9fff':
                text = text
            else:
                text = '您好，我是您的智能医疗助理，希望可以帮到您。如果没答上来，请更加详细的描述问题哦。祝您身体棒棒！'
            #text = text.replace('.', '。', 3)
            print(text)
            if j == len(data_original) - 1:
                f_new.write(text)
            else:
                f_new.write("{}\n".format(text))
        
        f_new.close()
        f_original.close()
        
        
        
        
    