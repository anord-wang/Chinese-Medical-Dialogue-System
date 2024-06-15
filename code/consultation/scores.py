#-*- codeing = utf-8 -*-
import lawrouge
import files2rouge
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.ribes_score import sentence_ribes
from bert_score import score
from bart_score import BARTScorer
import pyter
from fast_bleu import BLEU, SelfBLEU
from cider.cider import Cider
from meteor.meteor import Meteor

from collections import defaultdict
import os
import numpy as np

import spacy
import wmd
from stsmd import SMD
from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

smooth = SmoothingFunction() 
# nlp = spacy.load('en_core_web_md')
# nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
nlp = spacy.load("zh_core_web_sm", exclude=("tagger", "parser", "senter", "attribute_ruler", "ner"))
# nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)

def entropy(predicts):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in predicts:
        g = gg.rstrip().split()
        for n in range(4):
            # print('---n: ', n)
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                # print('----ngram: ', ngram)
                counter[n][ngram] += 1
    # print('---counter: ', counter)
    for n in range(4):
        # print('---scores n: ', n)
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score
    
def calc_diversity(predicts):
    '''
    生成结果加入空格
    '''
    tokens = [0.0, 0.0]
    types = [defaultdict(int), defaultdict(int)]
    for gg in predicts:
        g = gg.rstrip().split()
        for n in range(2):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys())/tokens[0]
    div2 = len(types[1].keys())/tokens[1]
    return [div1, div2]


def kl_divergence(labels, predicts):
    '''
    从predict的角度看，与labels有多大不同，Dkl=sigma P_p * log(P_p / P_l)
    '''
    counter_label = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in labels:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter_label[n][ngram] += 1
 
    counter_predict = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in predicts:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter_predict[n][ngram] += 1
    # Dkl=sigma P_p * log(P_p / P_l) 从p的角度看与l有多大不同
    kv_score = [0.0, 0.0, 0.0, 0.0]
    for n in range(4):
        total_label = sum(counter_label[n].values()) + 1e-10
        total_predict = sum(counter_predict[n].values()) + 1e-10
        for k in counter_label[n].keys():
            p_l = (counter_label[n][k] + 0.0) / total_label
            p_p = 0.0
            if k in counter_predict[n].keys():
                p_p = (counter_predict[n][k] + 0.0) / total_predict
            kv_score[n] += p_p * np.log((p_p + 1e-10) / p_l)
    return kv_score


# f = open("./sample/samples.txt","r", encoding="utf-8")
# output = f.readlines()

# dir_raw = "/data0/wxy/gpt2/sample/med_qa_test/0_raw_gpt2/"
# dir_none = "/data0/wxy/gpt2/sample/med_qa_test/1_none/"
# dir_graph_wo = "/data0/wxy/gpt2/sample/med_qa_test/2_only_graph/without_supply/"
# dir_graph_w = "/data0/wxy/gpt2/sample/med_qa_test/2_only_graph/with_supply/"
# dir_pretrain = "/data0/wxy/gpt2/sample/med_qa_test/3_only_pretrain/"
# dir_pretrain_graph_wo = "/data0/wxy/gpt2/sample/med_qa_test/4_pretrain_graph/without_supply"
# dir_pretrain_graph_w = "/data0/wxy/gpt2/sample/med_qa_test/4_pretrain_graph/with_supply"

dir_list = []
dir_title = "/data0/wxy/gpt2/sample/med_qa_test/"
#dir_name_1 = ['1_none', '3_only_pretrain']
#dir_name_2 = ['2_only_graph', '4_pretrain_graph']
dir_name_2 = ['4_pretrain_graph']
#for name1 in dir_name_1:
#    dir_list.append(dir_title + name1 + '/')
for name2 in dir_name_2:
    #dir_list.append(dir_title + name2 + '/without_supply/')
    dir_list.append(dir_title + name2 + '/with_supply/')
# 此时dir_list共应有6个
file = open('/data0/wxy/gpt2/sample/result.txt', 'a', encoding='utf-8')
print('dir_list:', len(dir_list))
print('dir_list:', len(dir_list), file=file)

file_list = ['/data0/wxy/gpt2/sample/med_qa_test/0_raw_gpt2/']

for dirs in dir_list:
    #for i in range(3, len(dir_list)):
    file_list.append(dirs + 'epoch1/')
    file_list.append(dirs + 'epoch5/')
    file_list.append(dirs + 'epoch10/')
    file_list.append(dirs + 'epoch20/')
    file_list.append(dirs + 'min_ppl_model/')
    #file_list.append(dir_list[i] + 'epoch1/')
    #file_list.append(dir_list[i] + 'epoch5/')
    #file_list.append(dir_list[i] + 'epoch10/')
    #file_list.append(dir_list[i] + 'epoch20/')
    #file_list.append(dir_list[i] + 'min_ppl_model/')
# 此时file_list共应有31个
print('file_list:', len(file_list))
print('file_list:', len(file_list), file=file)

# 构建gold文件列表
dir_gold = "/data0/wxy/gpt2/sample/med_qa_test/#gold/"
gold_list = []
for i_gold in range(5):
    gold_list.append(dir_gold + 'med_qa_gold_' + str(i_gold+1) + '.txt')

# 共计31个文件夹，评价31次
for file_dir in file_list: 
    print('==========================================================================================================================')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('==========================================================================================================================')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('==========================================================================================================================')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('file_dir:', file_dir)
    print('file_dir:', file_dir, file=file)
    generation_list = []
    for i in range(5):
        # 依次对比1-5的文件，首先构建文件列表
        generation_file = file_dir + 'med_dialogue_test_' + str(i + 1) + '.txt'
        print('i:', i)
        print('i:', i, file=file)
        print('generation_file:', generation_file)
        print('generation_file:', generation_file, file=file)
        
        # lawrouge评价
        files_rouge = lawrouge.FilesRouge()
        scores = files_rouge.get_scores(generation_file, gold_list[i], avg=2, ignore_empty=True)
        print('weighted score: ', scores)
        print('weighted score: ', scores, file=file)
        '''
        # files2rouge评价
        files2rouge.run(generation_file, gold_list[i], ignore_empty_reference = True)
        print('weighted score: ', scores)
        '''
        
        # 逐句评价 BLEU\meteor
        candidate_file = generation_file
        reference_file = gold_list[i]
        candidate_list = []
        reference_list = []
        print(candidate_file)
        print(candidate_file, file=file)
        print(gold_list[i])
        print(gold_list[i], file=file)
        f_candidate = open(candidate_file, "r", encoding="utf8")
        f_reference = open(reference_file, "r", encoding="utf8")
        data_candidate = f_candidate.readlines()
        data_reference = f_reference.readlines()
        bleu_1_gram = 0
        bleu_2_gram = 0
        bleu_3_gram = 0
        bleu_4_gram = 0
        #bleu_corpus_score_2 = 0
        #bleu_corpus_score_4 = 0
        meteor_score = 0
        chrf_score = 0
        gleu_score = 0
        nist_score = 0
        ribes_score = 0
        ter_score = 0
        wmd_score = 0
        smd_score = 0
        for j_bleu in range(len(data_candidate)):
            candidate_text = ' '.join(data_candidate[j_bleu].strip())
            reference_text = ' '.join(data_reference[j_bleu].strip())
            candidate_list.append(candidate_text)
            reference_list.append(reference_text)
            
            # BLEU
            try:
                bleu_1_gram = bleu_1_gram + sentence_bleu(reference_text, candidate_text, weights=(1, 0, 0, 0))
            except KeyError:
                bleu_1_gram = bleu_1_gram
            try:
                bleu_2_gram = bleu_2_gram + sentence_bleu(reference_text, candidate_text, weights=(0.5, 0.5, 0, 0))
            except KeyError:
                bleu_2_gram = bleu_2_gram
            try:
                bleu_3_gram = bleu_3_gram + sentence_bleu(reference_text, candidate_text, weights=(0.33, 0.33, 0.33, 0))
            except KeyError:
                bleu_3_gram = bleu_3_gram
            try:
                bleu_4_gram = bleu_4_gram + sentence_bleu(reference_text, candidate_text, weights=(0.25, 0.25, 0.25, 0.25))
            except KeyError:
                bleu_4_gram = bleu_4_gram
            #bleu_corpus_score_2 = bleu_corpus_score_2 + corpus_bleu(data_reference[j_bleu], data_candidate[j_bleu], weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
            #bleu_corpus_score_4 = bleu_corpus_score_4 + corpus_bleu(data_reference[j_bleu], data_candidate[j_bleu], smoothing_function=smooth.method1)

            # meteor
            meteor_score = meteor_score + single_meteor_score([reference_text], [candidate_text])
            
            # CHRF
            chrf_score = chrf_score + sentence_chrf(reference_text.split(), candidate_text.split())
            
            # GLEU
            gleu_score = gleu_score + sentence_gleu([reference_text.split()], candidate_text.split())
            
            # NIST
            try:
                nist_score = nist_score + sentence_nist([reference_text.split()], candidate_text.split())
            except ZeroDivisionError:
                nist_score = nist_score
            
            # RIBES score
            ribes_score = ribes_score + sentence_ribes([reference_text.split()], candidate_text.split())
            
            # TER
            ter_score = ter_score + pyter.ter(candidate_text.split(), reference_text.split())
            
            # WMD
            doc1 = nlp(data_candidate[j_bleu].strip())
            doc2 = nlp(data_reference[j_bleu].strip())
            wmd_score = wmd_score + doc1.similarity(doc2)
            
            
         
        print('bleu_1_gram:', bleu_1_gram/len(data_candidate))
        print('bleu_2_gram:', bleu_2_gram/len(data_candidate))
        print('bleu_3_gram:', bleu_3_gram/len(data_candidate))
        print('bleu_4_gram:', bleu_4_gram/len(data_candidate))
        print('meteor_score:', meteor_score/len(data_candidate))
        print('chrf_score:', chrf_score/len(data_candidate))
        print('gleu_score:', gleu_score/len(data_candidate))
        print('nist_score:', nist_score/len(data_candidate))
        print('ribes_score:', ribes_score/len(data_candidate))
        print('ter_score:', ter_score/len(data_candidate))
        print('wmd_score:', wmd_score/len(data_candidate))
        print('smd_score:', smd_score/len(data_candidate))
        
        print('bleu_1_gram:', bleu_1_gram/len(data_candidate), file=file)
        print('bleu_2_gram:', bleu_2_gram/len(data_candidate), file=file)
        print('bleu_3_gram:', bleu_3_gram/len(data_candidate), file=file)
        print('bleu_4_gram:', bleu_4_gram/len(data_candidate), file=file)
        print('meteor_score:', meteor_score/len(data_candidate), file=file)
        print('chrf_score:', chrf_score/len(data_candidate), file=file)
        print('gleu_score:', gleu_score/len(data_candidate), file=file)
        print('nist_score:', nist_score/len(data_candidate), file=file)
        print('ribes_score:', ribes_score/len(data_candidate), file=file)
        print('ter_score:', ter_score/len(data_candidate), file=file)
        print('wmd_score:', wmd_score/len(data_candidate), file=file)
        print('smd_score:', smd_score/len(data_candidate), file=file)
        
        
        ## BART相似度
        #bart_scorer = BARTScorer(device='cuda:0', checkpoint='/data0/wxy/gpt2/model/bart_base_Chinese')
        #bart_scorer.load(path='/data0/wxy/gpt2/model/bart_dialogue_base_Chinese/pytorch_model.bin')
        #bart_similiraty_score = bart_scorer.score(candidate_list, reference_list, batch_size=4)
        #print('bart_similiraty_score', bart_similiraty_score[0])
        
        # BERT score
        P, R, F1 = score(candidate_list, reference_list, lang="zh", verbose=True)
        print(f"System level F1 score: {F1.mean():.3f}") 
        print(f"System level P score: {P.mean():.3f}") 
        print(f"System level R score: {R.mean():.3f}")

        print(f"System level F1 score: {F1.mean():.3f}", file=file) 
        print(f"System level P score: {P.mean():.3f}", file=file) 
        print(f"System level R score: {R.mean():.3f}", file=file)        
        
        # ENTROPY 信息熵; diversity 词语丰富度; KL散度
        generation_predict = []
        for j_ENTROPY in range(len(data_candidate)):
            candidate_text = ' '.join(data_candidate[j_ENTROPY].strip())
            generation_predict.append(candidate_text)
        
        etp_score, div_score = entropy(generation_predict)
        [div1, div2] = calc_diversity(generation_predict)
        kl_d = kl_divergence(reference_list, candidate_list)

        print(f"System level etp_score: {etp_score}") 
        print(f"System level div_score: {div_score}") 
        print(f"System level div1: {div1:.3f}") 
        print(f"System level div2: {div2:.3f}") 
        print(f"System level kl_divergence: {kl_d}") 
        
        print(f"System level etp_score: {etp_score}", file=file) 
        print(f"System level div_score: {div_score}", file=file) 
        print(f"System level div1: {div1:.3f}", file=file) 
        print(f"System level div2: {div2:.3f}", file=file) 
        print(f"System level kl_divergence: {kl_d}", file=file) 
        

        # self_bleu
        weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
        self_bleu = SelfBLEU(generation_predict, weights)
        self_bleu.get_score()
        self_bleu_score = self_bleu.get_score()
        # print('self_bleu_score:', self_bleu_score)
        # print('self_bleu_score:', self_bleu_score['bigram'])
        bigram_score_list = self_bleu_score['bigram']
        trigram_score_list = self_bleu_score['trigram']
        bigram_score_sum = 0
        trigram_score_sum = 0
        for j_sb in range(len(bigram_score_list)):
            bigram_score_sum = bigram_score_sum + bigram_score_list[j_sb]
            trigram_score_sum = trigram_score_sum + trigram_score_list[j_sb]
        bigram_score = bigram_score_sum/len(bigram_score_list)
        trigram_score = trigram_score_sum/len(trigram_score_list)
        print('self_bleu_bigram_score:', bigram_score)
        print('self_bleu_trigram_score:', trigram_score)
        
        print('self_bleu_bigram_score:', bigram_score, file=file)
        print('self_bleu_trigram_score:', trigram_score, file=file)
        
        # 文本相似性
        web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction
        # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
        sts_score = 0
        for j_sts in range(len(data_candidate)):
            candidate_text = ' '.join(data_candidate[j_sts].strip())
            reference_text = ' '.join(data_reference[j_sts].strip())
            sts_score = sts_score + web_model.predict([(candidate_text,reference_text)])
        print('sts_score:', sts_score/len(data_candidate))
        print('sts_score:', sts_score/len(data_candidate), file=file)
        
        # cider meteor
        #Meteor_score = Meteor().compute_score(data_candidate, data_reference)
        #Cider_score = Cider().compute_score(data_candidate, data_reference)
        #print('Meteor_score', Meteor_score)
        #print('Cider_score', Cider_score)
        
        #print('Meteor_score', Meteor_score, file=file)
        #print('Cider_score', Cider_score, file=file)
        
        