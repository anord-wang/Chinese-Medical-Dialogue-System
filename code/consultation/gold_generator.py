for i in range(5):
    f = open("./data/med_qa_1000_"+ str(i+1) + '.txt', "r", encoding="gbk")
    data = f.readlines()
    
    dir_gold = "/data0/wxy/gpt2/sample/med_qa_test/#gold/"
    samples_file = open(dir_gold + 'med_qa_gold_' + str(i+1) + '.txt', 'a', encoding='utf8')
    
    for j in range(len(data)):
        text = data[j].split('  ')[1].strip()
        print(text)
        samples_file.write("{}\n".format(text))
    
    samples_file.close()
    f.close()