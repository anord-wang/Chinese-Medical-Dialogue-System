from transformers import BertTokenizerFast, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained('model/bart_dialogue_base_Chinese')

tokenizer = BertTokenizerFast(vocab_file='vocab/vocab2.txt', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")

f = open("./data/med_qa_1000_5.txt", "r", encoding="gbk")
data = f.readlines()

f_graph = open("./data/med_Ga_1000_5.txt", "r", encoding="gb18030", errors='ignore')
data_graph = f_graph.readlines()

question_list = []

for i in range(len(data)):
    text = data[i].split('  ')[0].strip()
    print(text)
    graph_text = data_graph[i].strip()
    if graph_text == '您好，我是您的智能医疗助理，希望可以帮到您。如果没答上来，请更加详细的描述问题哦。祝您身体棒棒！':
        graph_text = ''
    text = text + graph_text
    if text == '':
        print('hrghsvrzleirzleirzleirzleirzleirzleirzleiso;dnflovniz')
    inputs = tokenizer(text, max_length=512, return_tensors="pt", padding=True)
    #outputs = model(input_ids=text_ids)
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=512)
    answers = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print('answers',answers)

    # question_list.append(text)



inputs = tokenizer(question_list, max_length=512, return_tensors="pt", padding=True)
text_ids = tokenizer.encode(question_list, add_special_tokens=False, max_length=510)
print(inputs["input_ids"].min())
print(inputs["input_ids"].max())
print(text_ids.min())
print(text_ids.max())

summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=512)
answers = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(answers)