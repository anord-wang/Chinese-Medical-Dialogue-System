import numpy as np

with open('/data1/mx/data/gxy_2016_data2/gxy_2016_data2_dataset.txt', 'r',
          encoding='utf-8') as f:
    raw_texts = f.readlines()
    f.close()

texts = []
for r in raw_texts:
    texts.append(r.replace(' ', ''))

train_texts = texts[:int(0.6 * len(texts))]
test_texts = texts[int(0.6 * len(texts)):]

with open('/data1/mx/data/gxy_2016_data2/train_texts.txt', 'w') as f:
    f.writelines(train_texts)
    f.close()

with open('/data1/mx/data/gxy_2016_data2/test_texts.txt', 'w') as f:
    f.writelines(test_texts)
    f.close()