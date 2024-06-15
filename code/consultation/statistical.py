import numpy as np

len_sum = 0
line_sum = 0
f = open("/data0/wxy/gpt2/data/med_kb.txt")
data = f.readlines()
for i in range(len(data)):
    line_sum = line_sum + 1
    len_sum = len_sum + len(data[i])
f.close()
print(len_sum)
print(line_sum)
print(len_sum/line_sum)

