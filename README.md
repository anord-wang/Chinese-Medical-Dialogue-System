# Chinese Medical Dialogue System

The code and data in the paper: Building a Chinese Medical Dialogue System: Integrating Large-scale Corpora and Novel Models.

Because the data is too large, you can access it at this link: [Zenodo Download Link](https://zenodo.org/records/13771008?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNjU1NDAwMiwiZXhwIjoxNzM1NjAzMTk5fQ.eyJpZCI6IjA4Y2M0MDMyLTE0NTctNGZkZi1iYjAxLTBkZmQyYjRiNzVlZiIsImRhdGEiOnt9LCJyYW5kb20iOiI0OTExZTBhNzIyMjg5NzFhMmJmZWRhN2JmY2E2ZTljZCJ9.l7HobRPQVtt5gWBXs-2AuOsBX5fYViYkqKePsoDAvTmFYAu_1sH-2f1XwtWJJlppEAdd3C0wdWF7MbCtFLP6kA) or [Baidu Disk Download Link](https://pan.baidu.com/s/15XtsqDmzic3nIb6ZIFri4g) with code: **iy3b**

## Dataset Description

- **[data_all_raw.csv](./dataset/data_all_raw.csv)**: Contains the raw data, including patients' questions, doctors' responses, different level labels, and other basic information.

- **[Intelligent Triage System](./dataset/triage)**: Data are in the `data/triage` folder. There are two levels of labels. For the prompt learning method, categories with fewer samples are selected to form new datasets.

- **[Medical Consultant System](./dataset/consultation)**: Data are in the `data/consultation` folder.
  - Files starting with '[med_qa](./dataset/consultation/)' contain raw questions and responses, concatenated by 'tab'.
  - Files starting with '[med_Ga](./dataset/consultation/)' contain supplementary information from the knowledge graph, formatted as 'question\tinformation\tresponse'.
  - '[med_Ga_train.txt](./dataset/consultation/med_Ga_train.txt)' and '[med_qa_train.txt](./dataset/consultation/med_qa_train.txt)' refer to the training set, while the others refer to the testing set.
