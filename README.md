# ICDM-Submission

Here is the code and data in the paper: LCMDC: Large-scale Chinese Medical Dialogue Corpora for Automatic Triage and Medical Consultation.

Because the data is too large, you can access it at this link: [WeTransfer Download Link](https://we.tl/t-scDvwQ32W1)

## Dataset Description

- **[data_all_raw.csv](./dataset/data_all_raw.csv)**: Contains the raw data, including patients' questions, doctors' responses, different level labels, and other basic information.

- **[Intelligent Triage System](./dataset/triage)**: Data are in the `data/triage` folder. There are two levels of labels. For the prompt learning method, categories with fewer samples are selected to form new datasets.

- **[Medical Consultant System](./dataset/consultation)**: Data are in the `data/consultation` folder.
  - Files starting with '[med_qa](./dataset/consultation/)' contain raw questions and responses, concatenated by 'tab'.
  - Files starting with '[med_Ga](./dataset/consultation/)' contain supplementary information from the knowledge graph, formatted as 'question\tinformation\tresponse'.
  - '[med_Ga_train.txt](./dataset/consultation/med_Ga_train.txt)' and '[med_qa_train.txt](./dataset/consultation/med_qa_train.txt)' refer to the training set, while the others refer to the testing set.
