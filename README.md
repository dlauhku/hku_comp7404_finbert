# hku_comp7404_finbert

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necessary packages and load tensorboard extension.
```bash
pip install transformers
pip install --upgrade datasets
pip install pysentiment2
%load_ext tensorboard
```

## check GPU type
```python
print('='*40 + '\nChecking GPU resource type...\n' + '-'*40)
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
```

## check RAM resources
```python
print('\n' + '='*40 + '\nChecking RAM resources...\n' + '-'*40)
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
```

## Import the financial phrasebank dataset (all agree version) for training
note: the data are uploaded to a public github repository created by ourselves
ref: https://huggingface.co/datasets/financial_phrasebank
```python
import pandas as pd

# read data from url
raw_txt_url = r'https://raw.githubusercontent.com/starlikos/COMP7404_Group_Project/main/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'
data_df = pd.read_csv(raw_txt_url, sep='@', header=None, encoding="ISO-8859-1")

# change the column names
data_df.columns = ['text', 'label']

# change the sentiment label from text to integers to fit the configuration of FinBERT
# LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
label_to_int_dict = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
}
data_df['label'] = data_df['label'].apply(lambda my_label: label_to_int_dict.get(my_label))

print(f'There are {data_df.shape[0]} sample sentencens.')
data_df.head()
```

## use the datasets library to restructure the raw data for the training the model using the transformers library
```python
from datasets import load_dataset, Dataset, DatasetDict
print('Raw data df:')
print(data_df)

# convert the data_df to Dataset object and do the train/validation/test split 
# ref: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/2
full_dataset = Dataset.from_pandas(data_df)
train_testvalid = full_dataset.train_test_split(0.2) # Split the full dataset to 80% train + 20% test/validation
test_valid = train_testvalid['test'].train_test_split(0.5) # Split the 20% test/validation in half test, half validation
# gather everyone if you want to have a single DatasetDict
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'valid': test_valid['train'],
    'test': test_valid['test'],
    })

num_labels = len(set(dataset['train']['label']))

print('-'*20 + '\nTransformered Dataset object:\n')
print(dataset)
print(f'\nNumber of labels: {num_labels}')
```


## import required class from the transformers library and the pretrained FinBERT model
```python
#   - official training documentation from Hugging Face: https://huggingface.co/docs/transformers/training
#   - fine-tuning pretrained model with transformers: https://www.youtube.com/watch?v=V1-Hm2rNkik
from transformers import BertTokenizer, BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain', num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
```

## check if GPU is available and send the model to GPU
```python
# ref: https://github.com/huggingface/transformers/issues/2704
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Device used: {device}')

model = model.to(device)
```

## tokenize the dataset
```python
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=256, padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns('text') # remove the feature 'text' to avoid seeing warning message later on (as this is not used for model training); ref: https://huggingface.co/docs/datasets/process
print(tokenized_datasets)
```

## set the hyperparameters for training (using Hugging Face trainer API)
```python
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from torch import nn

# set the directory for exporting training logs
LOG_DIR = 'trainer_log'

## set the training hyperparameters
## ref: https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/trainer#transformers.TrainingArguments
## ref: https://zhuanlan.zhihu.com/p/363670628
training_args = TrainingArguments(
                  output_dir=LOG_DIR, 
                  overwrite_output_dir=True,
                  learning_rate=2e-5, # based on p.40 of the FinBERT paper
                  per_device_train_batch_size=32, # based on p.40 of the FinBERT paper
                  per_device_eval_batch_size=64,
                  num_train_epochs=4,
                  evaluation_strategy="steps",
                  logging_steps=5, # ref: https://discuss.huggingface.co/t/logs-of-training-and-validation-loss/1974
                  logging_strategy='steps',
                  report_to='tensorboard', # use tensorboard to track error; ref: https://www.tensorflow.org/tensorboard/get_started?hl=zh-tw
                  #warmup_steps=500, # no training loss will be computed for the first 500 steps by default
                  #weight_decay=0.01, # strength of weight decay
                  )

## define metric for measuring model performance
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

## set up the trainer for training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    compute_metrics=compute_metrics,
)
```

## Clear previous log and perform model training
```python
# clear any logs from previous runs
!rm -rf /content/trainer_log

# train the model
result = trainer.train()
```

## visualize training process using tensorboard
```python
# ref: https://www.tensorflow.org/tensorboard/get_started?hl=zh-tw
%tensorboard --logdir /content/trainer_log
```

## test the fine-tuned model on the out-of-the-sample test dataset
```python
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

y_true = tokenized_datasets['test']['label']

# use the model to predict label for given dataset via the trainer
# ref: https://huggingface.co/course/chapter3/3?fw=pt
predictions = trainer.predict(tokenized_datasets['test'])
y_pred = np.argmax(predictions.predictions, axis=-1)

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f'accuracy rate: {accuracy}')
print('confusion matrix:')
print(cm)
```

## prediction examples for demonstration
```python
int_to_label_dict = {v: k for k, v in label_to_int_dict.items()} # create a dictionary for mapping sentiment labels (integer) back to string values (e.g. positive)

# print the first n sentences in the test set with their predicted and actual sentiment labels
n = 10
print('Prediction examples of the test set using the fine-tuned FinBERT')
for i, sentence in enumerate(dataset['test']['text'][0:n]):
  print('-' * 20, sentence, sep='\n')
  print(f'Actual: {int_to_label_dict.get(dataset["test"]["label"][i])}; Predicted: {int_to_label_dict.get(y_pred[i])}')
 ```
 
## save the trained model
```python
trainer.save_model('finetuned_model')
```

## using the LM dictionary to classify the test dataset and calculate the accuracy
```python
# ref: https://github.com/hanzhichao2000/pysentiment
# note: LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
import pysentiment2 as ps
lm = ps.LM()

# itrerate through the test set and make the prediction
lm_predictions = [] # list for holding outputs

for sent in dataset['test']['text']:
  # generate the polarity score based on LM dictionary for the current sentence
  tokens = lm.tokenize(sent)
  score = lm.get_score(tokens)
  polarity = score['Polarity']

  # make prediction based on the polarity score and append to the output list
  if polarity < -0.3333:
    prediction = 2 # negative
  if polarity > 0.3333:
    prediction = 1 # positive
  else:
    prediction = 0 # neutral

  lm_predictions.append(prediction)

# calculate the test accuracy
lm_accuracy = accuracy_score(dataset['test']['label'], lm_predictions)
print(f'LM dictionary test accuracy: {lm_accuracy:2f}')
```

## Comparing performance of our fine-tuned FinBERT, FinBERT-tone and LM dictionary with a bar chart
```python
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Accuracy')
ax.set_title('Out-of-the-sample test set accuracy')

#x_values = ['LM dictionary', 'FinBERT-tone (Huang, 2020)', 'Our fine-tuned FinBERT']
#y_values = [lm_accuracy, finbert_tone_accuracy, accuracy]

x_values = ['LM dictionary', 'Our fine-tuned FinBERT']
y_values = [lm_accuracy, accuracy]

ax.bar(x_values, y_values)
plt.show()
```

## Link
https://github.com/dlauhku/hku_comp7404_finbert
