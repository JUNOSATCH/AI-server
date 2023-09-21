import os
import os.path as p
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, AutoTokenizer, AutoModel
import gluonnlp as nlp

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, tqdm_notebook

from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer


print("0")

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
print("1")
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=8,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

print("2")

bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
print("2")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
print("2")
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token="[PAD]")
print("2")

device = torch.device("cpu")
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
print("2")

model.load_state_dict(torch.load("./model.pt"))
print("2")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', return_dict=False)
tok = tokenizer.tokenize

print('3')

def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("평화가")
            elif np.argmax(logits) == 1:
                test_eval.append("CENSURE")
            elif np.argmax(logits) == 2:
                test_eval.append("HATE")
            elif np.argmax(logits) == 3:
                test_eval.append("DISCRIMINATION")
            elif np.argmax(logits) == 4:
                test_eval.append("SEXUAL")
            elif np.argmax(logits) == 5:
                test_eval.append("ABUSE")
            elif np.argmax(logits) == 6:
                test_eval.append("VIOLENCE")
            elif np.argmax(logits) == 7:
                test_eval.append("CRIME")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 이/가 느껴집니다.")

print("4")

end = 1
while end == 1 :
    sentence = input("input : ")
    if sentence == "0": break
    predict(sentence)
    print("\n")