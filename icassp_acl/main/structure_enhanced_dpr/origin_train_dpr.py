import random
from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder
from my_model import BiEncoder
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from datasets import Dataset
import json
import re
import os
import copy
import logging

logging.basicConfig(filename='pretrain_log',level=logging.DEBUG)
ctx_model = '../my_implement_offline_for_haomin/256_5e-5_25_graph_data_graph_method_normal/18_0.6317/ctx' #'/root/data/huihuan/model/ctx' #'facebook/dpr-ctx_encoder-multiset-base'
qry_model = '../my_implement_offline_for_haomin/256_5e-5_25_graph_data_graph_method_normal/18_0.6317/qry' #'/root/data/huihuan/model/qry' #'facebook/dpr-question_encoder-multiset-base'
os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') 
#os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement')
def load_data(file_addr):
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(qry_model)
    domains = ['train', 'validation']
    all_files = os.listdir(file_addr)
    concrete_dataset = []
    concrete_addr = []
    for every_split_domain in domains:
        for every_file in all_files:
            if every_split_domain in every_file:
                concrete_addr.append(file_addr + '/' + every_file)
                break
    for every_file_addr in concrete_addr:
        file_in = open(every_file_addr, 'r')
        dialogs = json.load(file_in)
        all_query = []
        all_negatives = []
        all_answer = []
        all_graph_content = []
        all_edge = []
        all_edge_type = []
        for every_dialog in dialogs:
            if len(every_dialog['hard_negative_ctxs']):
                now_negative = random.choice(every_dialog['hard_negative_ctxs'][:15])
            else:
                now_negative = random.choice(every_dialog['negative_ctxs'][:15])
            all_negatives.append(now_negative['text'].strip())
            all_answer.append(every_dialog['positive_ctxs'][0])
            all_query.append(every_dialog['question'])
        build_dataset = {
            'query': all_query,
            'negative': all_negatives,
            'answer': all_answer,
        }
        concrete_dataset.append(build_dataset)
    return concrete_dataset
file_addr = '../../../../preprocess/result_data/graph_data'
data = load_data(file_addr)
build_train_dataset = data[0]
build_dev_dataset = data[1]
train_dataset = Dataset.from_dict(build_train_dataset)
dev_dataset = Dataset.from_dict(build_dev_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=128, shuffle=True)
dev_batch_size = 128
model = BiEncoder(16)
model.cuda()
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 20
num_training_steps = num_epochs * len(build_train_dataset['query'])//dev_batch_size
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0.06 * num_training_steps,
    num_training_steps=num_training_steps
)
save_base_directory = './256_2e-5_25_graph_data_pretrained_by_add_connection_graph_method_new/'
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model)
qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(qry_model)
for epoch in range(num_epochs):
    whole_loss = 0
    whole_num = 0
    model.train()
    for index, sample in enumerate(train_dataloader):
        positive = torch.tensor(list(range(len(sample['answer'])))).cuda()
        sample['answer'].extend(sample['negative'])
        ctx = ctx_tokenizer(sample['answer'], padding=True, truncation=True, max_length=256, return_tensors='pt') # max 256
        qry = qry_tokenizer(sample['query'], padding=True, truncation=True, max_length=256, return_tensors='pt')
        ctx = {k: v.cuda() for k, v in ctx.items()}
        qry = {k: v.cuda() for k, v in qry.items()}
        batch = {'ctx': ctx, 'qry': qry, 'positive': positive}
        loss, accuracy = model(batch)
        loss.backward()
        whole_num += 1
        whole_loss += loss
        if (index-1) % 10 == 0:
            logging.info('{0} loss: {1}'.format(whole_num, whole_loss/whole_num))
            logging.info('lr: {0}'.format(lr_scheduler.get_last_lr()))
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    model.eval()
    all_prediction = []
    all_label = []
    correct_num = 0
    whole_num = 0
    whole_loss = 0
    for index, sample in enumerate(dev_dataloader):
        with torch.no_grad():
            positive = torch.tensor(list(range(len(sample['answer'])))).cuda()
            sample['answer'].extend(sample['negative'])
            ctx = ctx_tokenizer(sample['answer'], padding=True,truncation=True, max_length=256, return_tensors='pt')#128
            qry = qry_tokenizer(sample['query'], padding=True,truncation=True, max_length=256,  return_tensors='pt')#64
            ctx = {k: v.cuda() for k, v in ctx.items()}
            qry = {k: v.cuda() for k, v in qry.items()}
            batch = {'ctx': ctx, 'qry': qry, 'positive': positive}
            loss, accuracy = model(batch)
            whole_loss += loss
            whole_num += dev_batch_size
            correct_num += dev_batch_size * accuracy
    whole_accuracy = float(correct_num) / whole_num
    logging.info('eval loss: {0}'.format(whole_loss))
    logging.info('eval accuracy: {0}'.format(whole_accuracy))
    pt_save_directory = save_base_directory + str(epoch) + f'_{str(round(whole_accuracy,4))}'
    qry_tokenizer.save_pretrained(pt_save_directory + '/qry')
    ctx_tokenizer.save_pretrained(pt_save_directory + '/ctx')
    model.save_pretrained(pt_save_directory)
    data = load_data(file_addr)
    build_train_dataset = data[0]
    build_dev_dataset = data[1]
    train_dataset = Dataset.from_dict(build_train_dataset)
    dev_dataset = Dataset.from_dict(build_dev_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=128, shuffle=True)