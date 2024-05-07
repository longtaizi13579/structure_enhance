import random
from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder
from my_model import myGraphEncoder
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
import numpy as np
import logging

def new_collate_fn(batch):
    output_batch = {}
    for every_key in batch[0]:
        output_batch[every_key] = []
    for every_instance in batch:
        for every_key in output_batch:
            output_batch[every_key].append(every_instance[every_key])
    return output_batch

logging.basicConfig(filename='neighbor_without_bert_transformation',level=logging.DEBUG)
ctx_model = '/root/data/huihuan/model/ctx' #'/root/data/huihuan/model/ctx' #'facebook/dpr-ctx_encoder-multiset-base'
qry_model = '/root/data/huihuan/model/qry' #'/root/data/huihuan/model/qry' #'facebook/dpr-question_encoder-multiset-base'
#os.chdir('/mnt/workspace/yeqin/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') 
save_base_directory = './neighbor_no_zeros_without_bert_transformation/'
os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement')
file_addr = '../../../../preprocess/result_data/graph_data'
def load_data(file_addr):
    file_in = open('../../../../data/graph_method.json')
    document_data_dict = json.load(file_in)
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
        all_ground_id = []
        all_negative_id = []
        for every_dialog in dialogs:
            if len(every_dialog['hard_negative_ctxs']):
                now_negative = random.choice(every_dialog['hard_negative_ctxs'][:15])
            else:
                now_negative = random.choice(every_dialog['negative_ctxs'][:15])
            all_negatives.append(now_negative['text'].strip())
            all_answer.append(every_dialog['positive_ctxs'][0])
            all_query.append(every_dialog['question'])
            positive_list = re.split('<2>|<3>', every_dialog['positive_ctxs'][0])
            document_name = positive_list[0]
            graph_nodes = document_data_dict[document_name]
            new_graph = copy.deepcopy(graph_nodes)
            all_node_text = []
            ground_truth_node_id = -1
            for every_node in graph_nodes['node']:
                all_node_text.append(every_node['text'])
                if every_node['text'] == every_dialog['positive_ctxs'][0]:
                    ground_truth_node_id = every_node['id']
                    all_ground_id.append(ground_truth_node_id)
                if  every_node['text'] == now_negative['text']:
                    all_negative_id.append(every_node['id'])
            if len(every_dialog['hard_negative_ctxs']) == 0:
                candidate_node_ids = list(range(len(graph_nodes['node'])))
                candidate_node_ids.remove(eval(ground_truth_node_id))
                all_negative_id.append(str(random.choice(candidate_node_ids)))
            new_graph['node'] = all_node_text
            all_graph_content.append(tuple(new_graph['node']))
            all_edge.append(tuple(new_graph['edge']))
            all_edge_type.append(tuple(new_graph['edge_type']))
            if len(all_ground_id) != len(all_negative_id):
                print()
        build_dataset = {
            'query': all_query,
            'negative': all_negatives,
            'answer': all_answer,
            'graph_content': all_graph_content,
            'edge': all_edge,
            'edge_type': all_edge_type,
            'grounding_id': all_ground_id,
            'negative_id': all_negative_id
        }
        concrete_dataset.append(build_dataset)
    return concrete_dataset
data = load_data(file_addr)
build_train_dataset = data[0]
build_dev_dataset = data[1]
train_dataset = Dataset.from_dict(build_train_dataset)
dev_dataset = Dataset.from_dict(build_dev_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=new_collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=128, shuffle=True, collate_fn=new_collate_fn)
dev_batch_size = 128
model = myGraphEncoder(16)
model.cuda()
#optimizer = AdamW(model.parameters(), lr=4e-5)
optimizer = AdamW(model.parameters(), lr=3e-5)
num_epochs = 15
num_training_steps = num_epochs * len(build_train_dataset['query'])//dev_batch_size
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps,
    num_training_steps=num_training_steps
)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_training_steps=num_training_steps, num_warmup_steps=0
# )
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model)
qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(qry_model)
for epoch in range(num_epochs):
    whole_loss = 0
    whole_num = 0
    model.train()
    for index, sample in enumerate(train_dataloader):
        positive = torch.tensor(list(range(len(sample['answer'])))).cuda()
        sample['answer'].extend(sample['negative'])
        whole_graph = []
        for every_graph in sample['graph_content']:
            one_graph = ctx_tokenizer(every_graph, padding=True, truncation=True, max_length=256, return_tensors='pt')
            one_graph = {k: v.cuda() for k, v in one_graph.items()}
            whole_graph.append(one_graph)
        ctx = ctx_tokenizer(sample['answer'], padding=True, truncation=True, max_length=256, return_tensors='pt') # max 256
        qry = qry_tokenizer(sample['query'], padding=True, truncation=True, max_length=256, return_tensors='pt')
        ctx = {k: v.cuda() for k, v in ctx.items()}
        qry = {k: v.cuda() for k, v in qry.items()}
        edge_list = [torch.tensor(np.array([np.array(k)[:, 0], np.array(k)[:, 1]])).cuda() for k in sample['edge']] 
        edge_type_list = [torch.tensor(k).cuda() for k in sample['edge_type']] 
        ground_id = [torch.tensor(eval(k)).cuda() for k in sample['grounding_id']] 
        negative_id = [torch.tensor(eval(k)).cuda() for k in sample['negative_id']] 
        batch = {'ctx': ctx, 'qry': qry, 'positive': positive, 'graph': whole_graph, 'edge': edge_list, 'edge_type': edge_type_list, 'grounding_id': ground_id, 'negative_id': negative_id}
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
            whole_graph = []
            for every_graph in sample['graph_content']:
                one_graph = ctx_tokenizer(every_graph, padding=True, truncation=True, max_length=256, return_tensors='pt')
                one_graph = {k: v.cuda() for k, v in one_graph.items()}
                whole_graph.append(one_graph)
            ctx = ctx_tokenizer(sample['answer'], padding=True, truncation=True, max_length=256, return_tensors='pt') # max 256
            qry = qry_tokenizer(sample['query'], padding=True, truncation=True, max_length=256, return_tensors='pt')
            ctx = {k: v.cuda() for k, v in ctx.items()}
            qry = {k: v.cuda() for k, v in qry.items()}
            edge_list = [torch.tensor(np.array([np.array(k)[:, 0], np.array(k)[:, 1]])).cuda() for k in sample['edge']] 
            edge_type_list = [torch.tensor(k).cuda() for k in sample['edge_type']] 
            ground_id = [torch.tensor(eval(k)).cuda() for k in sample['grounding_id']] 
            negative_id = [torch.tensor(eval(k)).cuda() for k in sample['negative_id']] 
            batch = {'ctx': ctx, 'qry': qry, 'positive': positive, 'graph': whole_graph, 'edge': edge_list, 'edge_type': edge_type_list, 'grounding_id': ground_id, 'negative_id': negative_id}
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
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=new_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=128, shuffle=True, collate_fn=new_collate_fn)