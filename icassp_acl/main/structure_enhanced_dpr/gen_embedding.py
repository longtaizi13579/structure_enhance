from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder
import torch
from tqdm import tqdm
from origin_my_graph_method import myRGCNConv
import json
import os
import numpy as np
#os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') #os.chdir('/mnt/workspace/yeqin/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') #os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement')
file_addr = 'neighbor_no_zeros_without_bert_transformation'
epoch_name = '12_0.6358'
qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(f'./{file_addr}/{epoch_name}/qry')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(f'./{file_addr}/{epoch_name}/ctx')
qry_model = DPRQuestionEncoder.from_pretrained(f'./{file_addr}/{epoch_name}/qry')
ctx_model = DPRContextEncoder.from_pretrained(f'./{file_addr}/{epoch_name}/ctx')
conv = myRGCNConv(768, 768, 2, root_trans=False)
ckpt1 = torch.load(f'./{file_addr}/{epoch_name}/conv')
conv.load_state_dict(ckpt1['conv'])
qry_model.cuda()
ctx_model.cuda()
conv.cuda()
file_in = open('../../../../data/graph_method.json')
document_data_dict = json.load(file_in)
passages_file_in = open('../../../../data/all_passage.txt', 'r')
query_file_in = open('../../../../preprocess/result_data/graph_data/dpr_multidoc2dial_validation.json', 'r')
query_data = json.load(query_file_in)
all_querys = []
for every_dict in query_data:
    all_querys.append(every_dict['question'])
all_passages = []
for every_line in passages_file_in:
    all_passages.append(every_line.strip('\n'))
ctx_model.eval()
qry_model.eval()
conv.eval()
all_passages_candidates = []
passage_representation = {}
with torch.no_grad():
    for document_name in document_data_dict:
        graph_nodes = document_data_dict[document_name]
        all_node_text = []
        all_node_representation = []
        for every_node in graph_nodes['node']:
            all_node_text.append(every_node['text'])
            tokenizer_result = ctx_tokenizer(every_node['text'], truncation=True, max_length=512, return_tensors='pt')
            model_vector = ctx_model(tokenizer_result['input_ids'].cuda()).pooler_output
            all_node_representation.append(model_vector.view(1,-1))
        graph_vector = torch.cat(all_node_representation, dim=0).cuda()
        now_graph_edge = torch.tensor(np.array([np.array(graph_nodes['edge'])[:, 0], np.array(graph_nodes['edge'])[:, 1]])).cuda()
        now_graph_edge_type = torch.tensor(graph_nodes['edge_type']).cuda()
        #edge_weight = torch.zeros(len(now_graph_edge[0])).cuda()
        one_layer_graph_vector = conv(graph_vector, now_graph_edge, now_graph_edge_type)
        two_layer_graph_vector = conv(one_layer_graph_vector, now_graph_edge, now_graph_edge_type)
        for every_index in range(len(graph_vector)):
            now_passage = graph_nodes['node'][every_index]['text']
            passage_representation[now_passage] = two_layer_graph_vector[every_index] #+ graph_vector[every_index]
    for every_passage in all_passages:
        all_passages_candidates.append(passage_representation[every_passage].view(1,-1).cpu())
    passage_candidates = torch.cat(all_passages_candidates, 0)
    torch.save(passage_candidates, './result_pt/passage_candidates_test.pt')
    query_candidates = []
    for every_query in tqdm(all_querys):
        tokenizer_result = qry_tokenizer(every_query, truncation=True, max_length=512, return_tensors='pt')
        model_vector = qry_model(tokenizer_result['input_ids'].cuda()).pooler_output
        query_candidates.append(model_vector.cpu())
    query_candidates = torch.cat(query_candidates, 0)
    torch.save(query_candidates, './result_pt/query_candidates_test.pt')
