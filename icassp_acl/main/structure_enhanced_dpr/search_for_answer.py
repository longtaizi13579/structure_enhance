import torch
import heapq
from tqdm import tqdm
import json
import os
os.chdir('/mnt/workspace/yeqin/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') #os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement')
sections = torch.load('./result_pt/passage_candidates_test.pt')
querys = torch.load('./result_pt/query_candidates_test.pt')
dot_products = torch.matmul(querys, sections.transpose(0, 1))
query_file_in = open('../../../../preprocess/result_data/graph_data/dpr_multidoc2dial_validation.json', 'r')
query_data = json.load(query_file_in)
all_querys = []
positive_data = []
for every_dict in query_data:
    all_querys.append(every_dict['question'])
    positive_data.append(every_dict['positive_ctxs'][0])
all_passages = []
passages_file_in = open('../../../../data/all_passage.txt', 'r')
for every_line in passages_file_in:
    all_passages.append(every_line.strip('\n'))
whole_num = 0
correct_1 = 0
correct_5 = 0
correct_10 = 0
correct_100 = 0
mrr_value_5 = 0
error_num = 0
for every_positive_index in tqdm(range(len(positive_data))):
    now_positive = positive_data[every_positive_index]
    if now_positive in all_passages:
        line_result = dot_products[every_positive_index]
        topk_index = heapq.nlargest(100, range(len(line_result)), line_result.__getitem__)
        get_index = all_passages.index(now_positive)
        if get_index == topk_index[0]:
            correct_1 += 1
        if get_index in topk_index[:5]:
            correct_5 += 1
            order = topk_index.index(get_index)
            mrr_value_5 += 1/(order+1)
        if get_index in topk_index[:10]:
            correct_10 += 1
        if get_index in topk_index:
            correct_100 += 1
        whole_num += 1
    else:
        error_num += 1
print('top 1: ', correct_1/whole_num)
print('top 5: ', correct_5/whole_num)
print('top 10: ', correct_10/whole_num)
print('top 100: ', correct_100/whole_num)
print('mrr@5: ', mrr_value_5/whole_num)
print(error_num)