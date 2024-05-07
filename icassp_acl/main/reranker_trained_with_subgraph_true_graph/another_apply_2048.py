import logging
import os
import sys
sys.path.append(os.getcwd())
# sys.path.append('/mnt/workspace/yeqin/huihuan/acl_code')
# os.chdir('/mnt/workspace/yeqin/huihuan/acl_code')
from util.reporting import Reporting
from reranker_trained_with_subgraph_true_graph.another_model_2048 import RerankerHypers, load_for_apply
import torch
import ujson as json
from util.line_corpus import write_open, jsonl_lines, read_open
import torch.nn.functional as F
from eval.kilt.eval_downstream import evaluate
from eval.convert_for_kilt_eval import to_distinct_doc_ids
import transformers
from torch import nn
import random
import copy
transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class RerankerApplyArgs(RerankerHypers):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.output = 'predictions/new_test/new_test_.jsonl'
        self.kilt_data = ''
        self.max_batch_size = 64
        self.exclude_instances = ''
        self.include_passages = False
        self.model_type = 'roberta'
        #self.model_name_or_path = '/mnt/workspace/yeqin/huihuan/re2g_acl/kgi-slot-filling/models/another_method_common'
        self.initial_retrieval ='data_dir/multidoc2dial_section/validation_merge_new.jsonl'
        # self.__required_args__ = ['model_type', 'model_name_or_path',
        #                           'output', 'initial_retrieval']


def one_instance(args, model, tokenizer, query, passages, subgraphs, subgraphs_id):
    # special_token_id = self.tokenizer.added_tokens_encoder['[special]']
    now_subgraphs_id = copy.deepcopy(subgraphs_id)
    for every_cluster in now_subgraphs_id:
        every_cluster.insert(0, '-1')
    query_ids = tokenizer([query], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:195]
    query = tokenizer.decode(query_ids)
    # concate
    texts_b = []
    texts_c = []
    new_subgraphs = []
    now_edge = []
    now_edge_type = []
    now_ptr = 0
    whole_relative_id = []
    for every_cluster in subgraphs:
        now_cluster = []
        max_common_words = ''
        for every_passage in every_cluster:
            every_passage = every_passage.replace('[SEP]', '\\')
            now_cluster.append(every_passage)
            if len(max_common_words):
                now_slide_one = ''
                now_slide_two = ''
                ptr = 1
                while now_slide_one == now_slide_two and len(now_slide_one) != len(max_common_words):
                    ptr += 1
                    now_slide_one = max_common_words[:ptr]
                    now_slide_two = every_passage[:ptr]
                max_common_words = max_common_words[:ptr-1] + 'x'
            else:
                max_common_words = every_passage + 'x'
        max_common_words = '<' + max_common_words[:-1].strip(' <3> ').strip(' <2> ')
        new_cluster = [max_common_words]
        new_cluster.extend(now_cluster)
        distinction = list(range(now_ptr, now_ptr+len(new_cluster)))
        # 双向聚合
        first_one_flag = 1
        for every_one_dist in distinction:
            if first_one_flag == 1:
                for every_one_other in distinction:
                    # 去除自环
                    if every_one_dist != every_one_other:
                        now_edge.append([every_one_dist, every_one_other])
                        # summarization 
                        now_edge_type.append(1)
                first_one_flag = 0
            else:
                for every_one_other in distinction:
                    # 去除自环
                    if every_one_dist != every_one_other:
                        now_edge.append([every_one_dist, every_one_other])
                        now_edge_type.append(0)
        whole_relative_id.append(distinction)
        #now_edge.append(distinction)
        now_ptr = now_ptr+len(new_cluster)
        new_subgraphs.append(new_cluster)
    subgraphs = new_subgraphs
    expand_to_passage = []
    for every_cluster in subgraphs:
        for every_passage in every_cluster:
            expand_to_passage.append(every_passage)
    texts_a = [query] * len(expand_to_passage)
        # MARK
    #texts_b = [p['text'] for p in passages]
    inputs = tokenizer(
        texts_a, expand_to_passage,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=512,
        padding='longest',
        truncation=True)

    # texts_a = [query] * len(texts_c)
    # inputs_subgraph = self.tokenizer(
    #     texts_a, texts_c,
    #     add_special_tokens=True,
    #     return_tensors='pt',
    #     max_length=self.args.max_seq_length,
    #     padding='longest',
    #     truncation=True)
    # # track how often we truncate to max_seq_length
    # if inputs['input_ids'].shape[1] == self.args.max_seq_length:
    #     self.max_length_count += 1
    inputs = {n: t.to(model.device) for n, t in inputs.items()}
    # inputs_subgraph = {n: t.to(model.device) for n, t in inputs_subgraph.items()}
    ptr_mem = 0
    subgraph_id_group = []
    passage_id_group = []
    for every_cluster_id in now_subgraphs_id:
        for every_internal_id in every_cluster_id:
            if every_internal_id == '-1':
                subgraph_id_group.append(ptr_mem)
            else:
                passage_id_group.append(ptr_mem)
            ptr_mem += 1
    passage_logits, subgraph_logits = model(inputs, torch.tensor(subgraph_id_group).cuda(), torch.tensor(passage_id_group).cuda(), torch.tensor(now_edge).transpose(0, 1).cuda(), torch.tensor(now_edge_type).cuda())
    passage_logprobs = F.softmax(passage_logits, dim=0)  # log_softmax over the passages
    subgraph_logprobs = F.softmax(subgraph_logits, dim=0)  # log_softmax over the passages
    # we want the logits rather than the logprobs as the teacher labels
    return passage_logprobs.tolist(), subgraph_logprobs.tolist()

def main():
    args = RerankerApplyArgs()
    args.fill_from_args()
    args.set_seed()
    assert args.world_size == 1 and args.n_gpu == 1  # TODO: support distributed

    # load model and tokenizer
    model, tokenizer = load_for_apply(args)
    #tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]", "[unused3]", "[unused4]"]})
    if args.exclude_instances:
        with read_open(args.exclude_instances) as f:
            exclude_instances = set(json.load(f))
    else:
        exclude_instances = None
    model.eval()
    report = Reporting()
    with torch.no_grad(), write_open(args.output) as output:
        for line in jsonl_lines(args.initial_retrieval):
            jobj = json.loads(line)
            inst_id = jobj['id']
            if exclude_instances and inst_id in exclude_instances:
                continue
            query = jobj['input']
            passages = jobj['passages']
            subgraphs = jobj['subgraphs']
            subgraphs_id = jobj['subgraphs_id']
            probs, subgraph_logprobs = one_instance(args, model, tokenizer, query, passages, subgraphs, subgraphs_id)
            passages = []
            ptr = -1
            for every_cluster_index in range(len(subgraphs)):
                every_cluster = subgraphs[every_cluster_index]
                for every_passage_index in range(len(every_cluster)):
                    ptr += 1
                    probs[ptr] = probs[ptr] #* subgraph_logprobs[every_cluster_index]
                    passages.append({'pid': subgraphs_id[every_cluster_index][every_passage_index],'text': every_cluster[every_passage_index]})
            scored_pids = [(p['pid'], prob) for p, prob in zip(passages, probs)]
            scored_pids.sort(key=lambda x: x[1], reverse=True)
            wids = to_distinct_doc_ids([pid for pid, prob in scored_pids])  # convert to Wikipedia document ids
            pred_record = {'id': inst_id, 'input': query, 'scored_pids': scored_pids,
                           'output': [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]}
            if args.include_passages:
                pred_record['passages'] = passages
            output.write(json.dumps(pred_record) + '\n')
            if report.is_time():
                print(f'Finished {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.')
    if args.kilt_data:
        evaluate(args.kilt_data, args.output)


if __name__ == "__main__":
    main()
