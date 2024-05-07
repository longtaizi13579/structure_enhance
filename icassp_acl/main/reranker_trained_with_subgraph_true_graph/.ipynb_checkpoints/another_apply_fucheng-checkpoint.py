import logging
import os
import sys
sys.path.append(os.getcwd())
# sys.path.append('/mnt/workspace/yeqin/huihuan/acl_code')
# os.chdir('/mnt/workspace/yeqin/huihuan/acl_code')
from util.reporting import Reporting
from reranker_trained_with_subgraph.another_model import RerankerHypers, load_for_apply
import torch
import ujson as json
from util.line_corpus import write_open, jsonl_lines, read_open
import torch.nn.functional as F
from eval.kilt.eval_downstream import evaluate
from eval.convert_for_kilt_eval import to_distinct_doc_ids
import transformers
from torch import nn
import random
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
        self.model_type = 'bert'
        #self.model_name_or_path = '/mnt/workspace/yeqin/huihuan/re2g_acl/kgi-slot-filling/models/another_method_common'
        self.initial_retrieval ='predictions/dprKGI0/test_merge_new.jsonl'
        # self.__required_args__ = ['model_type', 'model_name_or_path',
        #                           'output', 'initial_retrieval']


def one_instance(args, model, tokenizer, query, passages, subgraphs, subgraphs_id):
    query_ids = self.tokenizer([query], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:195]
    query = self.tokenizer.decode(query_ids)

    # concate
    texts_b = []
    texts_c = []
    new_subgraphs = []
    for every_cluster in subgraphs:
        now_cluster = []
        for every_passage in every_cluster:
            every_passage = every_passage.replace('[SEP]', '[unused1]')
            now_cluster.append(every_passage)
        new_subgraphs.append(now_cluster)
    subgraphs = new_subgraphs
    for every_cluster_index in range(len(subgraphs)):
        now_cluster = subgraphs[every_cluster_index]
        now_sentence = ''
        for every_passage_index in range(len(now_cluster)):
            core_passage = now_cluster[every_passage_index]
            other_passages = []
            # order = list(range(len(now_cluster)))
            # random.shuffle(order)
            #random select
            # random_num = random.randint(1, len(order))
            # order = order[:random_num]
            # shuffle 
            for every_passage_internal_index in range(len(now_cluster)):
                if  every_passage_internal_index != every_passage_index:
                    other_passages.append(now_cluster[every_passage_internal_index])
            other_passages = [other_passage.split('[unused1]')[0] for other_passage in other_passages]
            others = '[unused2]'.join(other_passages)
            texts_b.append('[unused3]' +core_passage + '[unused4]' + others)
        texts_c.append('[unused1]'.join(now_cluster))
    texts_a = [query] * len(texts_b)
    # MARK
    #texts_b = [p['text'] for p in passages]
    inputs = self.tokenizer(
        texts_a, texts_b,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=self.args.max_seq_length,
        padding='longest',
        truncation=True)

    texts_a = [query] * len(texts_c)
    inputs_subgraph = self.tokenizer(
        texts_a, texts_c,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=self.args.max_seq_length,
        padding='longest',
        truncation=True)
    # track how often we truncate to max_seq_length
    if inputs['input_ids'].shape[1] == self.args.max_seq_length:
        self.max_length_count += 1
    inputs = {n: t.to(model.device) for n, t in inputs.items()}
    inputs_subgraph = {n: t.to(model.device) for n, t in inputs_subgraph.items()}

    passage_logits, subgraph_logits = model(inputs, inputs_subgraph)
    passage_logprobs = F.softmax(passage_logits, dim=0)  # log_softmax over the passages
    subgraph_logprobs = F.softmax(subgraph_logits, dim=0)  # log_softmax over the passages
    # we want the logits rather than the logprobs as the teacher labels
    return passage_logprobs.numpy().tolist(), subgraph_logprobs.numpy().tolist()


def main():
    args = RerankerApplyArgs()
    args.fill_from_args()
    args.set_seed()
    assert args.world_size == 1 and args.n_gpu == 1  # TODO: support distributed

    # load model and tokenizer
    model, tokenizer = load_for_apply(args)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]", "[unused3]", "[unused4]"]})
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
                    probs[ptr] = probs[ptr] * subgraph_logprobs[every_cluster_index]
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
