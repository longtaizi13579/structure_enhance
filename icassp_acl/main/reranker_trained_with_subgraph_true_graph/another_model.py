import logging
from torch_util.hypers_base import HypersBase
import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    PreTrainedTokenizer,
    BertForSequenceClassification,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)
from torch.utils.checkpoint import checkpoint
import copy
#from torch_geometric.nn import RGATConv, RGCNConv
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


class RerankerHypers(HypersBase):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.initial_retrieval = ''
        self.max_num_seq_pairs_per_device = 32
        self.full_train_batch_size = 32
        self.gradient_accumulation_steps = self.full_train_batch_size
        self.per_gpu_train_batch_size = 1
        self.num_train_epochs = 2
        self.train_instances = -1
        self.learning_rate = 3e-6
        self.max_seq_length = 512
        self.num_labels = 2
        self.fold = ''  # IofN
        self.add_all_positives = False
        self.doc_match_weight = 0.0
        self.model_type = 'roberta'
        self.model_name_or_path = 'roberta-large' #'models/1_27_graph_rgat_1e_5_best' #1_27_graph_rgat_1e_5_best'
        self.origin_model_name_or_path = 'roberta-large'
        self.do_lower_case = True
    def _post_init(self):
        super()._post_init()
        self.gradient_accumulation_steps = self.full_train_batch_size
        self.per_gpu_train_batch_size = 1
        # assert self.world_size == 1


def load_tokenizer(hypers: HypersBase, tokenizer_class):
    tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
        hypers.tokenizer_name if hypers.tokenizer_name else hypers.model_name_or_path,
        do_lower_case=hypers.do_lower_case,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None
    )
    return tokenizer


def load_pretrained(hypers: HypersBase, config_class, model_class, tokenizer_class, **extra_model_args):
    # Load pretrained model and tokenizer
    # if hypers.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = config_class.from_pretrained(
        hypers.config_name if hypers.config_name else hypers.model_name_or_path,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
        **extra_model_args
    )
    tokenizer = load_tokenizer(hypers, tokenizer_class)
    model = model_class.from_pretrained(
        hypers.model_name_or_path,
        from_tf=bool(".ckpt" in hypers.model_name_or_path),
        config=config,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
    )
    special_tokens = ["[others]"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    # if hypers.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(hypers.device)
    return model, tokenizer

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        whole_output = self.encoder(input_ids, attention_mask).pooler_output
        return whole_output


class Two_level_Model(nn.Module):
    def __init__(self, model):
        super(Two_level_Model, self).__init__()
        #self.dropout = model.dropout
        self.passage_classifier = nn.Linear(1024, 2)
        self.graph_classifier = nn.Linear(1024, 2)
        self.bert = EncoderWrapper(model)
        #self.subgraph_classifier = copy.deepcopy(model.classifier)
        self.device = 'cuda:0'
        self.encoder_gpu_train_limit = 10
        #self.conv = RGATConv(1024, 1024, 2)
        self.conv = RGCNConv(1024, 1024, 2)


    def encode(self, model, passages):
        input_ids = passages['input_ids']
        attention_mask = passages['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)

    
    def forward(self, passages, subgraph_id_group, passage_id_group, now_edge, now_edge_type):
        get_passage_length = len(passages['input_ids'])
        whole_outputs = self.encode(self.bert, passages)
        graph_enhance = self.conv(whole_outputs, now_edge, now_edge_type)
        passage_enhance = graph_enhance[passage_id_group]
        subgraph_enhance = graph_enhance[subgraph_id_group]
        passage_logits = self.passage_classifier(passage_enhance)
        subgraph_logits = self.graph_classifier(subgraph_enhance)
        return F.log_softmax(passage_logits, dim=-1)[:, 1].detach().cpu(), F.log_softmax(subgraph_logits, dim=-1)[:, 1].detach().cpu()

def load_pretrained_for_apply(hypers: HypersBase, config_class, model_class, tokenizer_class, **extra_model_args):
    # Load pretrained model and tokenizer
    # if hypers.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = config_class.from_pretrained(
        hypers.config_name if hypers.config_name else hypers.origin_model_name_or_path,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
        **extra_model_args
    )
    tokenizer = load_tokenizer(hypers, tokenizer_class)
    model = model_class.from_pretrained(
        hypers.origin_model_name_or_path,
        from_tf=bool(".ckpt" in hypers.origin_model_name_or_path),
        config=config,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
    )
    # if hypers.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    special_tokens = ["[others]"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model = Two_level_Model(model)
    my_checkpoint = torch.load(os.path.join(hypers.model_name_or_path, "model.pth.tar"), map_location='cpu')
    model.load_state_dict({k.replace('module.',''):v for k,v in my_checkpoint.items()})
    model.to(hypers.device)
    return model, tokenizer

def load_for_apply(hypers: RerankerHypers):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[hypers.model_type.lower()]
    resume_from = hypers.resume_from if hasattr(hypers, 'resume_from') else ''
    if not resume_from:
        model, tokenizer = load_pretrained_for_apply(hypers, config_class, model_class, tokenizer_class,
                                                    num_labels=hypers.num_labels)
    else:
        tokenizer = load_tokenizer(hypers, tokenizer_class)
        model = model_class.from_pretrained(resume_from)
        special_tokens = ["[others]"]
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model = Two_level_Model(model)
        my_checkpoint = torch.load(os.path.join(hypers.model_name_or_path, "model.pth.tar"), map_location='cpu')
        model.load_state_dict({k.replace('module.',''):v for k,v in my_checkpoint.items()})
        model.to(hypers.device)

    return model, tokenizer
    
def load(hypers: RerankerHypers):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[hypers.model_type.lower()]
    resume_from = hypers.resume_from if hasattr(hypers, 'resume_from') else ''
    if not resume_from:
        model, tokenizer = load_pretrained(hypers, config_class, model_class, tokenizer_class,
                                           num_labels=hypers.num_labels)
    else:
        tokenizer = load_tokenizer(hypers, tokenizer_class)
        model = model_class.from_pretrained(resume_from)
        special_tokens = ["[others]"]
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model.to(hypers.device)

    return model, tokenizer


def save_transformer(hypers: HypersBase, model, tokenizer, *, save_dir=None):
    if hypers.global_rank == 0:
        if save_dir is None:
            save_dir = hypers.output_dir
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = (
        #     model.module if hasattr(model, "module") else model
        # )  # Take care of distributed/parallel training
        torch.save(hypers, os.path.join(save_dir, "training_args.bin"))
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth.tar"))
        # model_to_save.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

