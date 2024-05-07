from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import RGCNConv
import copy
import os
from my_graph_method import myRGCNConv

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        pooler_output = self.encoder(input_ids, attention_mask).pooler_output
        return pooler_output

    def save_pretrained(self, address):
        self.encoder.save_pretrained(address)


class myGraphEncoder(torch.nn.Module):
    def __init__(self, encoder_gpu_train_limit):
        super().__init__()
        ctx_model = '/root/data/huihuan/model/ctx'#'/root/data/huihuan/model/ctx' #'facebook/dpr-ctx_encoder-multiset-base'
        qry_model = '/root/data/huihuan/model/qry' #'/root/data/huihuan/model/qry' #'facebook/dpr-question_encoder-multiset-base'
        os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') 
        self.ctx_model = EncoderWrapper(DPRContextEncoder.from_pretrained(ctx_model))
        self.qry_model = DPRQuestionEncoder.from_pretrained(qry_model)
        self.qry_model_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(qry_model)
        '''
        special_tokens = ["<user>", "<agent>"]
        self.qry_model_tokenizer.add_tokens(special_tokens)
        self.qry_model.resize_token_embeddings(len(self.qry_model_tokenizer))'''
        self.qry_model = EncoderWrapper(self.qry_model)
        self.encoder_gpu_train_limit = encoder_gpu_train_limit
        self.conv = myRGCNConv(768, 768, 2, root_trans=False, bias=False, edge_tran=False)
        #self.ckpt1 = torch.load('./256_4e-5_25_graph_data_graph_method_normal_add_connection/4_0.6233/conv')
        #self.conv.load_state_dict(self.ckpt1['conv'])
        #self.conv2 = RGCNConv(768, 768, 2)

    def encode(self, model, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)

    def forward(self, data):
        #print('?????')
        positive_vector_list = []
        negative_vector_list = []
        qry_vector = self.encode(self.qry_model, data['qry'])
        for every_graph_index in range(len(data['graph'])):
            now_graph = data['graph'][every_graph_index]
            now_positive_index = data['grounding_id'][every_graph_index]
            now_negative_index = data['negative_id'][every_graph_index]
            now_graph_edge = data['edge'][every_graph_index]
            now_graph_edge_type = data['edge_type'][every_graph_index]
            graph_vector = self.encode(self.ctx_model, now_graph)
            #edge_weight = torch.ones(len(now_graph_edge[0])).cuda()
            #one_layer_graph_vector = self.conv(graph_vector, now_graph_edge, now_graph_edge_type, edge_weight)
            #two_layer_graph_vector = self.conv(one_layer_graph_vector, now_graph_edge, now_graph_edge_type, edge_weight)
            one_layer_graph_vector = self.conv(graph_vector, now_graph_edge, now_graph_edge_type)
            two_layer_graph_vector = self.conv(one_layer_graph_vector, now_graph_edge, now_graph_edge_type)
            new_positive_vector = two_layer_graph_vector[now_positive_index]
            new_negative_vector = two_layer_graph_vector[now_negative_index]
            positive_vector_list.append(new_positive_vector.view(1,-1))
            negative_vector_list.append(new_negative_vector.view(1,-1))
        #neg_vector = self.encode(self.ctx_model, data['negative']['neg'])
        ctx_vector = torch.cat([torch.cat(positive_vector_list, dim=0), torch.cat(negative_vector_list, dim=0)], dim=0)
        dot_products = torch.matmul(qry_vector, ctx_vector.transpose(0, 1))
        probs = F.log_softmax(dot_products, dim=1)
        loss = F.nll_loss(probs, data['positive'].long())
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == data['positive']).sum() / (data['positive'].shape[0] - (data['positive'].long() == torch.ones(len(data['positive'])).long().cuda()*(-100)).sum())
        return loss, accuracy

    def save_pretrained(self, addr):
        self.ctx_model.save_pretrained(addr + '/ctx')
        self.qry_model.save_pretrained(addr + '/qry')
        torch.save({'conv':self.conv.state_dict()}, addr + '/conv')

class GraphEncoder(torch.nn.Module):
    def __init__(self, encoder_gpu_train_limit):
        super().__init__()
        ctx_model = '/root/data/huihuan/model/ctx'#'/root/data/huihuan/model/ctx' #'facebook/dpr-ctx_encoder-multiset-base'
        qry_model = '/root/data/huihuan/model/qry' #'/root/data/huihuan/model/qry' #'facebook/dpr-question_encoder-multiset-base'
        os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') 
        self.ctx_model = EncoderWrapper(DPRContextEncoder.from_pretrained(ctx_model))
        self.qry_model = DPRQuestionEncoder.from_pretrained(qry_model)
        self.qry_model_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(qry_model)
        '''
        special_tokens = ["<user>", "<agent>"]
        self.qry_model_tokenizer.add_tokens(special_tokens)
        self.qry_model.resize_token_embeddings(len(self.qry_model_tokenizer))'''
        self.qry_model = EncoderWrapper(self.qry_model)
        self.encoder_gpu_train_limit = encoder_gpu_train_limit
        self.conv = RGCNConv(768, 768, 2)
        #self.ckpt1 = torch.load('./256_4e-5_25_graph_data_graph_method_normal_add_connection/4_0.6233/conv')
        #self.conv.load_state_dict(self.ckpt1['conv'])
        #self.conv2 = RGCNConv(768, 768, 2)

    def encode(self, model, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)

    def forward(self, data):
        positive_vector_list = []
        negative_vector_list = []
        qry_vector = self.encode(self.qry_model, data['qry'])
        for every_graph_index in range(len(data['graph'])):
            now_graph = data['graph'][every_graph_index]
            now_positive_index = data['grounding_id'][every_graph_index]
            now_negative_index = data['negative_id'][every_graph_index]
            now_graph_edge = data['edge'][every_graph_index]
            now_graph_edge_type = data['edge_type'][every_graph_index]
            graph_vector = self.encode(self.ctx_model, now_graph)
            one_layer_graph_vector = self.conv(graph_vector, now_graph_edge, now_graph_edge_type)
            two_layer_graph_vector = self.conv(one_layer_graph_vector, now_graph_edge, now_graph_edge_type)
            new_positive_vector = two_layer_graph_vector[now_positive_index]
            new_negative_vector = two_layer_graph_vector[now_negative_index]
            #new_positive_vector = two_layer_graph_vector[now_positive_index]*torch.ones(two_layer_graph_vector[now_positive_index].shape).cuda()*torch.tensor(0.1).cuda() + graph_vector[now_positive_index]
            #new_negative_vector = two_layer_graph_vector[now_negative_index]*torch.ones(two_layer_graph_vector[now_positive_index].shape).cuda()*torch.tensor(0.1).cuda() + graph_vector[now_negative_index]
            positive_vector_list.append(new_positive_vector.view(1,-1))
            negative_vector_list.append(new_negative_vector.view(1,-1))
        #neg_vector = self.encode(self.ctx_model, data['negative']['neg'])
        ctx_vector = torch.cat([torch.cat(positive_vector_list, dim=0), torch.cat(negative_vector_list, dim=0)], dim=0)
        dot_products = torch.matmul(qry_vector, ctx_vector.transpose(0, 1))
        probs = F.log_softmax(dot_products, dim=1)
        loss = F.nll_loss(probs, data['positive'].long())
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == data['positive']).sum() / (data['positive'].shape[0] - (data['positive'].long() == torch.ones(len(data['positive'])).long().cuda()*(-100)).sum())
        return loss, accuracy

    def save_pretrained(self, addr):
        self.ctx_model.save_pretrained(addr + '/ctx')
        self.qry_model.save_pretrained(addr + '/qry')
        torch.save({'conv':self.conv.state_dict()}, addr + '/conv')


class BiEncoder(torch.nn.Module):
    def __init__(self, encoder_gpu_train_limit):
        ctx_model = '/root/data/huihuan/model/ctx'#'/root/data/huihuan/model/ctx' #'facebook/dpr-ctx_encoder-multiset-base'
        qry_model = '/root/data/huihuan/model/qry' #'/root/data/huihuan/model/qry' #'facebook/dpr-question_encoder-multiset-base'
        os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement') 
        #os.chdir('/root/data/huihuan/acl_project/retriever/dpr_retriever/passage_level/my_implement')
        super().__init__()
        self.ctx_model = EncoderWrapper(DPRContextEncoder.from_pretrained(ctx_model))
        self.qry_model = DPRQuestionEncoder.from_pretrained(qry_model)
        self.qry_model_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(qry_model)
        self.qry_model = EncoderWrapper(self.qry_model)
        self.encoder_gpu_train_limit = encoder_gpu_train_limit

    def encode(self, model, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)

    def forward(self, data):
        ctx_vector = self.encode(self.ctx_model, data['ctx'])
        qry_vector = self.encode(self.qry_model, data['qry'])
        dot_products = torch.matmul(qry_vector, ctx_vector.transpose(0, 1))
        probs = F.log_softmax(dot_products, dim=1)
        loss = F.nll_loss(probs, data['positive'].long())
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == data['positive']).sum() / data['positive'].shape[0]
        return loss, accuracy

    def save_pretrained(self, addr):
        self.ctx_model.save_pretrained(addr + '/ctx')
        self.qry_model.save_pretrained(addr + '/qry')
