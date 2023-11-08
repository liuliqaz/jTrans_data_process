import os
import json
import numpy as np
import torch
import math
from tqdm import tqdm
from model_utils import BertCustomModel
import random
import pickle
from transformers import AutoConfig, AutoTokenizer

MAX_STEP = 50

partition_i = 0

class FinetuneDataset():
    def __init__(self, graph_list, model, device, tokenizer, max_seq_length):
        self.graphs = graph_list
        self.model = model
        self.device =device
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # mapping all raw grap data into embedding form
        self.data_list = self.map_data(graph_list)
        self.model.cpu()
    
    def get_data_list(self):
        return self.data_list

    # traverse all data and transfer to embeddings
    def map_data(self, raw_graph_list):
        res_list = []
        progress_bar = tqdm(range(len(raw_graph_list)))
        for data_tri_tuple in raw_graph_list:
            target_func = data_tri_tuple[0]
            sim_func = data_tri_tuple[1]
            unsim_func = data_tri_tuple[2]

            target_nodes = target_func['nodes']
            target_preds = [target_nodes[edge[0]] for edge in target_func['edges']]
            target_succs = [target_nodes[edge[1]] for edge in target_func['edges']]
            nodes1 = self.gen_block_edge_feat_plus(target_nodes, target_func['arch'])
            edges1 = self.gen_block_edge_feat_plus(target_preds, target_func['arch'], is_pair=True, batch2=target_succs)

            sim_nodes = sim_func['nodes']
            sim_preds = [sim_nodes[edge[0]] for edge in sim_func['edges']]
            sim_succs = [sim_nodes[edge[1]] for edge in sim_func['edges']]
            nodes2 = self.gen_block_edge_feat_plus(sim_nodes, sim_func['arch'])
            edges2 = self.gen_block_edge_feat_plus(sim_preds, sim_func['arch'], is_pair=True, batch2=sim_succs)

            unsim_nodes = unsim_func['nodes']
            unsim_preds = [unsim_nodes[edge[0]] for edge in unsim_func['edges']]
            unsim_succs = [unsim_nodes[edge[1]] for edge in unsim_func['edges']]
            nodes3 = self.gen_block_edge_feat_plus(unsim_nodes, unsim_func['arch'])
            edges3 = self.gen_block_edge_feat_plus(unsim_preds, unsim_func['arch'], is_pair=True, batch2=unsim_succs)

            progress_bar.update(1)

            res_list.append(
                [
                    (self.trans_tensor_to_list(nodes1), target_func['edges'], self.trans_tensor_to_list(edges1)),
                    (self.trans_tensor_to_list(nodes2), sim_func['edges'], self.trans_tensor_to_list(edges2)),
                    # (self.trans_tensor_to_list(nodes3), unsim_func['edges'], self.trans_tensor_to_list(edges3)), # when use cross entropy
                ]
            )

            # if len(res_list) == 10000:
            #     global partition_i
            #     save_path = f'/home/liu/bcsd/datasets/edge_gnn_datas/finetune_triple_tensor_5w_{partition_i}.pkl'
            #     with open(save_path, 'wb') as f:
            #         pickle.dump(res_list, f)
            #     partition_i += 1
            #     res_list = []

        return res_list
    
    def trans_tensor_to_list(self, tt):
        target_list = tt.numpy().tolist()
        return target_list

def gen_block_edge_feat_plus(model, device, tokenizer, max_seq_length, batch1, arch, is_pair=False, batch2=None):
    if len(batch1) == 0:
        return torch.tensor([])

    if is_pair:
        result_input = tokenizer(
            batch1,
            batch2,            
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=False,
        )
    else:
        result_input = tokenizer(
            batch1,        
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=False,
        )
    result_input['arch_ids'] = [[arch]*max_seq_length for _ in batch1]

    step = MAX_STEP
    batch_len = len(batch1)
    if batch_len <= step:
        return gen_embedding(model, device, 
                             result_input['input_ids'],
                             result_input['attention_mask'],
                             result_input['token_type_ids'],
                             result_input['arch_ids'])
    iter = math.ceil(batch_len / step)
    partition = []
        
    for i in range(iter):
        temp_feat = gen_embedding(model, device, 
                                  result_input['input_ids'][i * step: min((i + 1) * step, batch_len)],
                                  result_input['attention_mask'][i * step: min((i + 1) * step, batch_len)],
                                  result_input['token_type_ids'][i * step: min((i + 1) * step, batch_len)],
                                  result_input['arch_ids'][i * step: min((i + 1) * step, batch_len)])
        partition.append(temp_feat)
    return torch.cat(partition, dim=0)
        
def gen_embedding(model, device, input_ids, attention_mask, token_type_ids, arch_ids):
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)
    token_type_ids = torch.LongTensor(token_type_ids).to(device)
    arch_ids = torch.LongTensor(arch_ids).to(device)

    model.to(device)
    model.eval()

    output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, arch_ids=arch_ids)
    pooled_output = output[0][:, 0].detach().cpu()
    return pooled_output


if __name__ == '__main__':

    file = './data/eval_func_list_node_1_50.pkl'
    with open(file, 'rb') as f:
        data =  pickle.load(f)

    tokenizer_name = '/home/liu/bcsd/datasets/edge_gnn_datas/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False, do_lower_case=False, do_basic_tokenize=False)

    config_name = '../bert_torch/saved_model/edge_big_384/config.json'
    model_name ='../bert_torch/saved_model/edge_big_384/pytorch_model.bin'
    config = AutoConfig.from_pretrained(config_name)
    encode_model = BertCustomModel.from_pretrained(
        model_name, 
        config=config,
        add_pooling_layer=False,
        custom_config={'arch_num':3}
    )
    device = torch.device('cuda:0')
    max_seq_len = 512

    progress_bar = tqdm(range(len(data)*72))

    for bin_func, opt_dict in data.items():
        for opt, func_dict in opt_dict.items():
            target_nodes = func_dict['nodes']
            target_preds = [target_nodes[edge[0]] for edge in func_dict['edges']]
            target_succs = [target_nodes[edge[1]] for edge in func_dict['edges']]
            nodes_embedding = gen_block_edge_feat_plus(encode_model, device, tokenizer, max_seq_len, target_nodes, func_dict['arch'])
            edges_embedding = gen_block_edge_feat_plus(encode_model, device, tokenizer, max_seq_len, target_preds, func_dict['arch'], is_pair=True, batch2=target_succs)
            
            func_dict['nodes_embedding'] = nodes_embedding
            func_dict['edges_embedding'] = edges_embedding            

            progress_bar.update(1)


    save_path = './data/eval_func_list_node_1_50_tensor.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)   
    
