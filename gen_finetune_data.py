import math
from unicodedata import name
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import torch.multiprocessing
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data import FunctionDataset_CL_Load
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import sys
import time
import data
import pickle
import random
from data import help_tokenize
from bert_pytorch import BERT, WordVocab

MAX_LEN=64

# decrept
def gen_block_feat(model, block_batch, vocab):
    bert_input_batch = []
    segment_label_batch = []
    for b in block_batch:
        t1 = [vocab.sos_index] + [vocab.stoi.get(ins, vocab.unk_index) for ins in b] + [vocab.eos_index]

        segment_label = [1 for _ in range(len(t1))][:MAX_LEN]
        bert_input = t1[:MAX_LEN]

        padding = [vocab.pad_index for _ in range(MAX_LEN - len(bert_input))]
        bert_input.extend(padding)
        segment_label.extend(padding)

        bert_input_batch.append(torch.tensor(bert_input))
        segment_label_batch.append(torch.tensor(segment_label))


    bert_input_batch = torch.stack(bert_input_batch).cuda()
    segment_label_batch = torch.stack(segment_label_batch).cuda()
    y = model(bert_input_batch, segment_label_batch)
    return y[:, 0]


def gen_block_edge_feat_plus(model, batch, vocab, device, is_edge_pair=False):
    step = 64
    if len(batch) <= step:
        return gen_block_edge_feat(model, batch, vocab, device, is_edge_pair)
    iter = math.ceil(len(batch) / step)
    partition = []
    for i in range(iter):
        temp_feat = gen_block_edge_feat(model, batch[i*step:min((i+1)*step, len(batch))], vocab, device, is_edge_pair)
        partition.append(temp_feat)
    return torch.cat(partition, dim=0)


def gen_block_edge_feat(model, batch, vocab, device, is_edge_pair=False):
    bert_input_batch = []
    segment_label_batch = []
    for b in batch:
        if is_edge_pair:
            e = b
            t1 = [vocab.sos_index] + [vocab.stoi.get(ins, vocab.unk_index) for ins in e[0]] + [vocab.eos_index]
            t2 = [vocab.stoi.get(ins, vocab.unk_index) for ins in e[1]] + [vocab.eos_index]
            segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:MAX_LEN]
            bert_input = (t1 + t2)[:MAX_LEN]
        else:
            t1 = [vocab.sos_index] + [vocab.stoi.get(ins, vocab.unk_index) for ins in b] + [vocab.eos_index]
            segment_label = [1 for _ in range(len(t1))][:MAX_LEN]
            bert_input = t1[:MAX_LEN]
        padding = [vocab.pad_index for _ in range(MAX_LEN - len(bert_input))]
        bert_input.extend(padding)
        segment_label.extend(padding)

        bert_input_batch.append(torch.tensor(bert_input))
        segment_label_batch.append(torch.tensor(segment_label))

    bert_input_batch = torch.stack(bert_input_batch).to(device)
    segment_label_batch = torch.stack(segment_label_batch).to(device)

    y = model(bert_input_batch, segment_label_batch)
    y = y.detach().cpu()
    return y[:, 0]


def gen_data(function, model, vocab, device):
    block_asm_list = function[0]
    edge_pair_list = function[1]

    block_batch = [block for block in block_asm_list]
    block_feat_list = gen_block_edge_feat_plus(model, block_batch, vocab, device)

    edge_feat_list = torch.tensor([])
    if len(edge_pair_list) > 0:
        edge_batch = [(block_asm_list[edge[0]], block_asm_list[edge[1]]) for edge in edge_pair_list]
        edge_feat_list = gen_block_edge_feat_plus(bert, edge_batch, vocab, device, is_edge_pair=True)

    ret_data = (block_feat_list, edge_pair_list, edge_feat_list)
    return ret_data


if __name__ == '__main__':
    device = torch.device('cuda')

    # train_path = './datautils/extract'
    train_path = '../jTrans/data/extract'

    tokenizer = './jtrans_tokenizer'
    load_train, load_test = False, False
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    ft_train_dataset = FunctionDataset_CL_Load(tokenizer, train_path, convert_jump_addr=True, load=load_train, opt=['O0', 'O1', 'O2', 'O3', 'Os'])

    vocab_path = './data/jtrans_x86.pkl'
    vocab = WordVocab.load_vocab(vocab_path)
    # bert = BERT(len(vocab), hidden=768, n_layers=12, attn_heads=12)
    # bert.load_state_dict(torch.load('./saved_model/bert.model.ep8'))
    bert = torch.load('./saved_model/bert.model.ep8')
    bert.eval()
    bert.to(device)

    triple_train_list = []
    for i in tqdm(range(len(ft_train_dataset.datas))):
        f, g_sim, g_unsim = ft_train_dataset[i]
        # f_data = gen_data(f, bert, vocab, device)
        # g_sim_data = gen_data(g_sim, bert, vocab, device)
        # g_unsim_data = gen_data(g_unsim, bert, vocab, device)
        #
        # triple_train_list.append((f_data, g_sim_data, g_unsim_data))

        triple_train_list.append((f, g_sim, g_unsim))

        if len(triple_train_list) == 1000:
            if os.path.exists('./data/person.pkl'):
                with open('./data/person.pkl', 'ab') as f:
                    pickle.dump(triple_train_list, f)
            else:
                with open('./data/person.pkl', 'wb') as f:
                    pickle.dump(triple_train_list, f)
            triple_train_list = []

    with open('./data/person.pkl', 'ab') as f:
        pickle.dump(triple_train_list, f)


