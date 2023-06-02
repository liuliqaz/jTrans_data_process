import sys
from datautils.playdata import DatasetBase as DatasetBase
import networkx
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import re
import readidadata
import torch
import random
import time
import math


MAXLEN = 512
vocab_data = open("./jtrans_tokenizer/vocab.txt").read().strip().split("\n") + ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
my_vocab = defaultdict(lambda: 512, {vocab_data[i] : i for i in range(len(vocab_data))})


def help_tokenize(line):
    global my_vocab
    ret = {}
    split_line = line.strip().split(' ')
    split_line_len = len(split_line)
    if split_line_len <= 509:
        split_line = ['[CLS]']+split_line+['[SEP]']
        attention_mask = [1] * len(split_line) + [0] * (512 - len(split_line))
        split_line = split_line + (512-len(split_line))*['[PAD]']
    else:
        split_line = ['[CLS]'] + split_line[:510] + ['[SEP]']
        attention_mask = [1]*512
    input_ids = [my_vocab[e] for e in split_line]
    ret['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
    ret['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    return ret


# gen func_str(ins_1 ins_2...) and paragraph_str(block_1 \t block_2...)
def gen_funcstr_parastr_for_pretrain(f, convert_jump):
    cfg = f[3]
    #print(hex(f[0]))
    bb_ls, code_lst, map_id = [], [], {}
    block_id = {}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()

    for idx, block in enumerate(bb_ls):
        block_id[block] = idx

    paragraphs = {}
    for bx in range(len(bb_ls)):
        bb = bb_ls[bx]
        asm = cfg.nodes[bb]['asm']
        map_id[bb] = len(code_lst)

        temp_asm = []
        for code in asm:
            operator, operand1, operand2, operand3, annotation = readidadata.parse_asm(code)
            code_lst.append(operator)
            temp_asm.append(operator)
            if operand1 is not None:
                code_lst.append(operand1)
                temp_asm.append(operand1)
            if operand2 is not None:
                code_lst.append(operand2)
                temp_asm.append(operand2)
            if operand3 is not None:
                code_lst.append(operand3)
                temp_asm.append(operand3)
        paragraphs[bb] = temp_asm
    #
    for b_addr, block in paragraphs.items():
        for c in range(len(block)):
            op = block[c]
            if op.startswith('hex_'):
                jumpaddr = int(op[4:], base=16)
                if block_id.get(jumpaddr):
                    jumpid = block_id[jumpaddr]
                    if jumpid < MAXLEN:
                        block[c] = 'JUMP_ADDR_{}'.format(jumpid)
                    else:
                        block[c] = 'JUMP_ADDR_EXCEEDED'
                else:
                    block[c] = 'UNK_JUMP_ADDR'
                if not convert_jump:
                    block[c] = 'CONST'
    for c in range(len(code_lst)):
        op = code_lst[c]
        if op.startswith('hex_'):
            jumpaddr = int(op[4:], base=16)
            if map_id.get(jumpaddr):
                jumpid = map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c] = 'JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c] = 'JUMP_ADDR_EXCEEDED'
            else:
                code_lst[c] = 'UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c] = 'CONST'
    func_str = ' '.join(code_lst)
    paragraphs_str_list = [' '.join(p) for p in paragraphs.values()]
    paragraphs_str = '\t'.join(paragraphs_str_list)
    return func_str, paragraphs_str


# gen block pair for pretrain (block_1 [relate_token] block_2)
def gen_block_pair_for_pretrain(f):
    cfg = f[3]
    # print(hex(f[0]))
    bb_ls, code_lst, map_id = [], [], {}
    block_id = {}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()

    for idx, block in enumerate(bb_ls):
        block_id[block] = idx

    paragraphs = {}
    for bx in range(len(bb_ls)):
        bb = bb_ls[bx]
        asm = cfg.nodes[bb]['asm']
        map_id[bb] = len(code_lst)

        temp_asm = []
        for code in asm:
            operator, operand1, operand2, operand3, annotation = readidadata.parse_asm(code)
            code_lst.append(operator)
            temp_asm.append(operator)
            if operand1 is not None:
                code_lst.append(operand1)
                temp_asm.append(operand1)
            if operand2 is not None:
                code_lst.append(operand2)
                temp_asm.append(operand2)
            if operand3 is not None:
                code_lst.append(operand3)
                temp_asm.append(operand3)
        paragraphs[bb] = temp_asm

    for b_addr, block in paragraphs.items():
        for c in range(len(block)):
            op = block[c]
            if op.startswith('hex_'):
                jumpaddr = int(op[4:], base=16)
                if block_id.get(jumpaddr):
                    jumpid = block_id[jumpaddr]
                    if jumpid < MAXLEN:
                        block[c] = 'JUMP_ADDR_{}'.format(jumpid)
                    else:
                        block[c] = 'JUMP_ADDR_EXCEEDED'
                else:
                    block[c] = 'UNK_JUMP_ADDR'

    edge_list = []
    # edge_id_list = [(block_id[addr[0]], block_id[addr[1]]) for addr in cfg.edges]
    for edge in cfg.edges:
        pre = paragraphs[edge[0]]
        suf = paragraphs[edge[1]]
        match_jump_id = re.match(r'^JUMP_ADDR_([0-9]*)$', pre[-1], re.M | re.I)
        if match_jump_id:
        # if len(pre) > 2 and pre[-1].startswith('hex_'):
            j_addr = pre[-1]
            j_op = pre[-2]
            # jumpaddr = int(j_addr[4:], base=16)
            jumpaddr = int(match_jump_id.group(1))
            if block_id[edge[1]] == jumpaddr:
                suf_str = f'[t_{j_op}] ' + ' '.join(suf)
            else:
                suf_str = f'[f_{j_op}] ' + ' '.join(suf)
            edge_str = ' '.join(pre) + '\t' + suf_str + '\n'
            edge_list.append(edge_str)
        else:
            suf_str = '[seq] ' + ' '.join(suf)
            edge_str = ' '.join(pre) + '\t' + suf_str + '\n'
            edge_list.append(edge_str)
    return edge_list


# gen
def gen_graph_triple_for_finetune(f):
    pass

def gen_all_path(f):
    cfg = f[3]
    # print(hex(f[0]))
    bb_ls, code_lst, map_id = [], [], {}
    block_id = {}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()

    for idx, block in enumerate(bb_ls):
        block_id[block] = idx

    paragraphs = {}
    for bx in range(len(bb_ls)):
        bb = bb_ls[bx]
        asm = cfg.nodes[bb]['asm']
        map_id[bb] = len(code_lst)

        temp_asm = []
        for code in asm:
            operator, operand1, operand2, operand3, annotation = readidadata.parse_asm(code)
            code_lst.append(operator)
            temp_asm.append(operator)
            if operand1 is not None:
                code_lst.append(operand1)
                temp_asm.append(operand1)
            if operand2 is not None:
                code_lst.append(operand2)
                temp_asm.append(operand2)
            if operand3 is not None:
                code_lst.append(operand3)
                temp_asm.append(operand3)
        paragraphs[bb] = temp_asm

    edge_list = []
    for edge in cfg.edges:
        pre = paragraphs[edge[0]]
        suf = paragraphs[edge[1]]
        if len(pre) > 2 and pre[-1].startswith('hex_'):
            j_addr = pre[-1]
            j_op = pre[-2]
            jumpaddr = int(j_addr[4:], base=16)
            if edge[1] == jumpaddr:
                suf_str = f'[t_{j_op}] ' + ' '.join(suf)
            else:
                suf_str = f'[f_{j_op}] ' + ' '.join(suf)
            edge_str = ' '.join(pre) + '\t' + suf_str + '\n'
            edge_list.append(edge_str)
        else:
            suf_str = '[seq] ' + ' '.join(suf)
            edge_str = ' '.join(pre) + '\t' + suf_str + '\n'
            edge_list.append(edge_str)
    return edge_list


def get_all_dir(data_dir):
    proj_list = []
    for file_name in os.listdir(data_dir):
        pickle_path = os.path.join(data_dir, file_name)
        if os.path.isdir(pickle_path):
            proj_list.append(file_name)
    return proj_list


def load_unpair_data(data_dir, target_dir):
    proj_list = get_all_dir(data_dir)
    proj_len = len(proj_list)
    step = 10
    ph_f = open(target_dir, 'a+')
    for i in tqdm(range(math.ceil(proj_len/step))):
        # prob = random.random()
        # if prob > 0.2:
        #     continue
        process_proj = proj_list[i*step: min((i+1)*step, proj_len)]
        para_write_to_file(data_dir, process_proj, ph_f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def para_write_to_file(data_dir, proj_list, ph_f):
    unpaired = defaultdict(list)
    for proj in proj_list:
        for filename in os.listdir(os.path.join(data_dir, proj)):
            if filename != 'saved_index.pkl':
                pkl_path = os.path.join(data_dir, proj, filename)
                pickle_data = load_pickle(pkl_path)
                unpaired[proj].append(pickle_data)
    for proj, pkl_list in tqdm(unpaired.items()):
        for pkl in pkl_list:
            for func_name, func_data in pkl.items():
                func_addr, asm_list, rawbytes_list, cfg, biai_featrue = func_data
                # proj, func_name, func_addr, asm_list, rawbytes_list, cfg, biai_featrue
                func_info = (func_addr, asm_list, rawbytes_list, cfg, biai_featrue)
                # -- dump as paragraph --
                # func_str, paragraphs_str = gen_funcstr(func_info, True)
                # if len(paragraphs_str) > 0:
                #     ph_f.write('[ph]' + str(proj) + '@' + str(func_name) + '\n')
                #     ph_f.write(paragraphs_str + '\n')

                # -- dump as pairs --
                edge_list = gen_block_pair_for_pretrain(func_info)
                # if len(edge_list) > 0:
                #     ph_f.writelines(edge_list)

                # edge_list = gen_block_pair(func_info)

def load_pretrain_data():
    # data_path = './data/extract'
    # target_path = 'jTrans_pairs.txt'
    # target_path = 'paragraphs_pair.txt'

    data_path = './datautils/extract'
    target_path = 'jTrans_pair_micro.txt'
    load_unpair_data(data_path, target_path)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, default='./data/extract')
    # args = parser.parse_args()
    #
    # load_unpair_data(args.dataset_path)

    load_pretrain_data()
    pass



    # with open('./paragraphs.txt', 'r') as f:
    #     p_lines = f.readlines()
    # res_lines = []
    # for line in p_lines:
    #     if line[:4] == '[ph]':
    #         continue
    #     res_lines.append(line[:-1].split('\t'))
    #
    # print('p_lines', len(p_lines))
    # print('res_line', len(res_lines))
    # help_tokenize(res_lines[0])
