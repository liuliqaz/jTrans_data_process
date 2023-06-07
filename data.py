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


def help_tokenize_blk_list(blk_list):
    res_list = []
    for blk in blk_list:
        res_list.append(help_tokenize(blk))
    return res_list

def help_tokenize(line):
    global my_vocab
    ret = {}
    if isinstance(line, list):
        split_line = line
    else:
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


def gen_funcstr(f, convert_jump):
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
        asm=cfg.nodes[bb]['asm']
        map_id[bb]=len(code_lst)

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
            jumpaddr = int(op[4:],base=16)
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


def gen_funcstr_old(f, convert_jump):
    cfg=f[3]
    #print(hex(f[0]))
    bb_ls,code_lst,map_id=[],[],{}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()
    for bx in range(len(bb_ls)):
        bb=bb_ls[bx]
        asm=cfg.nodes[bb]['asm']
        map_id[bb]=len(code_lst)
        for code in asm:
            operator,operand1,operand2,operand3,annotation=readidadata.parse_asm(code)
            code_lst.append(operator)
            if operand1!=None:
                code_lst.append(operand1)
            if operand2!=None:
                code_lst.append(operand2)
            if operand3!=None:
                code_lst.append(operand3)
    for c in range(len(code_lst)):
        op=code_lst[c]
        if op.startswith('hex_'):
            jumpaddr=int(op[4:],base=16)
            if map_id.get(jumpaddr):
                jumpid=map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c]='JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c]='JUMP_ADDR_EXCEEDED'
            else:
                code_lst[c]='UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c]='CONST'
    func_str=' '.join(code_lst)
    return func_str


def get_all_dir(data_dir):
    proj_list = []
    for file_name in os.listdir(data_dir):
        pickle_path = os.path.join(data_dir, file_name)
        if os.path.isdir(pickle_path):
            proj_list.append(file_name)
    return proj_list


def load_unpair_data(data_dir):
    proj_list = get_all_dir(data_dir)
    proj_len = len(proj_list)
    step = 10
    for i in tqdm(range(math.ceil(proj_len/step))):
        process_proj = proj_len[i*step: min((i+1)*step, proj_len)]




def load_unpair_data(datapath, filt=None, alldata=True, convert_jump=True, opt=None, fp=None, paragraph_file=None):
    dataset = DatasetBase(datapath, filt, alldata)
    dataset.load_unpair_data()
    functions = []
    for i in tqdm(dataset.get_unpaird_data()):  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        f = (i[2], i[3], i[4], i[5], i[6])
        func_str, paragraphs_str = gen_funcstr(f, convert_jump)
        print("debug use")
        # if len(func_str) > 0:
        #     fp.write(func_str + "\n")
        # if len(paragraphs_str) > 0:
        #     paragraph_file.write('[ph]' + str(i[0]) + '_' + str(i[1]) + '\n')
        #     paragraph_file.write(paragraphs_str + '\n')


def load_paired_data(datapath, filt=None, alldata=True, convert_jump=True, opt=None):
    dataset = DatasetBase(datapath, filt, alldata, opt=opt)
    functions = []
    func_emb_data = []
    SUM = 0
    for i in dataset.get_paired_data_iter():  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        functions.append([])
        func_emb_data.append({'proj': i[0], 'funcname': i[1]})
        for o in opt:
            if i[2].get(o):
                '''
                add graph process here
                '''
                f = i[2][o]
                cfg = f[3]
                blk_list, edge_list = gen_blocks_edges(cfg)
                if len(blk_list) > 0:
                    SUM += 1
                    func_emb_data[-1][o] = len(functions[-1])
                    functions[-1].append((blk_list, edge_list))
                    SUM += 1

    print('TOTAL ', SUM)
    return functions, func_emb_data


def gen_blocks_edges(cfg):
    # print(hex(f[0]))
    bb_ls = []
    block_id_map = {}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()

    for idx, block in enumerate(bb_ls):
        block_id_map[block] = idx

    block_asm_map = {}
    for bx in range(len(bb_ls)):
        bb = bb_ls[bx]
        asm = cfg.nodes[bb]['asm']
        temp_asm = []
        for code in asm:
            operator, operand1, operand2, operand3, annotation = readidadata.parse_asm(code)
            temp_asm.append(operator)
            if operand1 is not None:
                temp_asm.append(operand1)
            if operand2 is not None:
                temp_asm.append(operand2)
            if operand3 is not None:
                temp_asm.append(operand3)
        block_asm_map[bb] = temp_asm
    for b_addr, block in block_asm_map.items():
        for c in range(len(block)):
            op = block[c]
            if op.startswith('hex_'):
                jumpaddr = int(op[4:], base=16)
                if block_id_map.get(jumpaddr):
                    jumpid = block_id_map[jumpaddr]
                    if jumpid < MAXLEN:
                        block[c] = 'JUMP_ADDR_{}'.format(jumpid)
                    else:
                        block[c] = 'JUMP_ADDR_EXCEEDED'
                else:
                    block[c] = 'UNK_JUMP_ADDR'
    edge_list = []
    for edge in cfg.edges:
        edge_list.append([block_id_map[edge[0]], block_id_map[edge[1]]])

    block_asm_list = ['' for _ in range(len(bb_ls))]
    for blk_addr, idx in block_id_map.items():
        block_asm_list[idx] = block_asm_map[blk_addr]
    return block_asm_list, edge_list


class FunctionDataset_CL_Load(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer, path='../BinaryCorp/extract',filt=None,alldata=True,convert_jump_addr=True,opt=None,load=None,need_tokenize=False):  #random visit
        if load:
            start = time.time()
            self.datas = pickle.load(open(load, 'rb'))
            print('load time:', time.time() - start)
            self.tokenizer = tokenizer
            self.opt = opt
            self.convert_jump_addr = True
        else:
            functions, func_ebeds = load_paired_data(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt)
            self.datas=[]
            if need_tokenize:
                for func_list in functions:
                    tmp = []
                    for f in func_list:
                        tmp.append((help_tokenize_blk_list(f[0]), f[1]))
                    self.datas.append(tmp)
            else:
                self.datas=functions
            self.ebds = func_ebeds
            self.tokenizer = tokenizer
            self.opt = opt
            self.convert_jump_addr = True

    def __getitem__(self, idx):
        pairs = self.datas[idx]
        # get sim pair
        pos = random.randint(0, len(pairs) - 1)
        pos2 = random.randint(0, len(pairs) - 1)
        while pos2 == pos:
            pos2 = random.randint(0,len(pairs)-1)
        f1 = pairs[pos]   #give three pairs
        f2 = pairs[pos2]

        # get un-sim pair
        unsim_idx = random.randint(0, len(self.datas) - 1)
        while unsim_idx == idx:
            unsim_idx = random.randint(0,len(self.datas)-1)
        unsim_pairs = self.datas[unsim_idx]
        pos3 = random.randint(0, len(unsim_pairs)-1)
        f3 = unsim_pairs[pos3]

        return f1, f2, f3

    def __len__(self):
        return len(self.datas)


def load_filter_list(name):
    import csv
    f = csv.reader(open(name, 'r'))
    S = set()
    for i in f:
        S.add(i[1])
    return list(S)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/extract')
    parser.add_argument('--prefixfilter', type=str, default=None)
    parser.add_argument('--all_data', type=bool, default=True)
    args = parser.parse_args()

    f = open('./sentences.txt', 'a+')
    ph_f = open('./paragraphs.txt', 'a+')
    load_unpair_data(args.dataset_path, args.prefixfilter, fp=f, paragraph_file=ph_f)
    f.close()

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
