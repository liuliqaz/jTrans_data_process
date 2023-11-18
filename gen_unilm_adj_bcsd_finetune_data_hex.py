import pickle
import os
import argparse
import re
from tqdm import tqdm
import random
import numpy as np
import math
from collections import deque
import itertools
import base64


MIN_NODE_LEN = 2
MAX_NODE_LEN = 50
MAX_EDGE_LEN = 100

SAMPLE_NUM = 1000000

FILTER_CLANG = False
FILTER_64 = False

node_cnt_x = 0
edge_cnt_x = 0


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_all_pkl_file(data_dir):
    proj_list = []
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('pkl'):
            continue
        pickle_path = os.path.join(data_dir, file_name)
        proj_list.append(pickle_path)
    return proj_list


def tokenize_bracket_comma(ins_str):
    ins_str = ins_str.replace('[', ' [ ')
    ins_str = ins_str.replace(']', ' ] ')
    ins_str = ins_str.replace('(', ' ( ')
    ins_str = ins_str.replace(')', ' ) ')
    ins_str = ins_str.replace('{', ' { ')
    ins_str = ins_str.replace('}', ' } ')
    ins_str = ins_str.replace('-', ' - ')
    ins_str = ins_str.replace(':', ' : ')
    ins_str = ins_str.replace('!', ' ! ')
    ins_str = ins_str.replace('*', ' * ')
    ins_str = ins_str.replace('+', ' + ')
    ins_str = ins_str.replace(',', '')
    ins_str = ' '.join(ins_str.strip().split())
    return ins_str


def is_hexadecimal(s):
    try:
        if s[0] == '#':
            s = s[1:]
        int(s, 16)
        return True
    except ValueError:
        return False


def get_arch_emb(arch):
    if 'x86' in arch:
        return 0
    if 'arm' in arch:
        return 1
    if 'mips' in arch:
        return 2
    print(f'[!]unkown arch, arch={arch}')
    return -1


def process_hex(arch, func_dict, dyn_func_list, binary_name):
    func_map = dict()
    for func_addr, func_data in func_dict.items():
        func_name = func_data['name']
        edge_list = func_data['edges']
        node_list = func_data['nodes']
        basic_blocks = func_data['basic_blocks']

        if len(node_list) > MAX_NODE_LEN or len(node_list) < MIN_NODE_LEN:
            global node_cnt_x
            node_cnt_x += 1
            continue

        node_list.sort()

        func_addr_int = int(func_addr, 16)
        while node_list[0] != func_addr_int:
            node_list.insert(len(node_list)-1, node_list.pop(0))

        if func_name in dyn_func_list:
            continue

        # {addr: hex_str}
        hex_dict = {}

        for addr, block_data in basic_blocks.items():
            b64_str = block_data['b64_bytes']
            hex_str = base64.b64decode(b64_str).hex()
            hex_list = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
            hex_dict[addr] = ' '.join(hex_list)
        
        if func_name in func_map:
            func_map[f'{func_name}_{func_addr}'] = func_addr
            print('[!]dup func name')
        else:
            func_map[func_name] = func_addr
        
        new_edge_list = []
        new_node_list = []
        tmp_node_id_dict = {}
        # new_node_list has same index as node_list
        for idx, node in enumerate(node_list):
            if len(hex_dict[node]) == 0:
                continue
            tmp_node_id_dict[node] = len(new_node_list)
            new_node_list.append(hex_dict[node])

        # use index of new node list present edge
        for edge in edge_list:
            if edge[0] in tmp_node_id_dict and edge[1] in tmp_node_id_dict:
                new_edge_list.append((tmp_node_id_dict[edge[0]], tmp_node_id_dict[edge[1]]))
        
        func_dict[func_addr]['edges'] = new_edge_list
        func_dict[func_addr]['nodes'] = new_node_list

    return func_map


def gather_pkl_file_name(data_dir):
    pkl_file_list = get_all_pkl_file(data_dir)
    pkl_file_len = len(pkl_file_list)

    proj_bin_dict = dict()

    for file_name in pkl_file_list: 
        file_name = file_name.split('/')[-1]
        proj = file_name.split('_')[0]
        bin_name = file_name.split('_')[-2]
        arch_opt = '_'.join(file_name.split('_')[1:-2])

        dict_key = f'{proj}_{bin_name}'
        if dict_key in proj_bin_dict:
            proj_bin_dict[dict_key].append((file_name, arch_opt))
        else:    
            proj_bin_dict[dict_key] = [(file_name, arch_opt)]
    
    return proj_bin_dict, pkl_file_len


def process_and_gather_unilm_adj_pretrain_data(data_dir, outpur_dir, pkl_name):
    # get all file dict {proj_bin:[(file_name, arch_opt)]}
    proj_bin_dict, total_len = gather_pkl_file_name(data_dir)

    progress_bar = tqdm(range(total_len))
    res_list = []
    save_path = os.path.join(outpur_dir, pkl_name)

    for proj_bin, file_tuple_list in proj_bin_dict.items():
        # save_file_name = os.path.join(outpur_dir, f'{proj_bin}_index.pkl')
        
        proj_bin_func_set = set()   # save common function name
        proj_bin_opt_dict = dict()  # save {func_name:{opt1:{info}, opt2:{info}}}

        # traverse every arch_opt
        for file_tuple in file_tuple_list:
            file_name = file_tuple[0]
            arch_opt = file_tuple[1]

            # remove compiler clang to reduce dataset size
            if FILTER_CLANG and arch_opt.split('_')[0] == 'clang':
                progress_bar.update(1)
                continue

            # remove 64bit to reduce dataset size
            if FILTER_64 and arch_opt.split('_')[-2] == '64':
                progress_bar.update(1)
                continue

            binary_name = '_'.join(file_name[:-4].split('_')[:-1])

            file_path = os.path.join(data_dir, file_name)
            pickle_data = load_pickle(file_path)
            func_dict = pickle_data[binary_name]['func_dict']
            arch = pickle_data[binary_name]['arch']
            dyn_func_list = pickle_data[binary_name]['dyn_func_list']

            func_map = process_hex(arch, func_dict, dyn_func_list, binary_name)
            pickle_data[binary_name]['func_map'] = func_map

            for func_name, func_addr in func_map.items():
                # debug use, count node and edge > 160 funcs
                if len(func_dict[func_addr]['nodes']) > MAX_NODE_LEN or len(func_dict[func_addr]['nodes']) < MIN_NODE_LEN:
                    global node_cnt_x
                    node_cnt_x += 1
                    continue
                
                block_list = func_dict[func_addr]['nodes']
                edge_list = func_dict[func_addr]['edges']

                adj_matrix = gen_adj_matrix(edge_list, len(block_list))
                merge_input, block_index = get_input(block_list)

                depth_list = cal_node_depth(adj_matrix, 0)
                in_degrees, out_degrees = cal_degree(adj_matrix)

                if -1 in depth_list:
                    continue

                opt_map = {
                    'input': ' '.join(merge_input),
                    'block_index': block_index,
                    'adj_matrix': adj_matrix,
                    'depth_list': depth_list,
                    'in_degrees': in_degrees,
                    'out_degrees': out_degrees,
                    'arch': get_arch_emb(arch_opt)
                }

                if func_name not in proj_bin_opt_dict:
                    proj_bin_opt_dict[func_name] = {arch_opt: opt_map}
                else:    
                    proj_bin_opt_dict[func_name][arch_opt] = opt_map


            if len(proj_bin_func_set) == 0:
                proj_bin_func_set.update(func_map.keys())
            else:
                proj_bin_func_set &= set(func_map.keys())
            
            progress_bar.update(1)

        for func_name, func_opt_dict in proj_bin_opt_dict.items():
            if len(func_opt_dict) < 2:
                continue

            # full opt pairs
            all_sim_pairs = list(itertools.combinations(func_opt_dict.keys(), 2))
            sim_pair = random.sample(all_sim_pairs, 1)[0]
         
            src_data = func_opt_dict[sim_pair[0]]
            tgt_data = func_opt_dict[sim_pair[1]]

            # res_list.append([target_data, sim_data])
            res_list.append(
            {
                "src_hex": src_data['input'],
                "src_asm": '',
                'src_block_index': src_data['block_index'],
                'src_adj_matrix': src_data['adj_matrix'],
                'src_depth_list': src_data['depth_list'],
                'src_in_degrees': src_data['in_degrees'],
                'src_out_degrees': src_data['out_degrees'],
                "src_arch_id": src_data['arch'],

                "tgt_hex": tgt_data['input'],
                "tgt_asm": '',
                'tgt_block_index': tgt_data['block_index'],
                'tgt_adj_matrix': tgt_data['adj_matrix'],
                'tgt_depth_list':  tgt_data['depth_list'],
                'tgt_in_degrees':  tgt_data['in_degrees'],
                'tgt_out_degrees':  tgt_data['out_degrees'],
                "tgt_arch_id": tgt_data['arch'],
            }
        )

    with open(save_path, 'wb') as f:
        pickle.dump(res_list, f)                    


def sample_dataset(datalist, sample_num, save_path):
    sample_list = random.sample(datalist, sample_num)
    with open(save_path, 'wb') as f:
        pickle.dump(sample_list, f)


def cal_degree(adj_matrix):
    in_degrees = np.sum(np.array(adj_matrix), axis=0)
    out_degrees = np.sum(np.array(adj_matrix), axis=1)
    # d_center = degrees / (len(degrees) - 1)
    # d_center_l = d_center.tolist()
    return in_degrees.tolist(), out_degrees.tolist()


def cal_node_depth(matrix, start):
    num_nodes = len(matrix)
    distances = [-1] * num_nodes  # 初始化距离数组，-1表示不可达
    distances[start] = 0

    queue = deque()
    queue.append(start)

    while queue:
        node = queue.popleft()
        for neighbor in range(num_nodes):
            if matrix[node][neighbor] == 1 and distances[neighbor] == -1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    return distances


def gen_adj_matrix(edge_list, node_len):
    adj_matrix = [[0]*node_len for _ in range(node_len)]
    for edge in edge_list:
        adj_matrix[edge[0]][edge[1]] = 1
    
    return adj_matrix


def get_input(block_list):
    merge_input = []
    block_index = []
    for block in block_list:
        start_idx = len(merge_input)
        merge_input.extend(block.split())
        end_idx = len(merge_input)-1
        block_index.append((start_idx, end_idx))

    return merge_input, block_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gather project, bin_file, functions & generate finetune data.")
    parser.add_argument("--input_path", type=str, default='/home/liu/bcsd/train_set_extract_v2')
    parser.add_argument("--output_path", type=str, default='/home/liu/bcsd/datasets/edge_gnn_datas/')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # test use
    # input_path = '/home/liu/project/ida_script/extract'
    # output_path = './data'

    triple_name ='unilm_bcsd_finetune_max50_min1_full_hex.pkl'

    process_and_gather_unilm_adj_pretrain_data(input_path, output_path, triple_name)

    # data_path = os.path.join(output_path, triple_name)
    # data_triple = load_pickle(data_path)

    # cnt = 0
    # for i in data_triple:
    #     block_index = i['block_index']
    #     for dd in block_index:
    #         if dd[0] > dd[1]:
    #             print('err')
    #             cnt += 1


    # sample_path = os.path.join(output_path, 'finetune_triple_max50_min1_1M_without_clang64.pkl')
    # sample_dataset(data_triple, SAMPLE_NUM, sample_path)
    
    # small_data_triple = load_pickle('./data/finetune_triple_max100.pkl')

    print('done')
