import pickle
import os
import argparse
import re
from tqdm import tqdm
import random
import numpy as np
import math

MAX_LOOP_TIME = 100


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


def calculate_levenshtein_distance(text1, text2):
    m, n = len(text1), len(text2)
    dp = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n]


def get_rand_unsim_opt(target_opt, opt_list):
    rand_opt_list = opt_list.copy()
    unsim_opt = ""
    for opt in rand_opt_list:
        if opt == target_opt:
            continue
        if calculate_levenshtein_distance(target_opt, opt) >= 5:
            unsim_opt = opt
            break
        unsim_opt = opt
    if unsim_opt == '':
        print('debug')
    return unsim_opt


def get_arch_emb(arch):
    if 'x86' in arch:
        return 0
    if 'arm' in arch:
        return 1
    if 'mips' in arch:
        return 2
    print(f'[!]unkown arch, arch={arch}')
    return -1


def process_asm_x86(basic_blocks, func_dict, dyn_func_list, func_name):
    res_dict = dict()
    for block_addr, block_data in basic_blocks.items():
        block_asm_list = block_data['bb_disasm']
        res_block_asm_list = []
        for ins_str in block_asm_list:
            ins_str = tokenize_bracket_comma(ins_str)
            ins_list = ins_str.split()
            opcode = ins_list[0]
            # step1 parse jmp ins and parse target addr into DEC 
            if opcode[0] == 'j' and is_hexadecimal(ins_list[-1]):
                res_block_asm_list.append(f'{opcode} jump_addr')
                continue
            # step2 parse function call (stc_link, dyn_link, func)
            if 'call' in opcode:
                call_addr = ins_list[1]
                if call_addr not in func_dict:
                    callee_func_token = 'subxx'
                elif func_dict[call_addr]['name'] in dyn_func_list:
                    callee_func_token = 'outter_func_call'
                else:
                    callee_func_token = 'inner_func_call'
                res_block_asm_list.append(f'{opcode} {callee_func_token}')
                continue
            # step3 parse const into specific token
            tmp_ins_str = ins_str
            re_const_hex = r'(0x[0-9a-fA-F]+)'
            match_const_hex = re.findall(re_const_hex, tmp_ins_str, re.I)
            if match_const_hex:
                hex_regex = re.compile(re_const_hex)
                tmp_ins_str = hex_regex.sub('const_hex', tmp_ins_str)
            re_const_dec = r'\b(?!r[0-9])\d+\b'
            match_const_dec = re.findall(re_const_dec, tmp_ins_str, re.I)
            if match_const_dec:
                dec_regex = re.compile(re_const_dec)
                tmp_ins_str = dec_regex.sub('const_dec', tmp_ins_str)
            res_block_asm_list.append(tmp_ins_str)
        res_dict[int(block_addr)] = res_block_asm_list
    return res_dict


def process_asm_arm(basic_blocks, func_dict, dyn_func_list, func_name):
    res_dict = dict()
    for block_addr, block_data in basic_blocks.items():
        block_asm_list = block_data['bb_disasm']
        res_block_asm_list = []
        for ins_str in block_asm_list:
            ins_str = tokenize_bracket_comma(ins_str)
            ins_str = ins_str.replace('#', '')
            ins_list = ins_str.split()
            opcode = ins_list[0]
            # step1 parse brach instruction, in case  jump addr tokenized
            if opcode[0] == 'b' and opcode != 'bl'and is_hexadecimal(ins_list[-1]):
                res_block_asm_list.append(f'{opcode} jump_addr')
                continue
            # step1 parse function call (stc_link, dyn_link, func)
            if opcode == 'bl':
                call_addr = ins_list[1]
                if call_addr not in func_dict:
                    callee_func_token = 'subxxx'
                elif func_dict[call_addr]['name'] in dyn_func_list:
                    callee_func_token = 'outter_func_call'
                else:
                    callee_func_token = 'inner_func_call'
                res_block_asm_list.append(f'{opcode} {callee_func_token}')
                continue
            # step2 parse const into specific token
            tmp_ins_str = ins_str
            re_const_hex = r'(0x[0-9a-fA-F]+)'
            match_const_hex = re.findall(re_const_hex, tmp_ins_str, re.I)
            if match_const_hex:
                hex_regex = re.compile(re_const_hex)
                tmp_ins_str = hex_regex.sub('const_hex', tmp_ins_str)
            re_const_dec = r'\b(?!r[0-9])\d+\b'
            match_const_dec = re.findall(re_const_dec, tmp_ins_str, re.I)
            if match_const_dec:
                dec_regex = re.compile(re_const_dec)
                tmp_ins_str = dec_regex.sub('const_dec', tmp_ins_str)
            res_block_asm_list.append(tmp_ins_str)
        res_dict[int(block_addr)] = res_block_asm_list
    return res_dict


def process_asm_mips(basic_blocks, func_dict, dyn_func_list, func_name):
    res_dict = dict()
    for block_addr, block_data in basic_blocks.items():
        block_asm_list = block_data['bb_disasm']
        res_block_asm_list = []
        for ins_str in block_asm_list:
            ins_str = tokenize_bracket_comma(ins_str)
            ins_list = ins_str.split()
            opcode = ins_list[0]
            # step1 parse brach instruction, for mips, some brach instructions have more than one oprand
            if ((opcode[0] == 'b' and opcode != 'bal') or (opcode == 'j')) and is_hexadecimal(ins_list[-1]):
                new_ins_list = [i for i in ins_list]
                new_ins_list[-1] = 'jump_addr'
                res_block_asm_list.append(' '.join(new_ins_list))
                continue
            # step2 parse function call (stc_link, dyn_link, func)
            if opcode == 'jal' or opcode == 'bal':
                call_addr = ins_list[1]
                # [notion]sometimes bal is used for branch
                if call_addr not in func_dict:
                    callee_func_token = 'subxxx'
                    if int(call_addr, base=16) in basic_blocks:
                        callee_func_token = 'jump_addr'
                elif func_dict[call_addr]['name'] in dyn_func_list:
                    callee_func_token = 'outter_func_call'
                else:
                    callee_func_token = 'inner_func_call'
                res_block_asm_list.append(f'{opcode} {callee_func_token}')
                continue
            # step3 parse const into specific token
            tmp_ins_str = ins_str
            re_const_hex = r'(0x[0-9a-fA-F]+)'
            match_const_hex = re.findall(re_const_hex, tmp_ins_str, re.I)
            if match_const_hex:
                hex_regex = re.compile(re_const_hex)
                tmp_ins_str = hex_regex.sub('const_hex', tmp_ins_str)
            re_const_dec = r'\b(?!v[01]|a[0-3]|t[0-9]|s[0-8]|k[01])\d+\b'
            match_const_dec = re.findall(re_const_dec, tmp_ins_str, re.I)
            if match_const_dec:
                dec_regex = re.compile(re_const_dec)
                tmp_ins_str = dec_regex.sub('const_dec', tmp_ins_str)
            res_block_asm_list.append(tmp_ins_str)
        res_dict[int(block_addr)] = res_block_asm_list
    return res_dict


def process_asm(arch, func_dict, dyn_func_list, binary_name):
    func_map = dict()
    for func_addr, func_data in func_dict.items():
        func_name = func_data['name']
        edge_list = func_data['edges']
        node_list = func_data['nodes']
        basic_blocks = func_data['basic_blocks']

        if func_name in dyn_func_list:
            continue
        
        # step1 parse asm ins to token type
        if 'x86' in arch:
            asm_dict = process_asm_x86(basic_blocks, func_dict, dyn_func_list, func_name)
        elif 'arm' in arch:
            asm_dict = process_asm_arm(basic_blocks, func_dict, dyn_func_list, func_name)
        elif 'mips' in arch:
           asm_dict = process_asm_mips(basic_blocks, func_dict, dyn_func_list, func_name)
        else:
            print(f'[error] unknown arch: {arch}')
            return
        
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
            new_node_list.append(' '.join(asm_dict[node]))
            tmp_node_id_dict[node] = idx
        # use index of new node list present edge
        for edge in edge_list:
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


def process_and_gather_data(data_dir, outpur_dir):
    # get all file dict {proj_bin:[(file_name, arch_opt)]}
    proj_bin_dict, total_len = gather_pkl_file_name(data_dir)

    progress_bar = tqdm(range(total_len))

    res_list = []
    save_path = os.path.join(outpur_dir, 'finetune_triple_list.pkl')

    for proj_bin, file_tuple_list in proj_bin_dict.items():
        # save_file_name = os.path.join(outpur_dir, f'{proj_bin}_index.pkl')
        
        proj_bin_func_set = set()   # save common function name
        proj_bin_opt_dict = dict()  # save {func_name:{opt1:{info}, opt2:{info}}}

        # traverse every arch_opt
        for file_tuple in file_tuple_list:
            file_name = file_tuple[0]
            arch_opt = file_tuple[1]

            binary_name = '_'.join(file_name[:-4].split('_')[:-1])

            file_path = os.path.join(data_dir, file_name)
            pickle_data = load_pickle(file_path)
            func_dict = pickle_data[binary_name]['func_dict']
            arch = pickle_data[binary_name]['arch']
            dyn_func_list = pickle_data[binary_name]['dyn_func_list']

            func_map = process_asm(arch, func_dict, dyn_func_list, binary_name)
            pickle_data[binary_name]['func_map'] = func_map

            for func_name, func_addr in func_map.items():
                opt_map = {
                    'edges': func_dict[func_addr]['edges'],
                    'nodes': func_dict[func_addr]['nodes'],
                    'arch': get_arch_emb(arch_opt)
                }
                # if 'mips' in arch_opt and func_name[-2:] == '_0':
                #     func_name = func_name[:-2]
                if func_name not in proj_bin_opt_dict:
                    proj_bin_opt_dict[func_name] = {arch_opt: opt_map}
                else:    
                    proj_bin_opt_dict[func_name][arch_opt] = opt_map

            if len(proj_bin_func_set) == 0:
                proj_bin_func_set.update(func_map.keys())
            else:
                proj_bin_func_set &= set(func_map.keys())
            
            progress_bar.update(1)

        # generate triple dataset {[target, sim, dis-sim], [], ...}
        for func_name, func_opt_dict in proj_bin_opt_dict.items():
            if len(func_opt_dict) < 2:
                continue

            sample_num = math.ceil(len(func_opt_dict)/3)
            target_opt_list = random.sample(list(func_opt_dict.keys()), sample_num)

            for target_opt in target_opt_list:
                # get most unsim opt from all opt as sim sample
                sim_func_opt = get_rand_unsim_opt(target_opt, list(func_opt_dict.keys()))
                dis_sim_func_name = random.choice(list(proj_bin_opt_dict.keys()))
                loop_time = 0
                # get the another func with same opt as dis-sim sample
                while loop_time < MAX_LOOP_TIME :
                    if dis_sim_func_name != func_name and target_opt in proj_bin_opt_dict[dis_sim_func_name].keys():
                        tmp_target_data = func_opt_dict[target_opt]
                        tmp_dis_sim_data = proj_bin_opt_dict[dis_sim_func_name][target_opt]
                        if len(tmp_target_data['nodes']) == 1 and len(tmp_dis_sim_data['nodes']) == 1 and tmp_target_data['nodes'][0] == tmp_dis_sim_data['nodes'][0]:
                            dis_sim_func_name = random.choice(list(proj_bin_opt_dict.keys()))
                            loop_time += 1
                            continue
                        break
                    dis_sim_func_name = random.choice(list(proj_bin_opt_dict.keys()))
                    loop_time += 1
                # if search many times no result, emit this sample
                if loop_time >= MAX_LOOP_TIME:
                    print(f'[!]search unsim func loop time over {MAX_LOOP_TIME}')
                    continue
                
                target_data = func_opt_dict[target_opt]
                sim_data = func_opt_dict[sim_func_opt]
                dis_sim_data = proj_bin_opt_dict[dis_sim_func_name][target_opt]

                if len(target_data['nodes']) == 1 and len(dis_sim_data['nodes']) == 1:
                    if target_data['nodes'][0] == dis_sim_data['nodes'][0]:
                        continue

                res_list.append([target_data, sim_data, dis_sim_data])

        # with open(save_file_name, 'wb') as f:
        #     pickle.dump(proj_bin_opt_dict, f)

    with open(save_path, 'wb') as f:
        pickle.dump(res_list, f)                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gather project, bin_file, functions & generate finetune data.")
    parser.add_argument("--input_path", type=str, default='/home/liu/bcsd/train_set_extract_v2')
    parser.add_argument("--output_path", type=str, default='/home/liu/bcsd/datasets/edge_gnn_datas/')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # test use
    input_path = '/home/liu/project/ida_script/extract'
    output_path = './data'

    process_and_gather_data(input_path, output_path)

    # data = load_pickle('./data/finetune_triple_list.pkl')

    print('done')

