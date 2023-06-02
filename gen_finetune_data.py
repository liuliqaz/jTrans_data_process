from unicodedata import name
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import torch.multiprocessing
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data import load_paired_data, FunctionDataset_CL_Load
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


if __name__ == '__main__':
    train_path = './datautils/extract'
    tokenizer = './jtrans_tokenizer'
    load_train, load_test = False, False
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    ft_train_dataset = FunctionDataset_CL_Load(tokenizer, train_path, convert_jump_addr=True, load=load_train, opt=['O0', 'O1', 'O2', 'O3', 'Os'])

    for i in ft_train_dataset:
        print(i)
        break
