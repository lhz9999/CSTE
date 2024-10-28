from transformers import BertTokenizer, RobertaTokenizer
from config import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from os.path import join
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
import torchtext.vocab as vocab
from torchtext.data import get_tokenizer

# from matplotlib import pyplot as plt


def build_train_data(configs):
    # train_dataset = yanbao_Dataset(configs, data_type='train')
    # train_dataset = yanbao_yanbao_BERT_summary_DatasetBERT_Dataset(configs, data_type='train')
    train_dataset = yanbao_BERT_summary_Dataset(configs, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset.get_data(), batch_size=configs.batch_size,
                                               shuffle=True)
    return train_loader


def build_inference_data(configs, data_type):
    # dataset = yanbao_Dataset(configs, data_type)
    # dataset = yanbao_BERT_Dataset(configs, data_type)
    dataset = yanbao_BERT_summary_Dataset(configs, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset.get_data(), batch_size=configs.batch_size,
                                              shuffle=False)
    return data_loader


def tokenlize(content):
    content = re.sub('<.*?>', ' ', content)

    fileters = ['\/', '\(', '\)', ':', '\.', '\t', '\n', '\x97', '\x96', '#', '$', '%', '&']
    content = re.sub('|'.join(fileters), ' ', content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


def transfer_to_coarse(finegrained_prob):
    corse_prob = []
    for p in finegrained_prob:
        corse_prob.append([p[0] + p[2] + p[4], p[1] + p[3] + p[5]])

    return np.array(corse_prob)


def transfer_to_onehot(finegrained_prob):
    corse_prob = []
    for p in finegrained_prob:
        corse_prob.append([1 - int(p), int(p)])

    return np.array(corse_prob)


# 数据集加载方式，总共使用N个BERT对其进行分类
class yanbao_BERT_summary_Dataset(Dataset):

    def __init__(self, configs, data_type, data_dir=DATA_DIR_yanbao):
        self.data_dir = data_dir
        self.data_type = data_type

        self.train_file = join(data_dir, TRAIN_FILE_yanbao)
        self.test_file = join(data_dir, TEST_FILE_yanbao)
        self.train_summary_file = join(data_dir, TRAIN_FILE_yanbao_summary_75)
        self.test_summary_file = join(data_dir, TEST_FILE_yanbao_summary_75)
        
        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.embedding_size = configs.embedding_size
        self.pieces_size = configs.pieces_size
        self.split_len = configs.split_len
        self.overlap_len = configs.overlap_len

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.roberta_cache_path_Chinese)
        self.max_length = configs.max_length
        self.max_length_summary = configs.max_length_summary

    def get_split_text(self, text):
        split_text = []
        text.strip()
        text = [one for one in text]
        # text = text.split(' ')
        print(len(text))
        window = self.split_len - self.overlap_len
        length = max(math.ceil((len(text) - self.overlap_len) / window), 1)  # 防止有小于overlap_len长度的句子导致length为0
        for w in range(length):
            text_piece_word = text[w * window: w * window + self.split_len]
            # text_piece = ' '.join(text_piece_word)
            text_piece = ''.join(text_piece_word)
            split_text.append(text_piece)
        return split_text, length

    def get_length_matrix(self, length, size):
        matrix = torch.zeros([size, self.embedding_size, self.pieces_size])
        for i in range(size):
            if length[i] > self.pieces_size:
                lens = self.pieces_size
            else:
                lens = length[i]
            for j in range(self.embedding_size):
                for k in range(lens):
                    matrix[i][j][k] = 1
        return matrix

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
            data_file_summary = self.train_summary_file
        elif data_type == 'test':
            data_file = self.test_file
            data_file_summary = self.test_summary_file

        contents = []
        length = []
        labels = []

        with open(data_file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('@_@')
                split, lens = self.get_split_text(content)
                # print(split)
                contents.append(split)
                length.append(lens)
                labels.append(int(label))
        c_labels = np.array(labels)

        summaries = []
        with open(data_file_summary, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                summary, label = lin.split('@_@')
                summaries.append(summary)

        self.text_len = len(length)

        text_new = []
        print(len(contents))
        for i in range(len(contents)):
            line = contents[i] + ['padding'] * max(0, self.pieces_size - len(contents[i]))
            text_new.append(line)

        return text_new, c_labels, length, summaries

    def get_data(self):

        text, labels, length, summaries = self.read_data_file(self.data_type)

        tokenizer_summary = self.bert_tokenizer(
            summaries,
            padding=True,
            truncation=True,
            max_length=self.max_length_summary,
            return_tensors='pt'
        )
        input_ids_summary = tokenizer_summary['input_ids']
        token_type_ids_summary = tokenizer_summary['token_type_ids']
        attention_mask_summary = tokenizer_summary['attention_mask']
        zero = torch.zeros([self.text_len, self.embedding_size - input_ids_summary.shape[1]])
        zero = zero.long()
        input_ids_summary = torch.cat((input_ids_summary, zero), 1)
        token_type_ids_summary = torch.cat((token_type_ids_summary, zero), 1)
        attention_mask_summary = torch.cat((attention_mask_summary, zero), 1)

        text1 = [[row[i] for row in text] for i in range(self.pieces_size)]
        print(text1)
        for i in range(self.pieces_size):
            tokenizer = self.bert_tokenizer(
                text1[i],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenizer['input_ids']
            token_type_ids = tokenizer['token_type_ids']
            attention_mask = tokenizer['attention_mask']
            zero = torch.zeros([self.text_len, self.embedding_size - input_ids.shape[1]])
            zero = zero.long()
            one = torch.ones([self.text_len, self.embedding_size - input_ids.shape[1]])
            input_ids = torch.cat((input_ids, zero), 1)
            # print(input_ids)
            token_type_ids = torch.cat((token_type_ids, zero), 1)
            attention_mask = torch.cat((attention_mask, zero), 1)
            # print(token_type_ids)
            # print(attention_mask)
            # print(input_ids.shape)
            input_ids = input_ids.view([self.text_len, self.embedding_size, 1])
            token_type_ids = token_type_ids.view([self.text_len, self.embedding_size, 1])
            attention_mask = attention_mask.view([self.text_len, self.embedding_size, 1])

            if i != 0:
                input_ids = torch.cat((input_ids_old, input_ids), 2)
                token_type_ids = torch.cat((token_type_ids_old, token_type_ids), 2)
                attention_mask = torch.cat((attention_mask_old, attention_mask), 2)
            input_ids_old = input_ids
            token_type_ids_old = token_type_ids
            attention_mask_old = attention_mask

        labels = torch.tensor(labels)
        length = torch.tensor(length)
        length = self.get_length_matrix(length, self.text_len)
        print(labels.shape)
        print(length.shape)
        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels, length, input_ids_summary,
                             token_type_ids_summary, attention_mask_summary)

        return data