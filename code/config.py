import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129



DATA_DIR_yanbao = '/data1/lhz/medical_bert/yanbao/medical_dialogue'

TRAIN_FILE_yanbao = 'train_doctor_for_CSTE.txt'
TEST_FILE_yanbao = 'test_doctor_for_CSTE.txt'
TRAIN_FILE_yanbao_summary_75 = 'train_patient_for_CSTE.txt'
TEST_FILE_yanbao_summary_75 = 'test_patient_for_CSTE.txt'



# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
# SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'

# glove_cache_dir = "/data0/wxy/text classification/normal_deep_learning/glove_save"


class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.roberta_cache_path_Chinese = '/data1/lhz/PLM/diagBERT'
        self.feat_dim = 768
        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.embedding_size = 300#256
        self.pieces_size = 1

        self.split_len = 300#125
        self.overlap_len =0#25

        self.epochs = 10
        self.lr = 3e-5 
        self.other_lr = 2e-4
        self.batch_size = 16
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.05 #0.1
        self.adam_epsilon = 1e-8
        self.bert_output_dim = 768
        self.num_classes = 14 #
        self.max_length = 300#
        self.max_length_summary = 300
        self.lstm_hidden_dim = 512

        self.focal_loss_gamma = 2.
        self.focal_loss_alpha = None

        self.save_path = "/data1/lhz/medical_bert/compare_with_llm/CSTE_triage_vs_llm.ckpt"
        self.log_save_path = "/data1/lhz/medical_bert/compare_with_llm/CSTE_triage_vs_llm.txt"

        # Normal part
        self.normal_batch_size = 32
        self.LSTM_hidden_size = 256
        self.RNN_hidden_dim = 256
        self.GloVe_embedding_length = 300
        self.lr_NN = 0.001

