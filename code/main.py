import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
# import torch
# from config import *
from dataloader import *
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from model import *

from scipy.stats import pearsonr
import time
from datetime import timedelta
from torchinfo import summary

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def pearsonr_correlation_coefficient(pre, rea):
    if len(pre) != len(rea):
        print('error!!!!')
    # print(len(pre))
    # print(pre)
    # print(len(rea))
    # print(rea)
    result = np.zeros(len(pre))
    for i in range(len(pre)):
        # print i
        result[i] = pearsonr(pre[i], rea[i])[0]
        # print(result[i])
        if result[i] != result[i]:
            result[i] = 0
    average_average = sum(result) / len(pre)
    return average_average


def run(configs):
    # initialize
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministics = True

    # load data
    train_loader = build_train_data(configs)
    # valid_loader = build_inference_data(configs, data_type='valid')
    test_loader = build_inference_data(configs, data_type='test')

    # load model

    model = new_dual_BERT_for_medical(configs)
    summary(model)

    # device_ids = [1,2]
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
      no_decay = ['bias', 'gamma', 'beta']
    params = model.parameters()
    optimizer = AdamW(params, lr=configs.lr)
        # 线性warmup然后线性衰减
    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)  # why warm up?
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_steps_all)
    model.zero_grad()
    early_stop_flag = None

    total_batch = 0
    start_time = time.time()
    dev_best_loss = float('inf')
    dev_best_acc = 0
    dev_best_f1 = 0
    dev_best_recall = 0
    dev_best_precision = 0
    dev_best_mcc = 0

    for epoch in range(configs.epochs):
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            train_features, train_token_type, train_bert_mask, train_preds, train_length, train_features_summary, train_token_type_summary, train_bert_mask_summary = batch


            train_features_summary = train_features_summary.to(DEVICE)
            train_token_type_summary = train_token_type_summary.to(DEVICE)
            train_bert_mask_summary = train_bert_mask_summary.to(DEVICE)


            train_features = train_features.to(DEVICE)
            train_token_type = train_token_type.to(DEVICE)
            train_bert_mask = train_bert_mask.to(DEVICE)
            train_preds = train_preds.to(DEVICE)
            train_length = train_length.to(DEVICE)


            output_preds = model(train_features, train_bert_mask, train_token_type, train_length, train_features_summary, train_bert_mask_summary, train_token_type_summary)

            # 交叉熵损失
            entroy = nn.CrossEntropyLoss()
            loss = entroy(output_preds, train_preds)
            
            loss.backward()

            # 修剪梯度进行归一化防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                output_preds = F.softmax(output_preds, dim=1)

                true = train_preds.data.cpu()
                predic = torch.max(output_preds.data, 1)[1].cpu()
                # print(predic)
                # print(true)

                train_acc = metrics.accuracy_score(true, predic)
                train_f1 = metrics.f1_score(true, predic, average='macro')
                train_precision = metrics.precision_score(true, predic, average='macro')
                train_recall = metrics.recall_score(true, predic, average='macro')

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {},  Train Loss: {},  Train Acc: {},  Train F1: {},  Train Precision: {},  Train Recall: {},  Time: {}'
                file = open(configs.log_save_path, 'a', encoding='utf-8')
                print(msg.format(total_batch, loss.item(), train_acc, train_f1, train_precision, train_recall,
                                 time_dif), file=file)
                file.close()
                print(msg.format(total_batch, loss.item(), train_acc, train_f1, train_precision, train_recall,
                                 time_dif))
            total_batch += 1

        with torch.no_grad():
            model.eval()

            dev_loss, dev_acc, dev_f1, dev_precision, dev_recall, dev_mcc, labels_all, predict_all = evaluate(
                configs, model, test_loader)
            if epoch == 0:
                a = np.array(labels_all)
                np.save('/data1/lhz/medical_bert/predict_result_save/labels_all_1.npy', a)
                # a = np.load('a.npy')
                # a = a.tolist()

            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(model.state_dict(), configs.save_path)
                improve = '*'
                last_improve = total_batch
            else:
                improve = ''

            if dev_acc > dev_best_acc:
                dev_best_acc = dev_acc
                b = np.array(predict_all)
                np.save('/data1/lhz/medical_bert/predict_result_save/predict_all_1.npy', b)

            if dev_recall > dev_best_recall:
                dev_best_recall = dev_recall

            if dev_precision > dev_best_precision:
                dev_best_precision = dev_precision

            if dev_f1 > dev_best_f1:
                dev_best_f1 = dev_f1

            if dev_mcc > dev_best_mcc:
                dev_best_mcc = dev_mcc

            time_dif = get_time_dif(start_time)
            msg = 'Iter: {},  Dev Loss: {},  Dev Acc: {},  Dev F1: {},  Dev Precision: {},  Dev Recall: {}, Dev MCC: {}, Time: {}{}'
            print(msg.format(epoch, dev_loss.item(), dev_acc, dev_f1, dev_precision, dev_recall, dev_mcc,
                             time_dif, improve))
            file = open(configs.log_save_path, 'a', encoding='utf-8')
            print(msg.format(epoch, dev_loss.item(), dev_acc, dev_f1, dev_precision, dev_recall, dev_mcc,
                             time_dif, improve), file=file)
            file.close()

    print('best dev recall:', dev_best_recall)
    print('best dev precision:', dev_best_precision)
    print('best dev f1:', dev_best_f1)
    print('best dev acc:', dev_best_acc)
    print('best dev mcc:', dev_best_mcc)
    file = open(configs.log_save_path, 'a', encoding='utf-8')
    print('best dev recall:', dev_best_recall, file=file)
    print('best dev precision:', dev_best_precision, file=file)
    print('best dev f1:', dev_best_f1, file=file)
    print('best dev acc:', dev_best_acc, file=file)
    print('best dev mcc:',dev_best_mcc, file=file)
    file.close()


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    f_predict_all = np.array([], dtype=int)
    f_labels_all = np.array([], dtype=int)
    predict_prob = []
    labels_prob = []
    f_predict_prob = []
    f_labels_prob = []
    with torch.no_grad():
        for features, token_type, bert_mask, labels, length, features_summary, token_type_summary, bert_mask_summary in data_iter:

            # plus1
            features_summary = features_summary.to(DEVICE)
            token_type_summary = token_type_summary.to(DEVICE)
            bert_mask_summary = bert_mask_summary.to(DEVICE)
            # plus1end

            features = features.to(DEVICE)
            token_type = token_type.to(DEVICE)
            bert_mask = bert_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            length = length.to(DEVICE)

            outputs = model(features, bert_mask, token_type, length, features_summary, bert_mask_summary, token_type_summary)

            entroy = nn.CrossEntropyLoss()
            loss = entroy(outputs, labels)

            loss_total += loss

            # coarse
            labels = labels.data.cpu().numpy()
            outputs = F.softmax(outputs, dim=1)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    precision = metrics.precision_score(labels_all, predict_all, average='macro')
    recall = metrics.recall_score(labels_all, predict_all, average='macro')
    mcc = metrics.matthews_corrcoef(labels_all, predict_all, sample_weight=None)

    return loss_total / len(data_iter), acc, f1, precision, recall, mcc, labels_all, predict_all


if __name__ == '__main__':
    configs = Config()

    run(configs)
