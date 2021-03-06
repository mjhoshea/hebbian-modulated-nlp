import os
import random
import re

import torch
import numpy as np
from sklearn.cluster import KMeans

from src.datasets.text_classification_dataset import AGNewsDataset, AmazonDataset, YelpDataset, DBPediaDataset, \
    YahooAnswersDataset


def batch_encode(batch):
    text, labels = [], []
    for txt, lbl in batch:
        text.append(txt)
        labels.append(lbl)
    return text, labels


def rel_encode(batch):
    text, label, candidate_relations = [], [], []
    for txt, lbl, cand in batch:
        text.append(txt)
        label.append(lbl)
        candidate_relations.append(cand)
    return text, label, candidate_relations


def get_dataset(base_path, dataset_id):
    if dataset_id == 0:
        train_path = os.path.join(base_path, '../data/ag_news_csv/train.csv')
        test_path = os.path.join(base_path, '../data/ag_news_csv/test.csv')
        train_dataset = AGNewsDataset(train_path, 'train', reduce=True)
        test_dataset = AGNewsDataset(test_path, 'test', reduce=True)
    elif dataset_id == 1:
        train_path = os.path.join(base_path, '../data/amazon_review_full_csv/train.csv')
        test_path = os.path.join(base_path, '../data/amazon_review_full_csv/test.csv')
        train_dataset = AmazonDataset(train_path, 'train', reduce=True)
        test_dataset = AmazonDataset(test_path, 'test', reduce=True)
    elif dataset_id == 2:
        train_path = os.path.join(base_path, '../data/yelp_review_full_csv/train.csv')
        test_path = os.path.join(base_path, '../data/yelp_review_full_csv/test.csv')
        train_dataset = YelpDataset(train_path, 'train', reduce=True)
        test_dataset = YelpDataset(test_path, 'test', reduce=True)
    elif dataset_id == 3:
        train_path = os.path.join(base_path, '../data/dbpedia_csv/train.csv')
        test_path = os.path.join(base_path, '../data/dbpedia_csv/test.csv')
        train_dataset = DBPediaDataset(train_path, 'train', reduce=True)
        test_dataset = DBPediaDataset(test_path, 'test', reduce=True)
    elif dataset_id == 4:
        train_path = os.path.join(base_path, '../data/yahoo_answers_csv/train.csv')
        test_path = os.path.join(base_path, '../data/yahoo_answers_csv/test.csv')
        train_dataset = YahooAnswersDataset(train_path, 'train', reduce=True)
        test_dataset = YahooAnswersDataset(test_path, 'test', reduce=True)
    else:
        raise Exception('Invalid dataset ID')
    return train_dataset, test_dataset


def offset_labels(dataset):
    # if isinstance(dataset, AmazonDataset) or isinstance(dataset, YelpDataset):
    #     offset_by = 0
    # el
    if isinstance(dataset, AGNewsDataset):
        offset_by = 0
    elif isinstance(dataset, DBPediaDataset):
        offset_by = 4
    # elif isinstance(dataset, YahooAnswersDataset):
    #     offset_by = 5 + 4 + 14
    dataset.data['labels'] = dataset.data['labels'] + offset_by
    return dataset


def remove_return_sym(str):
    return str.split('\n')[0]


def get_max_len(text_list):
    return max([len(x) for x in text_list])


def glove_vectorize(text, glove, dim=300):
    max_len = get_max_len(text)
    lengths = []
    vec = torch.ones((len(text), max_len, dim))
    for i, sent in enumerate(text):
        sent_emb = glove.get_vecs_by_tokens(sent, lower_case_backup=True)
        vec[i, :len(sent_emb)] = sent_emb
        lengths.append(len(sent))
    lengths = torch.tensor(lengths)
    return vec, lengths
