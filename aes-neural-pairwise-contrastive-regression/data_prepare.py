# -*- coding: utf-8 -*-

import reader
import utils
import numpy as np
import torch
import math
import reader
from selectpair import *

logger = utils.get_logger("Prepare data ...")


def prepare_sentence_data(datapaths, prompt_id=1):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y), max_num = \
        reader.get_data(datapaths, prompt_id)

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, max_num, post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, max_num, post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, max_num, post_padding=True)

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)
    Y_train = y_train
    Y_dev = y_dev
    Y_test = y_test

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))
    # logger.info('  prompt shape: ' + str(prompt.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    return (X_train, Y_train, mask_train), (X_dev, Y_dev, mask_dev), (X_test, Y_test, mask_test), max_num


def data_shuffle(features):
    queries_test = []
    for i in range(len(features)):
        queries = np.random.choice(len(features[i][0]), len(features[i][0]), replace=False)
        queries_x = []
        queries_y = []
        for loc in queries:
            queries_x.append(features[i][0][loc])
            queries_y.append(features[i][1][loc])
        queries_y = np.squeeze(np.broadcast_arrays(queries_y))
        queries_test.append((queries_x, queries_y))

    return queries_test


# NPCR
def data_pre_opti(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, prompt_id, example_size):
    features_train = []
    y_train = []

    # C2 performs better than C1
    for i in range(X_train.shape[0] - 6):
        # C2
        if Y_train[i] != Y_train[i + 1]:
            y_train.append(Y_train[i] - Y_train[i + 1])
            features_train.append((X_train[i], X_train[i + 1]))
        if Y_train[i] != Y_train[i + 2]:
            y_train.append(Y_train[i] - Y_train[i + 2])
            features_train.append((X_train[i], X_train[i + 2]))
        if Y_train[i] != Y_train[i + 3]:
            y_train.append(Y_train[i] - Y_train[i + 3])
            features_train.append((X_train[i], X_train[i + 3]))
        # C1
        # y_train.append(Y_train[i] - Y_train[i + 1])
        # features_train.append((X_train[i], X_train[i + 1]))
        # y_train.append(Y_train[i] - Y_train[i + 2])
        # features_train.append((X_train[i], X_train[i + 2]))
        # y_train.append(Y_train[i] - Y_train[i + 3])
        # features_train.append((X_train[i], X_train[i + 3]))
    features_train = np.array(features_train)
    y_train = np.array(y_train)
    y_train = reader.get_model_friendly_scores(y_train, prompt_id)

    train_x0 = [j[0] for j in features_train]
    train_x1 = [j[1] for j in features_train]

    train_x0 = torch.LongTensor(train_x0)
    train_x1 = torch.LongTensor(train_x1)

    # dev
    features_dev = []
    dev_y_example = []
    dev_y_goal = []

    maxTrainLen = X_train.shape[0] - 1
    maxR = X_train.shape[0] - 1
    minL = 0
    for i in range(X_dev.shape[0]):
        num = example_size
        j = 0
        while num > 0:
            if Y_train[j] != Y_dev[i]:
                num -= 1
                features_dev.append((X_dev[i], X_train[j]))
                dev_y_example.append(Y_train[j])
            j += 1

        dev_y_goal.append(Y_dev[i])
    features_dev = np.array(features_dev)
    dev_y_example = np.array(dev_y_example)
    dev_y_goal = np.array(dev_y_goal)

    # test
    features_test = []
    test_y_example = []
    test_y_goal = []

    maxR = X_train.shape[0] - 1
    minL = 0
    for i in range(X_test.shape[0]):
        num = example_size
        j = 0
        while num > 0:
            if Y_train[j] != Y_test[i]:
                num -= 1
                features_test.append((X_test[i], X_train[j]))
                test_y_example.append(Y_train[j])
            j += 1

        test_y_goal.append(Y_test[i])
    features_test = np.array(features_test)
    test_y_example = np.array(test_y_example)
    test_y_goal = np.array(test_y_goal)

    return train_x0, train_x1, y_train, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal


# NPCR-Group
def data_pre_group(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, prompt_id, example_size):
    features_train = []
    y_train = []

    # C2 performs better than C1
    for i in range(X_train.shape[0] - 6):
        # C2
        # if Y_train[i] != Y_train[i + 1]:
        #     y_train.append(Y_train[i] - Y_train[i + 1])
        #     features_train.append((X_train[i], X_train[i + 1]))
        # if Y_train[i] != Y_train[i + 2]:
        #     y_train.append(Y_train[i] - Y_train[i + 2])
        #     features_train.append((X_train[i], X_train[i + 2]))
        # if Y_train[i] != Y_train[i + 3]:
        #     y_train.append(Y_train[i] - Y_train[i + 3])
        #     features_train.append((X_train[i], X_train[i + 3]))
        # C1
        y_train.append(Y_train[i] - Y_train[i + 1])
        features_train.append((X_train[i], X_train[i + 1]))
        y_train.append(Y_train[i] - Y_train[i + 2])
        features_train.append((X_train[i], X_train[i + 2]))
        y_train.append(Y_train[i] - Y_train[i + 3])
        features_train.append((X_train[i], X_train[i + 3]))
    features_train = np.array(features_train)
    y_train = np.array(y_train)
    y_train = reader.get_model_friendly_scores(y_train, prompt_id)

    train_x0 = [j[0] for j in features_train]
    train_x1 = [j[1] for j in features_train]

    train_x0 = torch.LongTensor(train_x0)
    train_x1 = torch.LongTensor(train_x1)

    essay_nums, score_groups_num, scores, score_ids = score_groups_divide(X_train, Y_train)
    slot_size = int(example_size / score_groups_num)
    extra_size = int(example_size % score_groups_num)

    # dev
    features_dev = []
    dev_y_example = []
    dev_y_goal = []

    for k in range(X_dev.shape[0]):
        # all slots
        for i in range(score_groups_num):
            cur_score = scores[i]
            cur_score_ids = score_ids[cur_score]
            for j in range(slot_size):
                features_dev.append((X_dev[k], X_train[cur_score_ids[j]]))
                dev_y_example.append(Y_train[cur_score_ids[j]])
        # extra
        for t in range(extra_size):
            cur_score = scores[t]
            cur_score_ids = score_ids[cur_score]
            features_dev.append((X_dev[k], X_train[cur_score_ids[0]]))
            dev_y_example.append(Y_train[cur_score_ids[0]])

        dev_y_goal.append(Y_dev[k])

    # test
    features_test = []
    test_y_example = []
    test_y_goal = []

    for k in range(X_test.shape[0]):
        # all slots
        for i in range(score_groups_num):
            cur_score = scores[i]
            cur_score_ids = score_ids[cur_score]
            for j in range(slot_size):
                features_test.append((X_test[k], X_train[cur_score_ids[j]]))
                test_y_example.append(Y_train[cur_score_ids[j]])
        # extra
        for t in range(extra_size):
            cur_score = scores[t]
            cur_score_ids = score_ids[cur_score]
            features_test.append((X_test[k], X_train[cur_score_ids[0]]))
            test_y_example.append(Y_train[cur_score_ids[0]])

        test_y_goal.append(Y_test[k])

    return train_x0, train_x1, y_train, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal







