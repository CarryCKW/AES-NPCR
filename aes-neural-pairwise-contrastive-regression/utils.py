# -*- coding: utf-8 -*-

import gzip
import logging
import sys
import numpy as np

import torch.nn as nn


def get_logger(name, level=logging.INFO, handler=sys.stdout,
        formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# TimeDistributed for pytorch
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        input_shape = input_seq.shape

        X = input_seq.reshape((-1, ) + input_shape[2:])  # (nb_samples * timesteps, ...)
        y = self.module(X)  # (nb_samples * timesteps, ...)
        # (nb_samples, timesteps, ...)
        if type(y) == tuple: y = y[0]   # lstm
        y = y.reshape((-1, input_shape[1]) + y.shape[1:])
        return y


def padding_sentence_sequences(index_sequences, scores, maxnum, post_padding=True):

    X = np.empty([len(index_sequences), maxnum], dtype=np.int32)
    Y = np.empty([len(index_sequences), 1], dtype=np.float32)
    mask = np.zeros([len(index_sequences), maxnum], dtype=np.float32)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)
        if num > maxnum:
            for k in range(maxnum):
                wid = sequence_ids[k]
                X[i,k] = wid
        else:
            for k in range(num):
                wid = sequence_ids[k]
                X[i, k] = wid
            X[i, num:] = 1   # pad 1 for robert
            mask[i, :num] = 1

        Y[i] = scores[i]
    return X, Y, mask


def padding_sentences(index_sequences, max_sentnum, max_sentlen, post_padding=True):

    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)
    mask = np.zeros([len(index_sequences), max_sentnum, max_sentlen], dtype=np.float32)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            # X_len[i] = length
            for k in range(length):
                wid = word_ids[k]
                # print wid
                X[i, j, k] = wid

            # Zero out X after the end of the sequence
            X[i, j, length:] = 0
            # Make the mask for this sample 1 within the range of length
            mask[i, j, :length] = 1

        X[i, num:, :] = 0
    return X, mask


def padding_sequences(word_indices, char_indices, scores, max_sentnum, max_sentlen, maxcharlen, post_padding=True):
    # support char features
    X = np.empty([len(word_indices), max_sentnum, max_sentlen], dtype=np.int32)
    Y = np.empty([len(word_indices), 1], dtype=np.float32)
    mask = np.zeros([len(word_indices), max_sentnum, max_sentlen], dtype=np.float32)

    char_X = np.empty([len(char_indices), max_sentnum, max_sentlen, maxcharlen], dtype=np.int32)

    for i in range(len(word_indices)):
        sequence_ids = word_indices[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            # X_len[i] = length
            for k in range(length):
                wid = word_ids[k]
                # print wid
                X[i, j, k] = wid

            # Zero out X after the end of the sequence
            X[i, j, length:] = 0
            # Make the mask for this sample 1 within the range of length
            mask[i, j, :length] = 1

        X[i, num:, :] = 0
        Y[i] = scores[i]

    for i in range(len(char_indices)):
        sequence_ids = char_indices[i]
        num = len(sequence_ids)
        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                charlen = len(wid)
                for l in range(charlen):
                    cid = wid[l]
                    char_X[i, j, k, l] = cid
                char_X[i, j, k, charlen:] = 0
            char_X[i, j, length:, :] = 0
        char_X[i, num:, :] = 0
    return X, char_X, Y, mask


def rescale_tointscore(scaled_scores, set_ids):
    '''
    rescale scaled scores range[0,1] to original integer scores based on  their set_ids
    :param scaled_scores: list of scaled scores range [0,1] of essays
    :param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
    '''
    # print type(scaled_scores)
    # print scaled_scores[0:100]
    global maxscore, minscore
    if isinstance(set_ids, int):
        prompt_id = set_ids
        set_ids = np.ones(scaled_scores.shape[0],) * prompt_id
    assert scaled_scores.shape[0] == len(set_ids)
    int_scores = np.zeros((scaled_scores.shape[0], 1))
    for k, i in enumerate(set_ids):
        assert i in range(1, 9)
        if i == 1:
            minscore = -10
            maxscore = 10
        elif i == 2:
            minscore = -5
            maxscore = 5
        elif i in [3, 4]:
            minscore = -3
            maxscore = 3
        elif i in [5, 6]:
            minscore = -4
            maxscore = 4
        elif i == 7:
            minscore = -30
            maxscore = 30
        elif i == 8:
            minscore = -60
            maxscore = 60
        else:
            print("Set ID error")
        # minscore = 0
        # maxscore = 60

        int_scores[k] = scaled_scores[k]*(maxscore - minscore) + minscore

    # return np.around(int_scores).astype(int)
    return int_scores

def domain_specific_rescale(y_true, y_pred, set_ids):
    '''
    rescaled scores to original integer scores based on their set ids
    and partition the score list based on its specific prompot
    return 8-prompt int score list for y_true and y_pred respectively
    :param y_true: true score list, contains all 8 prompts
    :param y_pred: pred score list, also contains 8 prompts
    :param set_ids: list that indicates the set/prompt id for each essay
    '''
    # prompts_truescores = []
    # prompts_predscores = []
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y1_true, y1_pred = [], []
    y2_true, y2_pred = [], []
    y3_true, y3_pred = [], []
    y4_true, y4_pred = [], []
    y5_true, y5_pred = [], []
    y6_true, y6_pred = [], []
    y7_true, y7_pred = [], []
    y8_true, y8_pred = [], []

    for k, i in enumerate(set_ids):
        assert i in range(1, 9)
        if i == 1:
            minscore = -10
            maxscore = 10
            y1_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y1_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 2:
            minscore = -5
            maxscore = 5
            y2_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y2_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 3:
            minscore = -3
            maxscore = 3
            y3_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y3_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 4:
            minscore = -3
            maxscore = 3
            y4_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y4_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        elif i == 5:
            minscore = -4
            maxscore = 4
            y5_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y5_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
        elif i == 6:
            minscore = -4
            maxscore = 4
            y6_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y6_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        elif i == 7:
            minscore = -30
            maxscore = 30
            y7_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y7_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        elif i == 8:
            minscore = -60
            maxscore = 60
            y8_true.append(y_true[k]*(maxscore-minscore) + minscore)
            y8_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

        else:
            print("Set ID error")
    prompts_truescores = [np.around(y1_true), np.around(y2_true), np.around(y3_true), np.around(y4_true), \
                            np.around(y5_true), np.around(y6_true), np.around(y7_true), np.around(y8_true)]
    prompts_predscores = [np.around(y1_pred), np.around(y2_pred), np.around(y3_pred), np.around(y4_pred), \
                            np.around(y5_pred), np.around(y6_pred), np.around(y7_pred), np.around(y8_pred)]

    return prompts_truescores, prompts_predscores