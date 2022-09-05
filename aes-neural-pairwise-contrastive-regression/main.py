# -*- coding: utf-8 -*-


import os
import sys
import argparse
from utils import *
import torch
import torch.nn as nn
import torch.utils.data as Data

from networks.core_networks import npcr_model

import data_prepare
from evaluator_core import Evaluator_opti
# from evaluator_optim import *
import random
import time
import numpy as np


logger = get_logger("Train...")
np.random.seed(100)


def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
    parser.add_argument('--train_flag', action='store_true', help='Train or eval')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
    parser.add_argument('--embedding', type=str, default='word2vec', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dict', type=str, default=None, help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Only useful when embedding is randomly initialised')
    parser.add_argument('--char_embedd_dim', type=int, default=30, help='char embedding dimension if using char embedding')

    parser.add_argument('--use_char', action='store_true', help='Whether use char embedding or not')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")

    parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')
    parser.add_argument('--char_nbfilters', type=int, default=20, help='Num of char filters in conv layer')
    parser.add_argument('--filter1_len', type=int, default=5, help='filter length in 1st conv layer')
    parser.add_argument('--filter2_len', type=int, default=3, help='filter length in 2nd conv layer or char conv layer')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
    parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

    # parser.add_argument('--project_hiddensize', type=int, default=100, help='num of units in projection layer')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--oov', choices=['random', 'embedding'], help="Embedding for oov word", required=True)
    parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory')
    parser.add_argument('--gpus',nargs = '+',type = str,help='gpus')
    parser.add_argument('--train')  # "data/word-level/*.train"
    parser.add_argument('--dev')
    parser.add_argument('--test')
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')
    parser.add_argument('--prompt')
    parser.add_argument('--init_bias', action='store_true', help='init the last layer bias with average score of training data')
    parser.add_argument('--mode', type=str, choices=['mot', 'att', 'merged'], default='mot', \
                        help='Mean-over-Time pooling or attention-pooling, or two pooling merged')
    parser.add_argument('--example_size', type=int, default=50, help='select example essay number')
    parser.add_argument('--npcr_group', action='store_true', help='use stragety of npcr-group if true')

    args = parser.parse_args()
    train_flag = args.train_flag
    fine_tune = args.fine_tune
    USE_CHAR = args.use_char
    example_size = args.example_size
    learning_rate = args.learning_rate

    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_path
    num_epochs = args.num_epochs

    modelname = "rank_core_bert.prompt%s.pt" % args.prompt_id
    imgname = "sent_hiclstm-%s.prompt%s.%sfilters.bs%s.png" % (args.mode, args.prompt_id, args.nbfilters, batch_size)

    if USE_CHAR:
        modelname = 'char_' + modelname
        imgname = 'char_' + imgname
    modelpath = os.path.join(checkpoint_dir, modelname)
    imgpath = os.path.join(checkpoint_dir, imgname)

    datapaths = [args.train, args.dev, args.test]
    prompt_path = args.prompt
    embedding_path = args.embedding_dict
    oov = args.oov
    embedding = args.embedding
    embedd_dim = args.embedding_dim
    prompt_id = args.prompt_id

    (X_train,Y_train,mask_train),(X_dev,Y_dev,mask_dev),(X_test,Y_test,mask_test),max_num = \
        data_prepare.prepare_sentence_data(datapaths,prompt_id)

    # train_x0, train_x1, train_y, dev_x,dev_y, dev_yy,test_x,test_y,test_yy = data_prepare.data_pre_con(X_train,
    # Y_train, X_dev, Y_dev, X_test, Y_test,prompt_id)
    if not args.npcr_group:
        train_x0, train_x1, train_y, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal = data_prepare.data_pre_opti(X_train, Y_train, X_dev, Y_dev, X_test, Y_test,prompt_id, example_size)
    else:
        train_x0, train_x1, train_y, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal = data_prepare.data_pre_group(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, prompt_id, example_size)

    logger.info("----------------------------------------------------")

    C_train, C_dev, C_test = None, None, None
    char_vocabsize = 0
    maxcharlen = 0
    init_mean_value = None
    model = npcr_model(512)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    evl = Evaluator_opti(args.prompt_id, args.use_char, checkpoint_dir, modelname, features_dev, dev_y_example, dev_y_goal, features_test, test_y_example, test_y_goal, example_size)

    logger.info("Train model")

    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW
    # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    torch_dataset = Data.TensorDataset(train_x0, train_x1, torch.Tensor(train_y))

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    for ii in range(args.num_epochs):
        logger.info('Epoch %s/%s' % (str(ii), args.num_epochs))
        start_time = time.time()
        for step, (batch_x0, batch_x1, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            Y_predict = model(batch_x0.cuda(), batch_x1.cuda())
            loss = loss_fn(Y_predict.squeeze(), batch_y.squeeze().cuda())
            print('epoch:', ii, 'step:', step, 'loss:', loss.item())
            loss.backward()
            optimizer.step()

        tt_time = time.time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)

        model.eval()
        with torch.no_grad():
            evl.evaluate(model, ii, True)
        model.train()

        ttt_time = time.time() - start_time - tt_time
        logger.info("Evaluate one time in %.3f s" % ttt_time)

    evl.print_final_info()


if __name__ == '__main__':
    main()

