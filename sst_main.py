

import pytreebank
import logging
import random
import numpy as np
import torch
from alphabet import Alphabet
from embedding import initialize_emb
from transformer.Models import Transformer_classification
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import time
from my_utils import pad_sequence

def load_data(directory, len_max_seq):

    dataset = pytreebank.load_sst(directory)

    train_sentences = []
    dev_sentences = []
    test_sentence = []

    for labeled_tree_sentence in dataset['train']:
        label, sentence = labeled_tree_sentence.to_labeled_lines()[0]
        logging.debug("{} | {}".format(sentence, label))
        tokens = sentence.split()[:len_max_seq]
        sentence = {}
        sentence['label'] = label
        sentence['token'] = tokens
        train_sentences.append(sentence)


    for labeled_tree_sentence in dataset['dev']:
        label, sentence = labeled_tree_sentence.to_labeled_lines()[0]
        logging.debug("{} | {}".format(sentence, label))
        tokens = sentence.split()[:len_max_seq]
        sentence = {}
        sentence['label'] = label
        sentence['token'] = tokens
        dev_sentences.append(sentence)

    for labeled_tree_sentence in dataset['test']:
        label, sentence = labeled_tree_sentence.to_labeled_lines()[0]
        logging.debug("{} | {}".format(sentence, label))
        tokens = sentence.split()[:len_max_seq]
        sentence = {}
        sentence['label'] = label
        sentence['token'] = tokens
        test_sentence.append(sentence)

    return train_sentences, dev_sentences, test_sentence

def build_alphabet(dataset, alphabet_token, alphabet_label):
    for sentence in dataset:
        alphabet_label.add(sentence['label'])
        for token in sentence['token']:
            alphabet_token.add(token)

def prepare_instance(dataset, alphabet_token, alphabet_label):
    for sentence in dataset:
        token_index = []
        position_index = []
        for index, token in enumerate(sentence['token']):
            token_index.append(alphabet_token.get_index(token))
            position_index.append(index+1) # position start from 1
        label_index = alphabet_label.get_index(sentence['label'])

        sentence['token_index'] = token_index
        sentence['label_index'] = label_index
        sentence['position_index'] = position_index

class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def my_collate(x):
    tokens = [x_['token_index'] for x_ in x]
    positions = [x_['position_index'] for x_ in x]
    labels = [x_['label_index'] for x_ in x]
    lengths = [len(x_['token_index']) for x_ in x]
    max_len = max(lengths)

    tokens = pad_sequence(tokens, max_len)
    positions = pad_sequence(positions, max_len)
    labels = torch.LongTensor(labels)

    if opt.gpu >= 0 and torch.cuda.is_available():
        tokens = tokens.cuda(opt.gpu)
        positions = positions.cuda(opt.gpu)
        labels = labels.cuda(opt.gpu)

    return tokens, positions, labels


def evaluate(data_loader, model):
    with torch.no_grad():
        model.eval()

        total = 0
        correct = 0

        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        for i in range(num_iter):

            batch_word, batch_position, batch_label = next(data_iter)

            score = model.forward(batch_word, batch_position)

            _, indices = torch.max(score, 1)

            total += batch_label.size(0)
            correct += (indices == batch_label).sum().item()


    return 100.0 * correct / total


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True, type=str)
    parser.add_argument('-verbose', action='store_true')
    parser.add_argument('-random_seed', type=int, default=1)
    parser.add_argument('-word_emb', type=str, default=None)
    parser.add_argument('-word_emb_dim', type=int, default=50)
    parser.add_argument('-len_max_seq', type=int, default=256)
    parser.add_argument('-hidden_dim', type=int, default=300)
    parser.add_argument('-hidden_layer', type=int, default=6)
    parser.add_argument('-head_num', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-l2', type=float, default=1e-8)
    parser.add_argument('-tune_emb', action='store_true', default=False)
    parser.add_argument('-iter', type=int, default=100)
    parser.add_argument('-gpu', type=int, default=-1)
    parser.add_argument('-patience', type=int, default=20)

    opt = parser.parse_args()

    logger = logging.getLogger()
    if opt.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.info(opt)

    if opt.random_seed != 0:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed_all(opt.random_seed)

    train_sentences, dev_sentences, test_sentences = load_data(opt.data, opt.len_max_seq)
    logging.info("training sentences {}".format(len(train_sentences)))
    logging.info("dev sentences {}".format(len(dev_sentences)))
    logging.info("test sentences {}".format(len(test_sentences)))

    logging.info("building alphabet")
    alphabet_token = Alphabet('token')
    alphabet_label = Alphabet('label', True)
    # position doesn't need alphabet
    build_alphabet(train_sentences, alphabet_token, alphabet_label)
    build_alphabet(dev_sentences, alphabet_token, alphabet_label)
    build_alphabet(test_sentences, alphabet_token, alphabet_label)

    logging.info("prepare instance")
    prepare_instance(train_sentences, alphabet_token, alphabet_label)
    prepare_instance(dev_sentences, alphabet_token, alphabet_label)
    prepare_instance(test_sentences, alphabet_token, alphabet_label)

    logging.info("prepare embedding")
    word_embedding, real_emb_dim = initialize_emb(opt.word_emb, alphabet_token, opt.word_emb_dim)

    logging.info("create model")
    model = Transformer_classification(alphabet_token.size(), opt.len_max_seq, alphabet_label.size(),
            d_word_vec=real_emb_dim, d_model=opt.hidden_dim, d_inner=4*opt.hidden_dim,
            n_layers=opt.hidden_layer, n_head=opt.head_num, d_k=opt.hidden_dim//opt.head_num,
                                       d_v=opt.hidden_dim//opt.head_num, dropout=opt.dropout)

    model.init_emb(word_embedding)


    logging.info("prepare training")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    if opt.tune_emb == False:
        model.free_emb()

    train_loader = DataLoader(MyDataset(train_sentences), opt.batch_size, shuffle=True, collate_fn=my_collate)
    dev_loader = DataLoader(MyDataset(dev_sentences), opt.batch_size, shuffle=False, collate_fn=my_collate)
    test_loader = DataLoader(MyDataset(test_sentences), opt.batch_size, shuffle=False, collate_fn=my_collate)

    logging.info("start training ...")

    best_dev = -10
    best_test = -10
    bad_counter = 0
    for idx in range(opt.iter):
        epoch_start = time.time()

        model.train()

        train_iter = iter(train_loader)
        num_iter = len(train_loader)

        sum_loss = 0
        correct, total = 0, 0

        for i in range(num_iter):
            batch_word, batch_position, batch_label = next(train_iter)

            score = model.forward(batch_word, batch_position)

            loss, total_this_batch, correct_this_batch = model.loss_function(score, batch_label)

            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
            model.zero_grad()

            total += total_this_batch
            correct += correct_this_batch

        epoch_finish = time.time()
        accuracy = 100.0 * correct / total
        logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
            idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

        accuracy = evaluate(dev_loader, model)
        logging.info("Dev accuracy %.4f" % (accuracy))
        if accuracy > best_dev:
            logging.info("Exceed previous best performance on dev: %.4f" % (best_dev))
            best_dev = accuracy

        accuracy = evaluate(test_loader, model)
        logging.info("Test accuracy %.4f" % (accuracy))
        if accuracy > best_test:
            logging.info("Exceed previous best performance on test: %.4f" % (best_test))
            best_test = accuracy
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    logging.info("end ......")
