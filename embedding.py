import logging
import numpy as np
import re
import codecs
import torch

def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0
        # temp = str()
        # temp = temp + c

        temp = bytes()
        temp = temp + c

        while i<continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s

import struct
def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()

    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f, 'utf-8'))
            embedd_dim = int(_readString(f, 'utf-8'))

            for i in range(wordTotal):
                word = _readString(f, 'utf-8')
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break
                # try:
                #     embedd_dict[word.decode('utf-8')] = word_vector
                # except Exception , e:
                #     pass
                embedd_dict[word] = word_vector

    else:
        with codecs.open(embedding_path, 'r', 'UTF-8') as file:
        # with open(embedding_path, 'r') as file:
            for line in file:
                # logging.info(line)
                line = line.strip()
                if len(line) == 0:
                    continue
                # tokens = line.split()
                tokens = re.split(r"\s+", line)
                # feili
                if len(tokens) == 2:
                    continue # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    # assert (embedd_dim + 1 == len(tokens))
                    if embedd_dim + 1 != len(tokens):
                        continue
                embedd = np.zeros([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd


    return embedd_dict, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def build_pretrain_embedding(embedding_path, word_alphabet, norm):

    embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    logging.info("alphabet size {}".format(alphabet_size))
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1

        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1

        elif re.sub('\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word)])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word)]
            digits_replaced_with_zeros_found += 1

        elif re.sub('\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word.lower())])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1

        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    logging.info("pretrained word emb size {}".format(pretrained_size))
    logging.info("prefect match:%s, case_match:%s, dig_zero_match:%s, "
                 "case_dig_zero_match:%s, not_match:%s"
                 %(perfect_match, case_match, digits_replaced_with_zeros_found,
                   lowercase_and_digits_replaced_with_zeros_found, not_match))
    logging.info('oov: %.2f%%' % (not_match*100.0/alphabet_size))
    return pretrain_emb, embedd_dim

def random_embedding(vocab_size, embedding_dim):
    pretrain_emb = np.zeros([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb

def initialize_emb(path, alphabet, expected_emb_dim):
    if path is not None:
        logging.info("{}: load pretrained embedding from {}".format(alphabet.name, path))
        pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(path, alphabet, False)
        word_embedding = torch.from_numpy(pretrain_word_embedding)
        embedding_size = pretrain_emb_dim
    else:
        logging.info("{}: randomly initialize embedding".format(alphabet.name))
        word_embedding = torch.from_numpy(random_embedding(alphabet.size(), expected_emb_dim))
        embedding_size = expected_emb_dim

    return word_embedding, embedding_size

def word_to_1234gram(word):
    gram1234 = []
    word_ = ["#BEGIN#"]
    for c in word:
        gram1234.append(c)
        word_.append(c)

    word_.append("#END#")

    for i in range(0, len(word_)-1):
        gram1234.append(''.join(word_[i:i+2]))

    for i in range(0, len(word_)-2):
        gram1234.append(''.join(word_[i:i+3]))

    for i in range(0, len(word_)-3):
        gram1234.append(''.join(word_[i:i+4]))

    return gram1234

import pytreebank
from alphabet import Alphabet
def generate_char_emb(dataset, input_path, output_path):

    # step 1, read JMT pre-trained char embedding
    embedd_dim = -1
    embedd_dict = dict()


    with codecs.open(input_path, 'r', 'UTF-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = re.split(r"\s+", line)

            if len(tokens) == 2:
                continue # it's a head
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                # assert (embedd_dim + 1 == len(tokens))
                if embedd_dim + 1 != len(tokens):
                    continue
            embedd = np.zeros([embedd_dim])
            embedd[:] = tokens[1:]

            # remove prefix "1gram-", "2gram-", "3gram-", "4gram-"
            embedd_dict[tokens[0][tokens[0].find('-')+1:]] = embedd

    # step 2, read all words from dataset
    dataset = pytreebank.load_sst(dataset)
    alphabet_token = Alphabet('token')
    for labeled_tree_sentence in dataset['train']:
        label, sentence = labeled_tree_sentence.to_labeled_lines()[0]
        tokens = sentence.split()
        for token in tokens:
            alphabet_token.add(token)

    for labeled_tree_sentence in dataset['dev']:
        label, sentence = labeled_tree_sentence.to_labeled_lines()[0]
        tokens = sentence.split()
        for token in tokens:
            alphabet_token.add(token)

    for labeled_tree_sentence in dataset['test']:
        label, sentence = labeled_tree_sentence.to_labeled_lines()[0]
        tokens = sentence.split()
        for token in tokens:
            alphabet_token.add(token)

    # step 3, iter alphabet, map each word into 1,2,3,4gram, find them in embedd_dict, average the embeddings
    word_embedd_dict = {}
    for word, _ in alphabet_token.iteritems():
        gram1234 = word_to_1234gram(word)
        avg_count = 0
        avg_embedd = np.zeros([embedd_dim])

        for gram in gram1234:
            if gram in embedd_dict:
                vector = embedd_dict[gram]
                avg_embedd += vector
                avg_count += 1

        if avg_count > 0:
            avg_embedd = avg_embedd/avg_count
            word_embedd_dict[word] = avg_embedd

    # step 4, dump into word2vec format
    with codecs.open(output_path, 'w', "UTF-8") as fp:
        for word, vector in word_embedd_dict.items():
            str_rep = word
            for element in vector.flat:
                str_rep = str_rep+" "+str(np.float32(element))

            fp.write(str_rep+"\n")




if __name__ == "__main__":
    generate_char_emb('/Users/feili/dataset/sst/trees', '/Users/feili/Downloads/jmt_pre-trained_embeddings/charNgram.txt',
                      '/Users/feili/Downloads/jmt_pre-trained_embeddings/jmt_word_emb.txt')

