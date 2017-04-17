# -*- coding: utf-8 -*-

import os
import csv
import itertools

import nltk
import numpy as np

import npdl


def load_data(corpus_path=os.path.join(os.path.dirname(__file__), 'data/lm/reddit-comments-2015-08.csv'),
              vocabulary_size=8000):
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    unknown_token = 'UNKNOWN_TOKEN'
    #read
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    with open(corpus_path, encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences
        sentences = [nltk.sent_tokenize(x[0]) for x in reader]
        sentences = itertools.chain(*sentences)
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique tokens in corpus '%s'." % (word_freq.B(), corpus_path))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." %
          (vocab[-1][0], vocab[-1][1]))
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    tokenized_sentences = [[word if word in word_to_index else unknown_token for word in sentence]
                           for sentence in tokenized_sentences]
    # Create the training data.
    train_x = np.asarray([[word_to_index[word] for word in sentence[:-1]] for sentence in tokenized_sentences])
    train_y = np.asarray([[word_to_index[word] for word in sentence[1:]] for sentence in tokenized_sentences])

    return index_to_word, word_to_index, train_x, train_y


def main(max_iter, corpus_path=os.path.join(os.path.dirname(__file__), 'data/lm/tiny_shakespeare.txt')):
    raw_text = open(corpus_path, 'r').read()
    chars = list(set(raw_text))
    data_size, vocab_size = len(raw_text), len(chars)
    print("data has %s charactres, % unique." % (data_size, vocab_size))
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for i, ch in enumerate(chars)}

    time_steps, batch_size = 30, 40

    length = batch_size * 20
    text_pointers = np.random.randint(data_size - time_steps - 1, size=length)
    batch_in = np.zeros([length, time_steps, vocab_size])
    batch_out = np.zeros([length, vocab_size], dtype=np.uint8)
    for i in range(length):
        b_ = [char_to_index[c] for c in raw_text[text_pointers[i]:text_pointers[i] + time_steps + 1]]
        batch_in[i, range(time_steps), b_[:-1]] = 1
        batch_out[i, b_[-1]] = 1

    print("Building model ...")
    net = npdl.Model()
    # net.add(model.layers.SimpleRNN(n_out=500, return_sequence=True))
    net.add(npdl.layers.SimpleRNN(n_out=500, n_in=vocab_size))
    net.add(npdl.layers.Softmax(n_out=vocab_size))
    net.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.00001, clip=5))

    print("Train model ...")
    net.fit(batch_in, batch_out, max_iter=max_iter, batch_size=batch_size)


if __name__ == '__main__':
    main(100)
