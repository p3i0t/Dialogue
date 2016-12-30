import collections
import numpy as np
import os


_buckets = [(10, 8), (12, 10), (15, 12), (18, 15), (22, 18), (25, 22)]


class Reader(object):
    def __init__(self, vocab_size=10000, num_steps=18, batch_size=128):
        suffix = '_1'
        self.post = 'stc_weibo_train_post' + suffix
        self.response = 'stc_weibo_train_response' + suffix
        self.vocab_size = vocab_size  #shared by post & response
        self.num_steps = num_steps
        self.batch_size = batch_size
        self._build_vocab()

        self.control_word_to_id={'<PAD>': 0,
                    '<GO>': 1, '<UNK>': 2, '<EOS>': 3}
        self.control_id_to_word = dict(zip(range(4), ['<PAD>', '<GO>', '<UNK>', '<EOS>']))

        assert len(self.word_to_id.keys()) == len(self.id_to_word.keys())
        print "Total size of the vocab(including control symbols): {}".format(len(self.id_to_word.keys()))

    def read_words(self, path):
        words = []
        for line in open(path):
            words += line.split()
        return  words

    def _build_vocab(self):
        print "Building vocabulary ..."
        if os.path.isfile('vocab.npz'):
            print "Loading vocab ..."
            d = np.load('vocab.npz')
            self.word_to_id = d['word_to_id'].item() #call item() to transform numpy.ndarray() to dict
            self.id_to_word = d['id_to_word'].item()
            if len(self.id_to_word.keys()) == self.vocab_size + 4:
                return
            else:
                print "vocab.npz data outdated! rebuilding vocab ..."

        vocabs = []
        vocabs.extend(self.read_words(self.post))
        vocabs.extend(self.read_words(self.response))
        counter = collections.Counter(vocabs)
        count_pairs = sorted(counter.most_common(self.vocab_size), key = lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        print("Real words in vocab: ", len(words))
        words = ['<PAD>', '<GO>', '<UNK>', '<EOS>'] + list(words)
        self.word_to_id = dict(zip(words, range(len(words))))
        self.id_to_word = dict(zip(self.word_to_id.values(), self.word_to_id.keys()))
        np.savez_compressed('vocab.npz', word_to_id=self.word_to_id, id_to_word=self.id_to_word)

        print "Add control symbols ..."
        print("Total words: ", len(self.id_to_word))

    def tokenize_sentence(self, sentence):
        words = sentence.split()
        l = len(words)
        if l < self.num_steps:
            words = words + ['<PAD>'] * (self.num_steps - l)
        else:
            words = words[:self.num_steps]
            l = self.num_steps
        return map(lambda x: self.word_to_id.get(x, self.control_word_to_id['<UNK>']), words), l

    def recover_sentence(self, ids):
        ids = filter(lambda id: id!=self.control_word_to_id['<PAD>'], ids)
        return map(lambda id: self.id_to_word.get(id, '<UNK>'), ids)

    def iterator(self):
        posts = []
        posts_lens = []
        for step, line in enumerate(open(self.post)):
            tokens, l = self.tokenize_sentence(line)
            posts_lens.append(l)
            posts.append(tokens)

        responses = []
        responses_lens = []
        for step, line in enumerate(open(self.response)):
            tokens, l = self.tokenize_sentence(line)
            responses_lens.append(l)
            responses.append(tokens)

        data_len = len(responses)

        X = np.array(posts)
        X_early_stops = np.array(posts_lens)
        Y = np.array(responses)
        Y_early_stops = np.array(responses_lens)

        num_batches = data_len / self.batch_size
        shuffle = np.random.permutation(num_batches)
        for ind in range(num_batches):
            i = shuffle[ind]
            x = X[i*self.batch_size: (i+1)*self.batch_size]
            x_early_stops = X_early_stops[i*self.batch_size: (i+1)*self.batch_size]
            y = Y[i*self.batch_size: (i+1)*self.batch_size]
            y_early_stops = Y_early_stops[i*self.batch_size: (i+1)*self.batch_size]
            yield (x, y, x_early_stops, y_early_stops)

    def dynamic_iterator(self):
        posts = []
        posts_lens = []
        for step, line in enumerate(open(self.post)):
            tokens, l = self.tokenize_sentence(line)
            posts_lens.append(l)
            posts.append(tokens)

        responses = []
        responses_lens = []
        for step, line in enumerate(open(self.response)):
            tokens, l = self.tokenize_sentence(line)
            responses_lens.append(l)
            responses.append(tokens)

        data_len = len(responses)

        X = np.array(posts)
        X_early_stops = np.array(posts_lens)
        Y = np.array(responses)

        num_batches = data_len / self.batch_size
        shuffle = np.random.permutation(num_batches)
        for ind in range(num_batches):
            i = shuffle[ind]
            x = X[i*self.batch_size: (i+1)*self.batch_size]
            x_early_stops = X_early_stops[i*self.batch_size: (i+1)*self.batch_size]
            y = Y[i*self.batch_size: (i+1)*self.batch_size]

            #y_early_stops = Y_early_stops[i*self.batch_size: (i+1)*self.batch_size]

            yield (x, y, x_early_stops)

    def test(self):
        posts = []
        posts_lens = []
        for step, line in enumerate(open(self.post)):
            tokens, l = self.tokenize_sentence(line)
            assert np.sum(np.array(tokens) != 0) == l

        print "Len tests passed!"

        inds =  self.id_to_word.keys()
        print 'len of inds', len(inds)
        print 'max of inds', np.max(inds)
        print 'min of inds', np.min(inds)

            #posts_lens.append(l)
            #posts.append(tokens)
        exit(0)
        for step , line in enumerate(open(self.post)):
            if step % 10 == 0:
                input(">> ")
            print "*******"
            print "post: ", line
            ids = map(lambda x: self.word_to_id.get(x, self.word_to_id['<UNK>']), line.split())
            words = map(lambda x: self.id_to_word.get(x), filter(lambda id: id !=0, ids))
            print " ".join(words)
            
            
if __name__ == '__main__':
    r = Reader(vocab_size=40000)
    #r.dynamic_iterator()
    r.test()
    exit(0)
    for step, (x, y, x_early_stops, y_early_stops) in enumerate(r.iterator()):
        print zip(x_early_stops, y_early_stops)
        print "***************"
        if step == 10:
            exit(0)
