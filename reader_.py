import tensorflow as tf
import collections


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


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
            return

        vocabs = []
        vocabs.extend(self.read_words(self.post))
        vocabs.extend(self.read_words(self.response))
        counter = collections.Counter(vocabs)
        count_pairs = sorted(counter.most_common(self.vocab_size), key = lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        print("Real words in vocab: ", len(words))
        words = ['<GO>', '<PAD>', '<UNK>', '<EOS>'] + list(words)
        self.word_to_id = dict(zip(words, range(len(words))))
        self.id_to_word = dict(zip(self.word_to_id.values(), self.word_to_id.keys()))
        np.savez_compressed('vocab.npz', word_to_id=self.word_to_id, id_to_word=self.id_to_word)

        print "Add control symbols ..."
        print("Total words: ", len(self.id_to_word))

def tokenize_sentence(self, sentence):
        words = sentence.split()
        l = len(words)
        if len(words) < self.num_steps:
            words = words + ['<PAD>']* (self.num_steps - len(words))
        else:
            words = words[:self.num_steps]
        return map(lambda x: self.word_to_id.get(x, self.word_to_id['<UNK>']), words), l