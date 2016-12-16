import tensorflow as tf
import numpy as np
import reader
import time

import seq2seq


class Config(object):
    init_scale = 0.05
    learning_rate = 1.0
    momentum_rate = 0.35
    max_grad_norm = 1
    num_layers = 1
    num_steps = 18
    hidden_size = 800
    keep_prob = 0.5
    lr_decay = 0.9
    momentum_decay = lr_decay # same for lr_decay
    batch_size = 64
    vocab_size = 10000 + 4


class Dialogue(object):
    def __init__(self, config, variational=False, forward_only=False):
        self.vocab_size = config.vocab_size
        num_layers = config.num_layers
        num_units = config.hidden_size

        #self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        num_samples = 512

        with tf.variable_scope("RNN_cell"):
            cell = tf.nn.rnn_cell.GRUCell(num_units)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        self.inputs = [tf.placeholder(tf.int64, shape=(None,), name='input_%i' % t)
                    for t in range(self.num_steps)]
        self.early_stops = tf.placeholder(tf.int64, shape=(None,), name='early_stops')

        self.targets = [tf.placeholder(tf.int64, shape=(None,), name='input_%i' % t)
                    for t in range(self.num_steps)]

        self.weights = [tf.ones_like(target, dtype=tf.float32) for target in self.targets]

        self.dec_inputs = ([tf.zeros_like(self.inputs[0], dtype=tf.int32, name="GO")] + self.targets[:-1])

        output_projection = None
        softmax_loss_function = None

        with tf.variable_scope("sampled_softmax"):
            if 0 < num_samples < self.vocab_size:
                w = tf.get_variable("proj_w", [num_units, self.vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.vocab_size])
                output_projection = (w, b)

                def sampled_loss(inputs, labels):
                    labels = tf.reshape(labels, [-1, 1])
                    # compute the sampled_softmax_loss using 32bit floats to avoid numerical instabilities.
                    local_w_t = tf.cast(w_t, tf.float32)
                    local_b = tf.cast(b, tf.float32)
                    local_inputs = tf.cast(inputs, tf.float32)

                    return tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                     num_samples, self.vocab_size)
            softmax_loss_function = sampled_loss

        outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(self.inputs, self.targets, cell, self.vocab_size, self.vocab_size,
                                                  num_units, output_projection=output_projection, feed_previous=forward_only)

        self.output_indices = [tf.squeeze(tf.argmax(output, 1)) for output in outputs] # for output

        with tf.variable_scope("Loss"):
            # a Scalar
            self.loss = tf.nn.seq2seq.sequence_loss(outputs, self.targets, self.weights, softmax_loss_function=softmax_loss_function)#, self.vocab_size)
            loss_summary = tf.scalar_summary('loss', self.loss)

        with tf.variable_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer(name='adam')
            self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()
        #self.scalar_summaries = tf.merge_summary([recon_loss_summary, loss_summary])
        self.merged_summaries = tf.merge_all_summaries()

    def step(self, session, x, y, x_early_stops, forward_only=False):
        feed_dict = {self.inputs[t]: x[:, t] for t in range(self.num_steps)}
        feed_dict.update({self.targets[t]: y[:, t] for t in range(self.num_steps)})
        feed_dict.update({self.early_stops: x_early_stops})
        if forward_only:
            _, loss, indices, weights = session.run([tf.no_op(), self.loss, self.output_indices, self.weights], feed_dict)
            return loss, indices, weights
        else:
            _, loss, summaries, weights = session.run([self.train_op, self.loss, self.merged_summaries, self.weights], feed_dict)
            #summary_writer.add_summary(summaries)
            return loss, weights


def main():
    with tf.Session() as session:
        config = Config()
        r = reader.Reader(vocab_size=config.vocab_size - 4, num_steps=config.num_steps)
        print "vocab_size", len(r.word_to_id)

        with tf.variable_scope("Model", reuse=None):
            dialogue = Dialogue(config, variational=False, forward_only=False)
        with tf.variable_scope("Model", reuse=True):
            test_dialogue = Dialogue(config, variational=False, forward_only=True)

        tf.initialize_all_variables().run()

        for epoch in xrange(20):
            r.batch_size=128
            for step, (x, y, x_early_steps, y_early_steps) in enumerate(r.iterator()):
                loss, weights = dialogue.step(session, x, y, x_early_steps)
                if step % 10 == 1:
                    print "step {:<4}, loss: {:.4}".format(step, loss)
                    if loss < 0.2:
                        print "weights[0] : {}".format(weights[0])
                        print "weights[1] : {}".format(weights[1])
                        print "weights[-2] : {}".format(weights[-2])
                        print "weights[-1] : {}".format(weights[-1])

            r.batch_size = 10

            for ind, (x, y, x_early_steps, y_early_steps) in enumerate(r.iterator()):
                loss, indices, weights = test_dialogue.step(session, x, y, x_early_steps, True)

                indices = np.array(indices)
                for i in range(indices.shape[1]):
                    print "************"
                    print "post:     ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind!=r.control_word_to_id['<PAD>'], x[i])))
                    print "candidate:     ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind!=r.control_word_to_id['<PAD>'], y[i])))
                    print "response: ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind!=r.control_word_to_id['<PAD>'], indices[:, i])))

                break # evaluate only one batch

if __name__ == '__main__':
    main()

