import tensorflow as tf
import numpy as np
import reader
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
    batch_size = 128
    vocab_size = 40000 + 4


class Dialogue(object):
    def __init__(self, config, forward_only=False):
        self.vocab_size = config.vocab_size
        num_layers = config.num_layers
        num_units = config.hidden_size

        #self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        #num_samples = 768

        self.inputs = tf.placeholder(tf.int64, shape=(None, self.num_steps, 1), name='input')
        self.early_stops = tf.placeholder(tf.int64, shape=(None, ), name='early_stops')
        self.targets = tf.placeholder(tf.int64, shape=(None, self.num_steps), name='output')

        num_samples = 512

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

        with tf.variable_scope("RNN_cell"):
            cell = tf.nn.rnn_cell.GRUCell(num_units)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        with tf.variable_scope("Dynamic_RNN"):
            encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, embedding_classes=self.vocab_size,
                                                           embedding_size=num_units)
            outputs, self.encoder_states = tf.nn.dynamic_rnn(encoder_cell, self.inputs, self.early_stops,
                                                             dtype=tf.float32, time_major=False)

        # Split the outputs to list of 2D Tensors with length num_steps with shape[batch_size, embedding_size]
        split_outputs = [tf.squeeze(output, [1]) for output in tf.split(1, self.num_steps, outputs)]
        with tf.variable_scope("atten_states"):
            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                for e in split_outputs]
            attention_states = tf.concat(1, top_states)

        mask = tf.sign(tf.to_float(self.targets), name='mask')
        split_weights = [tf.squeeze(weight, [1]) for weight in tf.split(1, self.num_steps, mask)]
        split_targets = [tf.squeeze(target, [1]) for target in tf.split(1, self.num_steps, self.targets)]

        self.dec_inputs = [tf.ones_like(split_targets[0], dtype=tf.int32, name="GO")] + split_targets[:-1]

        def seq2seq_f(forward_only):
            outcell = cell
            return seq2seq.embedding_attention_decoder(self.dec_inputs, self.encoder_states, attention_states,
                                                       outcell, self.vocab_size, num_units,
                                                       output_projection=output_projection, feed_previous=forward_only)

        outputs, states, self.atten_distributions = seq2seq_f(forward_only)

        self.output_indices = [tf.squeeze(tf.argmax(output, 1)) for output in outputs] # for output

        self.loss = tf.nn.seq2seq.sequence_loss(outputs, split_targets, split_weights,
                                                softmax_loss_function=softmax_loss_function)
        self.train_op = tf.train.AdamOptimizer(name='adam').minimize(self.loss)

    def step(self, session, x, y, x_early_stops, forward_only=False):
        feed_dict = {self.inputs: np.expand_dims(x, 2)}
        feed_dict.update({self.targets: y})
        feed_dict.update({self.early_stops: x_early_stops})
        #print "feed_dict: ", feed_dict

        if forward_only:
            _, loss, indices, atten_distributions = session.run([tf.no_op(), self.loss, self.output_indices,
                                                                 self.atten_distributions], feed_dict)
            return loss, indices, atten_distributions
        else:
            _, loss = session.run([self.train_op, self.loss], feed_dict)
            return loss


if __name__ == '__main__':

    config = Config()

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        with tf.variable_scope('Model', reuse=None):
            dialogue = Dialogue(config, forward_only=False)
        with tf.variable_scope('Model', reuse=True):
            evaluate_dialogue = Dialogue(config, forward_only=True)

        r = reader.Reader(num_steps=config.num_steps, batch_size=config.batch_size)

        tf.global_variables_initializer().run()

        for epoch in xrange(10):
            r.batch_size = config.batch_size
            '''
            for step, (x, y, x_early_steps) in enumerate(r.dynamic_iterator()):

                loss = dialogue.step(session, x, y, x_early_steps)
                if step % 10 == 1:
                    print "step {:<4}, loss: {:.4}".format(step, loss)

            r.batch_size = 10
            '''
            for ind, (x, y, x_early_steps) in enumerate(r.dynamic_iterator()):
                #loss, indices, atten_disbributions = evaluate_dialogue.step(session, x, y, x_early_steps, True)

                #indices = np.array(indices)
                for i in range(x.shape[1]):
                #for i in range(indices.shape[1]):
                    print "************"
                    print "post     : ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], x[i])))
                    print "reference: ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], y[i])))
                    #print "response : ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], indices[:, i])))

                break # evaluate only one batch
            exit(0)