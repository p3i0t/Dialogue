import tensorflow as tf
import numpy as np
import reader
import seq2seq
import time
import os

#import debug_bi_dy_rnn
import rnn_cell_impl
import my_seq2seq


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
    vocab_size = 40000 + 4


class Dialogue(object):
    def __init__(self, config, forward_only=False, bidirectional=True):
        self.vocab_size = config.vocab_size
        num_layers = config.num_layers
        self.num_units = config.hidden_size

        #self.batch_size = config.batch_size
        self.num_steps = config.num_steps

        self.inputs = tf.placeholder(tf.int64, shape=(None, self.num_steps, 1), name='input')
        self.early_stops = tf.placeholder(tf.int64, shape=(None, ), name='early_stops')
        self.targets = tf.placeholder(tf.int64, shape=(None, self.num_steps), name='output')

        num_samples = 1024

        with tf.variable_scope("define_sampled_softmax_with_outprojection"):
            if 0 < num_samples < self.vocab_size:
                if bidirectional:
                    num_units = 2*self.num_units
                else:
                    num_units = self.num_units
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

        with tf.variable_scope("RNN_cells"):
            cell = tf.nn.rnn_cell.GRUCell(self.num_units)
            #cell = rnn_cell_impl.SoftmaxWrapper(cell, w, b)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

            if bidirectional:
                bi_cell = tf.nn.rnn_cell.GRUCell(2 * self.num_units)
                bi_cell = tf.nn.rnn_cell.MultiRNNCell([bi_cell] * num_layers)

        with tf.variable_scope("Dynamic_RNN"):
            encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, embedding_classes=self.vocab_size,
                                                           embedding_size=self.num_units)
            if bidirectional:
                outputs, self.encoder_states = tf.nn.bidirectional_dynamic_rnn(encoder_cell, encoder_cell, self.inputs,
                #outputs, self.encoder_states = tf.nn.bidirectional_dynamic_rnn(encoder_cell, encoder_cell, self.inputs,
                                                            sequence_length=self.early_stops, dtype=tf.float32,
                                                                               time_major=False, scope='bi_rnn')
            else:
                outputs, self.encoder_states = tf.nn.dynamic_rnn(encoder_cell, self.inputs, self.early_stops,
                                                             dtype=tf.float32, time_major=False)

        #print 'encoder_state: ', self.encoder_states
	with tf.variable_scope("atten_states"):
	    if bidirectional:# and isinstance(outputs, tuple) and isinstance(self.encoder_states, tuple):

		outputs = tf.concat(2, [outputs[0], outputs[1]])
		self.encoder_states = tf.concat(1, [self.encoder_states[0][0], self.encoder_states[1][0]])
		self.encoder_states = (self.encoder_states,)
		#print 'encoder_state: ', self.encoder_states
		assert outputs.get_shape()[2] == 2 * self.num_units
		#assert isinstance(self.encoder_states, tuple) and len(self.encoder_states) == 1

	    # Split the outputs to list of 2D Tensors with length num_steps with shape[batch_size, embedding_size]
	    split_outputs = [tf.squeeze(output, [1]) for output in tf.split(1, self.num_steps, outputs)]
	    if bidirectional:
		output_size = cell.output_size * 2
	    else:
		output_size = cell.output_size
	    # First calculate a concatenation of encoder outputs to put attention on.
	    top_states = [tf.reshape(e, [-1, 1, output_size])
		for e in split_outputs]
	    attention_states = tf.concat(1, top_states)

        #with tf.device('/gpu:2'):
	with tf.variable_scope("split_tensors"):
	    mask = tf.sign(tf.to_float(self.targets), name='mask')
	    split_weights = [tf.squeeze(weight, [1]) for weight in tf.split(1, self.num_steps, mask)]
	    split_targets = [tf.squeeze(target, [1]) for target in tf.split(1, self.num_steps, self.targets)]

	    self.dec_inputs = [tf.ones_like(split_targets[0], dtype=tf.int32, name="GO")] + split_targets[:-1]

        def seq2seq_decoder(forward_only):
            if bidirectional:
                outcell = bi_cell
                num_units = 2*self.num_units
            else:
                outcell = cell
                num_units = self.num_units
            beam_search = forward_only
            beam_size = 10

            return my_seq2seq.embedding_attention_decoder(self.dec_inputs, self.encoder_states, attention_states,
                                                       outcell, self.vocab_size, num_units,
                                                       output_projection=output_projection, feed_previous=forward_only,
                                                          beam_search=beam_search, beam_size=beam_size)

        #print 'encoder_state: ', self.encoder_states
        with tf.variable_scope('decoder'):
            if forward_only:
                outputs, states, self.path, self.symbols, self.attentions = seq2seq_decoder(forward_only)
                return
                #outputs, states, self.atten_distributions, path, symbols = seq2seq_decoder(forward_only)
            else:
                outputs, states = seq2seq_decoder(forward_only)

            #print 'outputs: ', outputs
            #print 'states: ', states

        with tf.variable_scope('output_indices'):
            #output_projection first
            projected_outputs = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in outputs]
            self.output_indices = [tf.squeeze(tf.argmax(output, 1)) for output in projected_outputs]

        with tf.variable_scope("loss"):
            self.loss = tf.nn.seq2seq.sequence_loss(outputs, split_targets, split_weights,
                                                    softmax_loss_function=softmax_loss_function)

        with tf.variable_scope('Optimizer'):
            self.train_op = tf.train.AdamOptimizer(name='adam').minimize(self.loss)

        with tf.variable_scope("Saver"):
            self.saver = tf.train.Saver()

        with tf.variable_scope('summaries'):
            loss_summary = tf.scalar_summary('loss', self.loss)
            self.merged_summaries = tf.merge_all_summaries()

    def step(self, session, x, y, x_early_stops, forward_only=False):
        feed_dict = {self.inputs: np.expand_dims(x, 2)}
        feed_dict.update({self.targets: y})
        feed_dict.update({self.early_stops: x_early_stops})

        if forward_only:
            path, symbols, attentions = session.run([self.path, self.symbols, self.attentions], feed_dict)
            return path, symbols, attentions
        else:
            _, loss = session.run([self.train_op, self.loss], feed_dict)
            return loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0,1' 

    config = Config()
    bidirectional = True
    with tf.Session() as session:
        with tf.variable_scope('Model', reuse=None):
            dialogue = Dialogue(config, forward_only=False, bidirectional=bidirectional)
        with tf.variable_scope('Model', reuse=True):
            evaluate_dialogue = Dialogue(config, forward_only=True, bidirectional=bidirectional)

        r = reader.Reader(vocab_size=config.vocab_size - 4, num_steps=config.num_steps, batch_size=config.batch_size)

        tf.initialize_all_variables().run()

        for epoch in xrange(500):
            print "Epoch: {}".format(epoch+1) 
            loss_list = []
            r.batch_size = config.batch_size
            s = time.time()
            for step, (x, y, x_early_steps) in enumerate(r.dynamic_iterator()):
                loss = dialogue.step(session, x, y, x_early_steps)
                loss_list.append(loss)

                if step % 10 == 1:
                    print "step {:<4}, loss: {:.4}".format(step, loss)
                    dialogue.saver.save(session, 'logdir/train/dialogue.ckpt') # save the model periodically

            print "Mean loss: {:.4f}".format(np.mean(loss_list))
            print "Time Elapsed: {:.4f}".format(time.time() - s)

            r.batch_size = 1
            for step, (x, y, x_early_steps) in enumerate(r.dynamic_iterator()):
                path, symbols, attentions = evaluate_dialogue.step(session, x, y, x_early_steps, True)
                
                print "*"*40
                print "post     : ", ' '.join(
                    map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], x[0])))
                print "reference: ", ' '.join(
                    map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], y[0])))
                #print path
                #print symbols #[-1, beam_size]
                print "="*20
                for i in xrange(symbols.shape[1]):
                    print "response : ", ' '.join(map(lambda ind: r.id_to_word[ind],
                                                      filter(lambda ind: ind != r.control_word_to_id['<PAD>'],
                                                             symbols[:, i])))
                '''
                assert x.shape[0] == len(indices[0])
                indices = np.array(indices)#shape: (T, B)
                for i in range(indices.shape[1]):
                    print "************"
                    print "post     : ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], x[i])))
                    print "reference: ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], y[i])))
                    #print "response : ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], indices[:, i])))
                    print "path: ", path
                    print "symbols: ", symbols
                    break # evaluate only one batch
                '''
                print "Attentions: "
                for step, attn in enumerate(attentions):
                    print "atten of step %d: " %(step+1) , attn[0]
                    print "argmax: ", np.argmax(attn[0]), "the word attentioned: ", r.id_to_word[x[0][np.argmax(attn[0])]]
                #print "len of attentions: ", len(attentions)
                #print "attention[0] shape: ", attentions[0].shape, 'sum: ', np.sum(attentions[0])
                #print "sum of one entry: ", np.sum(attentions[0][0])
                if step == 10:
                    break
