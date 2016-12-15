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
    batch_size = 128 
    vocab_size = 40000 + 4 


class Dialogue():
    def __init__(self, config, variational=False, forward_only=False):
        self.vocab_size = config.vocab_size
        num_layers = config.num_layers
        num_units = config.hidden_size

        #self.batch_size = config.batch_size
        self.num_steps = config.num_steps
	num_samples = 768 

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
		if num_samples > 0 and num_samples < self.vocab_size:
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
	''' 
        with tf.variable_scope("encoder"):
            encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, embedding_classes=self.vocab_size, embedding_size=num_units)
            self.encoder_outputs, self.encoder_state = tf.nn.rnn(encoder_cell, self.inputs, dtype=tf.float32, sequence_length=self.early_stops)

        with tf.variable_scope("atten_states"):
            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                for e in self.encoder_outputs]
            attention_states = tf.concat(1, top_states)

	if variational:
		with tf.variable_scope("Variational_Inference"):
		    latent_size = 1

		    self.encoder_state = tf.nn.relu(self.encoder_state)
		    self.enc_mu = linear([config.hidden_size, latent_size])
		    self.enc_logvar_sigma = linear([config.hidden_size, latent_size])

		    self.mu = self.enc_mu(self.encoder_state)
		    self.logvar_sigma = self.enc_logvar_sigma(self.encoder_state)

		    #sampling
		    self.epsilon = tf.random_normal(tf.shape(self.logvar_sigma), name='epsilon')
		    self.std = tf.exp(0.5 * self.logvar_sigma)
		    self.z = self.mu + self.std * self.epsilon

		    self.dec = linear([latent_size, config.hidden_size])

		    self.encoder_state = tf.nn.relu(self.dec(self.z))

		    self.trans = linear([config.hidden_size, config.hidden_size])
		    self.encoder_state = self.trans(self.encoder_state)
		    self.KL_D = -0.5 * tf.reduce_sum(1 + self.logvar_sigma - tf.pow(self.mu, 2) - tf.exp(self.logvar_sigma))

		    KL_loss_summary = tf.summary.scalar("KL_loss", self.KL_D)

	"""
	forward_only: False if training, True if testing.
	"""
	def seq2seq_f(forward_only):
            return seq2seq.embedding_attention_decoder(self.dec_inputs, 
                                       self.encoder_state, attention_states, out_cell, self.vocab_size,
                                       output_projection=output_projection, embedding_size=num_units,
					feed_previous=forward_only)

        with tf.variable_scope("decoder"):
            out_cell = cell #tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)
            #out_cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)
            outputs, states, self.atten_distributions = seq2seq_f(forward_only) 
			#seq2seq.embedding_attention_decoder(self.dec_inputs, 
                         #              self.encoder_state, attention_states, out_cell, self.vocab_size,
            #                           self.encoder_state, out_cell, self.vocab_size, embedding_size=num_units)
	'''

	def seq2seq_f(forward_only):
		return embedding_attention_seq2seq(self.inputs, self.targets, cell,
					 self.vocab_size, self.vocab_size, embedding_size=num_units,
					 output_projection=output_projection, feed_previous=forward_only)
        '''
	attn_dis = [tf.reshape(attn_d, [-1, int(attn_d.get_shape()[1]), 1, 1])
		for attn_d in self.atten_distributions]
	attn_images = tf.concat(2, attn_dis)
	img_summary = tf.image_summary('attention_images', attn_images)
        '''
        self.output_indices = [tf.squeeze(tf.argmax(output, 1)) for output in outputs] # for output
       
	seq2seq.model_with_buckets()
	'''
        with tf.variable_scope("Loss"): 
		# a Scalar
		self.recon_loss = tf.nn.seq2seq.sequence_loss(outputs, self.targets, self.weights, softmax_loss_function=softmax_loss_function)#, self.vocab_size)
		recon_loss_summary = tf.scalar_summary('recon_loss', self.recon_loss)

		if variational:
		    self.loss = self.KL_D + self.recon_loss
		else:
		    self.loss = self.recon_loss

		loss_summary = tf.scalar_summary('loss', self.loss)

	'''
        with tf.variable_scope("Optimizer"):
		optimizer = tf.train.AdamOptimizer(name='adam')
		self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()
	self.scalar_summaries = tf.merge_summary([recon_loss_summary, loss_summary])
        self.merged_summaries = tf.merge_all_summaries()

    def step(self, session, summary_writer, x, y, x_early_stops, forward_only=False):
	feed_dict = {self.inputs[t]: x[:, t] for t in range(self.num_steps)}
	feed_dict.update({self.targets[t]: y[:, t] for t in range(self.num_steps)})
	feed_dict.update({self.early_stops: x_early_stops})
	if forward_only:
		_, loss, indices, scalar_summaries, atten_distributions = session.run([tf.no_op(), self.loss, self.output_indices, self.scalar_summaries, self.atten_distributions], feed_dict) 
		summary_writer.add_summary(scalar_summaries)
		return loss, indices, atten_distributions
	else:
		_, loss, summaries = session.run([self.train_op, self.loss, self.merged_summaries], feed_dict) 
		summary_writer.add_summary(summaries)
                return loss


	#loss_list.append(loss)
	'''
	if step % 1000 == 1:
	print "Saving model ..."
	self.saver.save(session, 'logdir/train/dialogue.ckpt') # save the model every 100 steps

	'''

    '''
    # r: reader object
    def fit(self, session, summary_writer, r, nb_epoch=10):
        session.run(tf.initialize_all_variables())
        for ind in xrange(nb_epoch):
            print "================================"
            print "Epoch %d" % (ind+1)
            self._step(session, summary_writer, r)
    def evaluate(self, session, summary_writer, r):
        for step, (x, y, x_early_stops, y_early_stops) in enumerate(r.iterator()):
            feed_dict = {self.inputs[t]: x[:, t] for t in range(self.num_steps)}
            feed_dict.update({self.targets[t]: y[:, t] for t in range(self.num_steps)})
            feed_dict.update({self.early_stops: x_early_stops})

            _, indices, summaries = session.run([tf.no_op(name='empty'), self.output_indices, self.merged_summary], feed_dict)
            summary_writer.add_summary(summaries)


    def test(self):
        r = reader.Reader(1)
        print "starts ..."
        for step, (x, y, x_early_stops, y_early_stops) in enumerate(r.iterator()):
            print "post %d: " %(step+1) , ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind!=0, x[0])))
            if step % 10 == 1:
                s = input(">> ")
    '''
'''            
def main():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True

    with tf.Session(config=sess_config) as session:
        config = Config()
        r = reader.Reader(vocab_size=config.vocab_size - 4, num_steps=config.num_steps)
        print "vocab_size", len(r.word_to_id)
        dialogue = Dialogue(config, variational=False, forward_only=False)
        summary_writer = tf.train.SummaryWriter('logdir', session.graph)
        dialogue.fit(session, summary_writer, r, nb_epoch= 555)


if __name__ == '__main__':
    main()
'''
