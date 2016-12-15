import tensorflow as tf
import os
import sys
import numpy as np

import reader
from DialogueModel import Dialogue, Config
import time


def create_model(session, forward_only):
    """Create dialogue model and initialize with parameters in session."""
    config = Config()
    dialogue = Dialogue(config, variational=False, forward_only=forward_only)
    path = 'logdir/train/dialogue.ckpt'
    if os.path.exists(path):
        dialogue.saver.restore(session, 'logdir/train/dialogue.ckpt')
        return dialogue
    else:
        raise("{} doesn't exist, cann't recover the model!".format(path))


def train(num_epoch=100):
    with tf.Session() as sess:
        config = Config()
        with tf.variable_scope("Model", reuse=None):
            dialogue = Dialogue(config, variational=False, forward_only=False)
            train_summary_writer = tf.train.SummaryWriter('logdir/train', sess.graph)

        with tf.variable_scope("Model", reuse=True):
            evaluate_dialogue = Dialogue(config, variational=False, forward_only=True)
            evaluate_summary_writer = tf.train.SummaryWriter('logdir/evaluate', sess.graph)

        tf.initialize_all_variables().run()

        r = reader.Reader(vocab_size=config.vocab_size - 4, num_steps=config.num_steps, batch_size=128)

        for epoch in xrange(num_epoch):
            print "Epoch {:>3}".format(epoch+1)
            loss_list = []

            start_time = time.time()
            for step, (x, y, x_early_stops, y_early_stops) in enumerate(r.iterator()):
                loss = dialogue.step(sess, train_summary_writer, x, y, x_early_stops, forward_only=False)
                loss_list.append(loss)
                if step % 1000 == 1:
                    print 'Saving Model ..., step: {:<5}'.format(step)
                    dialogue.saver.save(sess, 'logdir/train/dialogue.ckpt')

            print "Mean loss: {:.4f}".format(np.mean(loss_list))
            print "Time elapsed: {:.4f}".format(time.time()-start_time)

            # evaluate after each epoch
            r.batch_size = 20
            for step, (x, y, x_early_stops, y_early_stops) in enumerate(r.iterator()):
                loss, indices, attention_distributions = evaluate_dialogue.step(sess, evaluate_summary_writer, x, y, x_early_stops, forward_only=True)

                indices = np.array(indices)
                for i in range(indices.shape[1]):
                    print "************"
                    print "post:     ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind!=r.control_word_to_id['<PAD>'], x[i])))
                    print "response: ", ' '.join(map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind!=r.control_word_to_id['<PAD>'], indices[:, i])))

                break # evaluate only one batch


def decode():
    with tf.Session() as sess:
        config = Config()
        dialogue = create_model(sess, True)
        summary_writer = tf.train.SummaryWriter('logdir/test', sess.graph)
        r = reader.Reader(vocab_size=config.vocab_size - 4, num_steps=config.num_steps, batch_size=128)

        # Decode from standard input.
        sys.stdout.write(">> ")
        sys.stdout.flush()
        post = sys.stdin.readline()

        while post:
            # Get token-ids for the input sentence.
            post_tokens, l = r.tokenize_sentence(post)
            response_tokens, _ = r.tokenize_sentence('<GO>')
            loss, indices, attention_distributions = dialogue.step(sess, summary_writer, np.array([post_tokens]),
                                         np.array([response_tokens]), np.array([l]), forward_only=True)
            #_, indices, attention_distributions = sess.run([tf.no_op(), dialogue.output_indices, dialogue.atten_distributions],
            #						feed_dict=feed_dict)
            #print "attentions: ", type(attention_distributions), len(attention_distributions)
            print attention_distributions[0].shape
            m = np.vstack(attention_distributions)
            m = np.transpose(m)

            post = r.recover_sentence(post_tokens)
            response = r.recover_sentence(indices)
            print "len post ", len(post), "len response ", len(response)
            #plt.draw(m[:len(post), :len(response)], post, response)

            print ' '.join(response)
            print ">> ",
            sys.stdout.flush()
            post = sys.stdin.readline()

if __name__ == '__main__':
    train(num_epoch=100)
    #decode()
