import tensorflow as tf
import os
import sys
import numpy as np

import reader
from DialogueModel import Dialogue, Config
import time


def create_model(session, forward_only, bidirectional):
    """Create dialogue model and initialize with parameters in session."""
    config = Config()
    dialogue = Dialogue(config, forward_only=forward_only, bidirectional=bidirectional)
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
            dialogue = Dialogue(config, forward_only=False)
            train_summary_writer = tf.train.SummaryWriter('logdir/train', sess.graph)

        with tf.variable_scope("Model", reuse=True):
            evaluate_dialogue = Dialogue(config, forward_only=True)
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
        bidirectional = True
        with tf.variable_scope("Model", reuse=None):
            dialogue = create_model(sess, False, bidirectional)
        with tf.variable_scope("Model", reuse=True):
            eval_dialogue = create_model(sess, True, bidirectional)

        r = reader.Reader(vocab_size=config.vocab_size - 4, num_steps=config.num_steps, batch_size=config.batch_size)

        tf.global_variables_initializer().run()

        def ids_to_words(ids):
            assert isinstance(ids, list)
            words = map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], ids))
            return ' '.join(words)

        r.batch_size = 1
        for step, (x, y, x_early_steps) in enumerate(r.dynamic_iterator()):
            path, symbols, entropies, attentions = eval_dialogue.step(sess, x, y, x_early_steps, True)

            print("*" * 40)
            print "post     : ", ' '.join(
                map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], x[0])))

            print "reference: ", ' '.join(
                map(lambda ind: r.id_to_word[ind], filter(lambda ind: ind != r.control_word_to_id['<PAD>'], y[0])))
            print "=" * 20

            print "=" * 60
            print "Generations"
            print "=" * 60
            assert path.shape == symbols.shape
            assert path.shape == (config.num_steps - 1, config.beam_size)

            candidates = []
            for i in xrange(config.beam_size):
                sentence = []
                depth = config.num_steps - 1 - 1
                j = i
                while True:
                    sentence.append(symbols[depth, j])
                    if depth > 0:
                        j = path[depth, j]  # get the parent node current j
                    elif depth == 0:
                        break
                depth -= 1
            print "candidates {}: {}".format(i + 1, ids_to_words(sentence[::-1]))
            candidates.append(sentence)

            print "Attentions: "
            for ind, attn in enumerate(attentions):
                # print "atten of step %d: " %(step+1) , attn[0]
                if ind < 4:
                    print "argmax: ", np.argmax(attn[0]), "the word attentioned: ", r.id_to_word[x[0][np.argmax(attn[0])]]

            if step == 10:
                break
        '''
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
        '''

if __name__ == '__main__':
    train(num_epoch=100)
    #decode()
