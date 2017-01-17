from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves import cPickle as pickle
import math



with open('LSTM\\reverse_dictionary.pickle', 'rb') as f:   #reverse_dictionary open
    reverse_dictionary = pickle.load(f)
    print('reverse_dictionary is loaded')

with open('LSTM\\dictionary.pickle', 'rb') as f:   #reverse_dictionary open
    dictionary = pickle.load(f)
    print('dictionary is loaded')

with open('LSTM\\data_in_number.pickle', 'rb') as f:   #reverse_dictionary open
    text = pickle.load(f)
    print('data is loaded')

with open('LSTM\\embeddings.pickle', 'rb') as f:   #reverse_dictionary open
    embedd = pickle.load(f)
    print('embedding is loaded')






#print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
#print(train_size, train_text[:64])
#print(valid_size, valid_text[:64])









######################################################

batch_size = 64
num_unrollings = 10
embedding_size = 128
vocabulary_size = 5000




try:
    f = open('LSTM_ver2\\cursor.pickle', 'rb')
    cursor = pickle.load(f)
    print('cursor : position %d loaded' % cursor[0])
    f.close()
except FileNotFoundError :
    cursor = [offset * ((len(text))//batch_size) for offset in range(batch_size)]

cursor_for_valid = [offset * ((len(text))//batch_size) for offset in range(batch_size)]


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings, cursor, label = False):   #객체 불러올때만 한번 실행되고 그 이후에는 실행 안됨
        self._text = text
        self._text_size = len(text)   #99999000
        self._label = label

        self._batch_size = batch_size   #64
        self._num_unrollings = num_unrollings  #10

        self._cursor = cursor   #99999000 // 64  * [0, 1, 2, ,,,,, 63]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        if self._label == False :
            batch = np.zeros(shape=(self._batch_size), dtype=np.float)  # (64, 26)
            for b in range(self._batch_size):  # 64
                batch[b] = self._text[self._cursor[b]]  # (0,0) : 0 번째 글짜, (0,1) : 99999000//64 번째 글짜을
            """
            batch = np.zeros(shape=(self._batch_size, embedding_size), dtype=np.float)   #(64, 128)
            for b in range(self._batch_size):   #64
                batch[b] = embedd[self._text[self._cursor[b]]]   #cursor b에 해당하는 embedding 할당
                self._cursor[b] = (self._cursor[b] + 1) % self._text_size   #커서의 위치를 +1 씩
                """
        else :
            batch = np.zeros(shape=(self._batch_size, 1), dtype=np.float)  # (64, 26)
            for b in range(self._batch_size):  # 64
                batch[b,0] = self._text[self._cursor[b]]  # (0,0) : 0 번째 글짜, (0,1) : 99999000//64 번째 글짜을 숫자로,,, one hot
                self._cursor[b] = (self._cursor[b] + 1) % self._text_size  # 커서의 위치를 +1 씩
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]   #
        for step in range(self._num_unrollings):   #10
            batches.append(self._next_batch())  #옮겨진 커서 +1 에서의 글짜들 가져오기
        self._last_batch = batches[-1]
        return batches, self._cursor


def embedd_to_chr(data) :
    similarity = np.dot(data, embedd.transpose())
    word_num = np.argmax(similarity, axis=1)
    word = list()
    for num in word_num :
        word.append(reverse_dictionary[num])
    return word

def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [reverse_dictionary[int(c)] for c in probabilities]

def onehot_characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [reverse_dictionary[c] for c in np.argmax(probabilities, 1)]


def batches2string(batches, label = False):
    s = [''] * batches[0].shape[0]  # 64개의 빈공간
    if label == False :
        for b in batches:
            s = [''.join(x) for x in zip(s, embedd_to_chr(b))]   #zip : 앞에꺼 차레대로 묶어주기. join : 각각의 항목 상이에 ''값 넣어주기
    else :
        for b in batches:
            s = [''.join(x) for x in zip(s, onehot_characters(b))]   #zip : 앞에꺼 차레대로 묶어주기. join : 각각의 항목 상이에 ''값 넣어주기
    return s

def batches2onehot(batches):
    onehot = np.zeros(shape=(num_unrollings, batch_size, vocabulary_size), dtype=np.float)

    for r in range(num_unrollings):
        for b in range(batch_size):
            onehot[r,b,int(batches[r][b][0])] = 1
    return onehot

def word2onehot(word):
    onehot = np.zeros(shape=(1, vocabulary_size), dtype=np.float)
    onehot[0, int(word)] = 1
    return onehot

def save_pickle (data, filename) :
    f = open(filename, 'wb')
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()




train_batches = BatchGenerator(train_text, batch_size, num_unrollings, cursor)
train_labels_batches = BatchGenerator(train_text, batch_size, num_unrollings,cursor, True)
valid_batches = BatchGenerator(valid_text, 1, 1, cursor_for_valid)
valid_labels_batches = BatchGenerator(valid_text, 1, 1, cursor_for_valid, True)


"""
print(batches2string(train_batches.next()))    #train_batches.next() -> (11, 64, 27)
print(batches2string(train_labels_batches.next(), True))   #9글자씩 (64,)

print(batches2string(train_batches.next()))    #train_batches.next() -> (11, 64, 27)
print(batches2string(train_labels_batches.next(), True))   #9글자씩 (64,)
"""

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]



def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):  #5000
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):  #prediction : random_distribution or ligit softmax (1, 27)
  """Turn a (column) prediction into 1-hot encoded samples."""

  p = sample_distribution(prediction[0])
  return p


def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]














num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    #embeddings = tf.Variable(embedd, trainable=False)
    # Input gate: input, previous output , and bias. + logist   / it = σ(Wixxt + Wimmt−1 + Wicct−1 + bi)
    ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.   / ft = σ(Wfxxt + Wmfmt−1 + Wcf ct−1 + bf )
    fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.  /  ct = ft  ct−1 + it  g(Wcxxt + Wcmmt−1 + bc)
    cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.  /  ot = σ(Woxxt + Wommt−1 + Wocct + bo)
    ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.  /  yt = Wymmt + by  추가로 mt = ot  h(ct)
    #w = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], -0.1, 0.1))
    #b = tf.Variable(tf.zeros([embedding_size]))
    w = tf.Variable(
        tf.truncated_normal([num_nodes, vocabulary_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    b = tf.Variable(tf.zeros([vocabulary_size]))


    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state


    # Input data.
    train_data = list()
    label_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.int32, shape=[batch_size]))
    for _ in range(num_unrollings + 1):
        label_data.append(
            tf.placeholder(tf.int32, shape=[batch_size, 1]))
    train_inputs = train_data[:num_unrollings] # 1~ 10번째 까지 알파벳
    train_labels = label_data[1:]  # labels are inputs shifted by one time step.  2~11번째까지 예상

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output   #tf.Variable(tf.zeros([batch_size, num_nodes])
    state = saved_state     #tf.Variable(tf.zeros([batch_size, num_nodes])
    for i in train_inputs:   #10개의 [batch_size, vocabulary_size]
        i_embed = tf.nn.embedding_lookup(embeddings, i)
        drop_i = tf.nn.dropout(i_embed, 0.5)
        drop_o = tf.nn.dropout(output, 0.5)
        output, state = lstm_cell(drop_i, drop_o, state)   #cells 생성
        outputs.append(output)   #logists  (10, batch_size, num_nodes)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),   #dependencies 값들 이후 아래의 것들이 실행 됨
                                  saved_state.assign(state)]):#save_output 에 output 을 assign

        # Classifier.

        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(tf.transpose(w), b, tf.concat(0, outputs),  # 샘플링된 값들만 가지고 softmax 돌림
                                       tf.concat(0, train_labels), 64,
                                       vocabulary_size))  # weights, biases, inputs, labels, num_sampled, num_classes,

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.3, global_step, 5000, 1, staircase=True)   #5000 스텝 마다 r * 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))   #여러 optimizer중에서 gradients, variable 리턴
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)  #gradient가 1.25를 넘지 않게 하여 Exploding을 막음
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction =  tf.nn.softmax(tf.nn.xw_plus_b(tf.concat(0, outputs), w, b))

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.int32, shape=[1])
    sample_embed_i = tf.nn.embedding_lookup(embeddings, sample_input)
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(   #cell 생성
        sample_embed_i, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))









num_steps = 700000001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()  # 세이브
    ckpt = tf.train.get_checkpoint_state('LSTM_ver2\\cps\\')  # 세이브 폴더 리턴
    if ckpt and ckpt.model_checkpoint_path:
        print('load learning')
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        tf.global_variables_initializer().run()
        print("Initialized")

    mean_loss = 0
    for step in range(num_steps):
        batches, cursor_save = train_batches.next()   #(11,64,128)
        batch_labels, _ = train_labels_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
            feed_dict[label_data[i]] = batch_labels[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(batches2onehot(list(batch_labels[1:])))  #(64*10, 27)
            print('Minibatch perplexity: %.2f' % float(logprob(predictions, labels)))



            if step != 0 and step % (summary_frequency * 10) == 0:
                # Generate some samples.

                """


                print('=' * 80)
                feed = dictionary['。']
                sentence = ''
                for i in range(6):
                    #reset_sample_state.run()
                    for _ in range(80):
                        prediction = sample_prediction.eval({sample_input: [feed]})
                        feed = sample(prediction)
                        sentence += reverse_dictionary[feed]
                        if i == 5 and feed == dictionary['。']:
                            break
                    sentence += '\n'
                print(sentence)
                print('=' * 80)

                """




                valid_logprob = 0
                for _ in range(valid_size):  # 1000
                    b, _ = valid_batches.next()
                    l, _ = valid_labels_batches.next()
                    predictions = sample_prediction.eval({sample_input: b[0]})
                    valid_logprob = valid_logprob + logprob(predictions, word2onehot(b[1]))
                print('Validation set perplexity: %.2f' % float((
                    valid_logprob / valid_size)))





                print('cursor at %1.2f %%' %(100 * cursor_save[0] / len(text)))
                saver.save(session, 'LSTM_ver2\\cps\\' + 'model.ckpt')
                save_pickle(cursor_save, 'LSTM_ver2\\cursor.pickle')
            # Measure validation set perplexity.
            reset_sample_state.run()

