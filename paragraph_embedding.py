import tensorflow as tf
import numpy as np
import random
from time import perf_counter
from functools import partial

batch_size = 48
sent_hidden = 256
par_hidden = 512
embedding_size = 300
max_token = 10603
learning_rate = 0.002
grad_clip = 100.0
run = 11
epsilon = 1e-4
beta1 = 0.9
data_size = 75000
epoch = 30
momentum = 0.5
question_min_len = 15

max_batch = data_size // batch_size

meta_list = []

with open('../imdb_train.code.meta') as m:
    for line in m:
        meta_list.append(tuple([int(v) for v in line.split()]))
    random.shuffle(meta_list)
        
f = open('../imdb_train.code')

def imdb_batch_skip(start, end):
    meta = np.array(meta_list[start:end])
    sen_max = meta[:, 1].max()
    word_max = meta[:, 2].max()
    X = np.zeros((end - start, sen_max - 1, word_max))
    y_candidate = []
    answer = np.random.randint(0, 2, end - start)

    def _random_sample():
        while True:
            y_id = random.choice(meta_list)
            y_sen_id = random.randint(1, y_id[1])
            f.seek(y_id[0])
            for j in range(y_sen_id):
                line = f.readline()
            val = [int(v) for v in line.split()]
            if len(val) > question_min_len - 1:
                return val
    
    for i in range(end - start):
        f.seek(meta[i, 0])
        
        #skip = random.randint(0, meta[i, 1] - 1)
        X_candidate = []
        X_len = []

        for j in range(meta[i, 1]):
            val = [int(v) for v in f.readline().split()]
            
            X_len.append(len(val))
            X_candidate.append(val)
            #X[i, j_ind, :length] = val

        if max(X_len) < question_min_len:
            index = random.randint(0, len(X_candidate) - 1)
            y_sample = X_candidate.pop(index)
            X_len.pop(index)

        else:
            choose = []
            for ind, val in enumerate(X_candidate):
                if len(val) > question_min_len - 1:
                    choose.append(ind)
            random.shuffle(choose)
            y_sample = X_candidate.pop(choose[0])
            X_len.pop(choose[0])

        for ind, (val, length) in enumerate(zip(X_candidate, X_len)):
            X[i, ind, :length] = val
            
        if answer[i] == 0:
            y_candidate.append(_random_sample())
            
        else:
            y_candidate.append(y_sample)
            
    y_len = list(map(len, y_candidate))
    y_max = max(y_len)
    y_size = len(y_candidate)
    y = np.zeros((y_size, y_max))

    for i in range(y_size):
        y[i, :y_len[i]] = y_candidate[i]
    return X, y, answer

paragraph_in = tf.placeholder(tf.int64, [None, None, None])
question_in = tf.placeholder(tf.int64, [None, None])
answer_in = tf.placeholder(tf.int64, [None])

cell = tf.nn.rnn_cell.GRUCell(sent_hidden)
W = tf.Variable(tf.random_uniform([max_token + 1, embedding_size], -0.05, 0.05))
par_tr = tf.nn.embedding_lookup(W, tf.transpose(paragraph_in, [1, 0, 2]))

seq_sign = tf.reduce_sum(tf.sign(paragraph_in), 2)
par_len = tf.reduce_sum(tf.sign(seq_sign), 1)
sen_len = tf.transpose(seq_sign)

par_shape = tf.shape(paragraph_in)

output_ta = tf.TensorArray(tf.float32, par_shape[1])
input_ta = tf.TensorArray(tf.float32, par_shape[1])
input_ta = input_ta.unpack(par_tr)
input_len_ta = tf.TensorArray(tf.int64, par_shape[1])
input_len_ta = input_len_ta.unpack(sen_len)

time = tf.constant(0, dtype=tf.int32)

def _step(time, output_ta_t):
    input_t = input_ta.read(time)
    input_len = input_len_ta.read(time)

    with tf.variable_scope('sent'):
        _, output = tf.nn.dynamic_rnn(cell, input_t, sequence_length=input_len, dtype=tf.float32)
    
    output_ta_t = output_ta_t.write(time, output)
    return time + 1, output_ta_t

_, output_final_ta = tf.while_loop(cond=lambda time, _: time < par_shape[1],
                                  body=_step, loop_vars=(time, output_ta), parallel_iterations=32)

final_output = output_final_ta.pack()

cell2 = tf.nn.rnn_cell.GRUCell(par_hidden)

with tf.variable_scope('comp_rnn'):
    _, comp_state = tf.nn.dynamic_rnn(cell2, final_output, dtype=tf.float32, time_major=True)

cell3 = tf.nn.rnn_cell.GRUCell(par_hidden)

question_embed = tf.nn.embedding_lookup(W, question_in)
question_len = tf.reduce_sum(tf.sign(question_in), 1)

with tf.variable_scope('answer_rnn'):
    answer_output, answer_state = tf.nn.dynamic_rnn(cell3, question_embed, question_len, dtype=tf.float32)

comp_weight = tf.Variable(tf.random_normal([par_hidden, par_hidden], stddev=0.05))
comp_bias = tf.Variable(tf.constant(0.0, shape=[par_hidden]))
ans_weight = tf.Variable(tf.random_normal([par_hidden, par_hidden], stddev=0.05))
ans_bias = tf.Variable(tf.constant(0.0, shape=[par_hidden]))

embed_prod = (tf.matmul(comp_state, comp_weight) + comp_bias) + \
            (tf.matmul(answer_state, ans_weight) + ans_bias)

#embed_prod = comp_state * answer_state
#embed_prod = comp_state + answer_state

weight = tf.Variable(tf.random_normal([par_hidden, 1], stddev=0.05))
bias = tf.Variable(tf.constant(0.0, shape=[1]))
logit = tf.matmul(embed_prod, weight) + bias

answer_in_shape = tf.cast(tf.reshape(answer_in, [-1, 1]), tf.float32)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logit, answer_in_shape)
mean_loss = tf.reduce_mean(cross_entropy)

tf.scalar_summary('Loss', mean_loss)

optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, epsilon=epsilon)
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.AdagradOptimizer(learning_rate)
tvars = tf.trainable_variables()
grads = tf.gradients(mean_loss, tvars)
grads, _ = tf.clip_by_global_norm(grads, grad_clip)
train_op = optimizer.apply_gradients(zip(grads, tvars))

batch_index = list(range(max_batch))
random.shuffle(batch_index)

saver = tf.train.Saver()
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter('../../tensorboard/paragraph_2nd/run{}'.format(run), sess.graph,
        flush_secs=5)

    for i in range(epoch):
        loss_sum = 0
        batch_count = 0
        start = perf_counter()
        for j in range(max_batch):
            index = batch_index[j]
            X, Q, y = imdb_batch_skip(index * batch_size, (index + 1) * batch_size)
            #X, Q, y = imdb_batch_skip(0 * batch_size, (0 + 1) * batch_size)
            _, loss, summary = sess.run([train_op, mean_loss, merged],
                {paragraph_in: X, question_in: Q, answer_in: y})
            loss_sum += loss
            batch_count += 1
            writer.add_summary(summary, max_batch * i + j)

            if j % 10 == 0:
                end = perf_counter()
                print('epoch: {} ; batch {} ; loss: {}; time : {}'.format(
                    i + 1, j + 1, loss_sum / batch_count, end - start))
                loss_sum = 0
                batch_count = 0
                start = perf_counter()

        saver.save(sess, './paragraph{}.ckpt'.format(str(i + 1).zfill(2)))

        