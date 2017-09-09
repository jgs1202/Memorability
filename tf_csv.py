import tensorflow as tf

nb_classes = 6

####################################################################
##      Input from file using tensorflow queue runners
####################################################################

train_queue = tf.train.string_input_producer(['pretarget.csv'],
                                                shuffle=False,
                                                name='pretarget_queue')

test_queue = tf.train.string_input_producer(['pretest.csv'],
                                                shuffle=False,
                                                name='pretest_queue')

reader = tf.TextLineReader()

train_key, train_value = reader.read(pretarget_queue)
test_key, test_value = reader.read(pretest_queue)

record_defaults = [[0], [0], [0], [0], [0], [0], [0]]

train_xy = tf.decode_csv(train_value, record_defaults=record_defaults)
train_x_batch, train_y_batch = tf.train.batch([train_xy[0:-1], train_xy[-1:]], batch_size=5)

test_xy = tf.decode_csv(test_value, record_defaults=record_defaults)
test_x_batch, test_y_batch = tf.train.batch([test_xy[0:-1], test_xy[-1:]], batch_size=10)

####################################################################
##      Model definition
####################################################################

x = tf.placeholder(tf.float32, [None, 6])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

h = tf.matmul(x, W) + b
hypo = tf.nn.softmax(h)

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot_reshaped = tf.reshape(Y_one_hot, [-1, nb_classes])

x_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=Y_one_hot_reshaped)
cost = tf.reduce_mean(x_entropy)

####################################################################
##      Initialization
####################################################################

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

####################################################################
##      Training
####################################################################

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(0, 2001):

    x_train, y_train = sess.run([train_x_batch, train_y_batch])

    sess.run(optimizer, feed_dict={x:x_train, Y:y_train})

    if step%20 == 0:

        print(sess.run(cost, feed_dict={x:x_train, Y:y_train}))

####################################################################
##      Evaluation
####################################################################

x_test, y_test = sess.run([test_x_batch, test_y_batch])
prediction = sess.run(hypo, feed_dict={x:x_test})
prediction_one_hot = sess.run(tf.reshape(tf.arg_max(prediction, 1), [-1, 1]))

accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(prediction_one_hot, y_test), dtype=tf.float32)))

print('accuracy: %.3f'%(accuracy))

coord.request_stop()
coord.join(threads)
