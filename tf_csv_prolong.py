import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

for N in range(1,10):

####################################################################
##      Input from file using tensorflow queue runners
####################################################################

    exec("train_queue%d = tf.train.string_input_producer(['prolong_target'+ str(N) +'.csv',], num_epochs=None, name='train_queue%d')" % (N,N))
    exec("test_queue%d = tf.train.string_input_producer(['prolong_test'+ str(N) +'.csv'], num_epochs=None, name='test_queue%d')" % (N,N))

    exec("reader1%d = tf.TextLineReader()" % (N))
    exec("reader2%d = tf.TextLineReader()" % (N))
    exec("train_key%d, train_value%d = reader1%d.read(train_queue%d)" % (N,N,N,N))
    exec("test_key%d, test_value%d = reader2%d.read(test_queue%d)" % (N,N,N,N))

    record_defaults1 = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    record_defaults2 = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]

    train_xy, train_x_batch, train_y_batch, test_xy, test_x_batch, test_y_batch=[],[],[],[],[],[]
    exec("train_xy.append(tf.decode_csv(train_value%d, record_defaults=record_defaults1))" % (N))
    train_x_batch.append(tf.train.batch(train_xy[N][0:-1], batch_size=50, name="train%d" % (N), capacity=5000))
    train_y_batch.append(tf.train.batch(train_xy[N][-1:], batch_size=50, name="train%d" % (N), capacity=5000))

    test_xy[N] = tf.decode_csv(test_value[N], record_defaults=record_defaults2)
    test_x_batch[N], test_y_batch[N] = tf.train.batch([test_xy[N,0:-1], test_xy[N,-1:]], batch_size=44, name="test", capacity=4400)
    #test_x = test_xy[0:-1]
    #test_y = test_xy[-1:]

####################################################################
##      Model definition
####################################################################

    x[N]= tf.placeholder(tf.float32, [None, 13])
    Y[N] = tf.placeholder(tf.float32, [None, 1])

    """W1 = tf.Variable(tf.zeros([13, 3]), name='weight1')
    W2 = tf.Variable(tf.zeros([3, 1]), name='weight2')
    b1 = tf.Variable(tf.zeros([3]), name='bias1')
    b2 = tf.Variable(tf.zeros([1]), name='bias2')"""

    W1[N] = tf.Variable(tf.truncated_normal([13, 4], stddev=0.1), name='weight1')
    W2[N] = tf.Variable(tf.truncated_normal([4, 1], stddev=0.1), name='weight2')
    #W3 = tf.Variable(tf.truncated_normal([4, 1], stddev=0.1), name='weight3')
    b1[N] = tf.Variable(tf.truncated_normal([4], stddev=0.1), name='bias1')
    b2[N] = tf.Variable(tf.truncated_normal([1], stddev=0.1), name='bias2')
    #b3 = tf.Variable(tf.truncated_normal([1], stddev=0.1), name='bias3')

    h1[N] = tf.matmul(x[N], W1[N]) + b1[N]
    h2[N] = tf.nn.relu(h1[N])
    h[N] = tf.matmul(h2[N], W2[N]) + b2[N]
    #h4 = tf.nn.relu(h3)
    """keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h1_drop = tf.nn.dropout(h1, 0.5)"""
    #h = tf.matmul(h4, W3) + b3

    #hypo = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=h)
    cost[N] = tf.reduce_mean(tf.square(h[N] - Y[N]))
    #cost = tf.reduce_mean(tf.abs(h3 - Y))

####################################################################
##      Initialization
####################################################################

    optimizer[N] = tf.train.AdamOptimizer(0.1).minimize(cost[N])
    _ = tf.summary.scalar('cost[N]', tf.sqrt(cost[N]))

    sess[N] = tf.Session()
    summary_writer = tf.summary.FileWriter('tf_csv_data', graph_def=sess.graph_def)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

####################################################################
##      Training
####################################################################

    coord[N] = tf.train.Coordinator()
    threads[N] = tf.train.start_queue_runners(sess=sess[N], coord=coord[N])
    testanswer = sess.run(test_y_batch[N])
    print("testanswer is")
    print (testanswer)
    trainanswer = sess.run(train_y_batch[N])
    print("trainanswer is")
    print (trainanswer)

    for step in range(100):
        x_train[N], y_train[N] = sess.run([train_x_batch[N], train_y_batch[N]])
        sess.run(optimizer[N], feed_dict={x:x_train[N], Y:y_train[N]})
        if (step)%40 == 0:
            summary_str = sess.run(tf.summary.merge_all(), feed_dict={x:x_train[N], Y:y_train[N]})
            summary_writer.add_summary(summary_str, step)
        if (step)%500 == 0:
            try:
                print("step : %d" %(step))
                hypo[N] = sess.run(h, feed_dict={x:x_train[N]})
                print("hypo is ")
                print(hypo[N,0:4])
                print("ans is ")
                print(y_train[N,0:4])
                w1[N] = sess.run(W1[N], feed_dict={x:x_train})
                #print("W1 is")
                #print(w1)
                """print("W3 is")
                print(w3[0:4])"""
                """print("y_train is ")
                print(y_train)"""
            except tf.errors.OutOfRangeError:
                print("out of range.")

####################################################################
##      Evaluation
####################################################################

    x_test[N], y_test[N] = sess.run([test_x_batch[N], test_y_batch[N]])
    prediction[N] = sess.run(h[N], feed_dict={x:x_test[N]})
    print("prediction is ")
    print(prediction[N])
    print("y_test is ")
    print(y_test[N])
    accuracy[N] = sess.run(tf.reduce_mean(tf.cast(np.corrcoef(prediction[N].flatten(), y_test[N].flatten()), dtype=tf.float32)))
    print('accuracy: %.3f'%(accuracy[N]))

predictions = tf.stack(prediction[1], prediction[2], prediction[3], prediction[4], prediction[5], prediction[6], prediction[7], prediction[8], prediction[9], prediction[10])
y_tests = tf.stack(y_test[1], y_test[2], y_test[3], y_test[4], y_test[5], y_test[6], y_test[7], y_test[8], y_test[9], y_test[10])
print("accuracies are ")
sum_acc = 0
for i in range(1,10):
    print('accuracy['+str(N)+']='+str(accuracy[N]))
    sum_acc = aum_acc + accuracy[N]
mean_acc = sum_acc / 10
print ('mean accuracy is ' + str(mean_acc))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(y_tests, predictions)
ax.set_title('prediction result')
ax.set_xlabel('answer')
ax.set_ylabel('prediction')
plt.show()
plt.savefig("prolong-result.png")

coord.request_stop()
coord.join(threads)
