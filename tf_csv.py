import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

####################################################################
##      Input from file using tensorflow queue runners
####################################################################

train_queue = tf.train.string_input_producer(['pretarget.csv'],
                                                num_epochs=None,
                                                name='train_queue')

test_queue = tf.train.string_input_producer(['pretest.csv'],
                                                num_epochs=None,
                                                name='test_queue')

reader1 = tf.TextLineReader()
reader2 = tf.TextLineReader()
train_key, train_value = reader1.read(train_queue)
test_key, test_value = reader2.read(test_queue)

record_defaults1 = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
record_defaults2 = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]

train_xy = tf.decode_csv(train_value, record_defaults=record_defaults1)
train_x_batch, train_y_batch = tf.train.batch([train_xy[0:-1], train_xy[-1:]], batch_size=5, name="train")#, capacity=50)

test_xy = tf.decode_csv(test_value, record_defaults=record_defaults2)
test_x_batch, test_y_batch = tf.train.batch([test_xy[0:-1], test_xy[-1:]], batch_size=19, name="test")
#test_x = test_xy[0:-1]
#test_y = test_xy[-1:]

####################################################################
##      Model definition
####################################################################

x = tf.placeholder(tf.float32, [None, 6])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([6, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

h = tf.matmul(x, W) + b
#hypo = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=h)
cost = tf.reduce_mean(tf.abs(h - Y))

####################################################################
##      Initialization
####################################################################

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
summary_writer = tf.summary.FileWriter('tf_csv_data', graph_def=sess.graph_def)

####################################################################
##      Training
####################################################################

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
testanswer = sess.run(test_y_batch)
print("testanswer is")
print (testanswer)
trainanswer = sess.run(train_y_batch)
print("trainanswer is")
print (trainanswer)

for step in range(30000):
    x_train, y_train = sess.run([train_x_batch, train_y_batch])
    sess.run(optimizer, feed_dict={x:x_train, Y:y_train})

    if (step)%500 == 0:
        try:
            print("step : %d" %(step))
            hypo=sess.run(h, feed_dict={x:x_train})
            print("hypo is ")
            print(hypo)
            print("y_train is ")
            print(y_train)
        except tf.errors.OutOfRangeError:
            print("out of range.")

####################################################################
##      Evaluation
####################################################################

x_test, y_test = sess.run([test_x_batch, test_y_batch])
prediction = sess.run(h, feed_dict={x:x_test})
print("prediction is ")
print(prediction)
print("y_test is ")
print(y_test)
accuracy = sess.run(tf.reduce_mean(tf.cast(np.corrcoef(prediction.flatten(), y_test.flatten()), dtype=tf.float32)))

print('accuracy: %.3f'%(accuracy))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(y_test, prediction)
ax.set_title('prediction result')
ax.set_xlabel('answer')
ax.set_ylabel('prediction')
plt.show()
plt.savefig("result.png")

coord.request_stop()
coord.join(threads)
