# -*- coding : utf-8 -*-

import tensorflow as tf


########################target#############################
# csvデータを取得）
data_queue = tf.train.string_input_producer(["pretarget.csv"], shuffle=True)

# TextLineReader 生成（1行ずつ読み込む Reader）
reader = tf.TextLineReader()
key, value = reader.read(data_queue)

# CSVデコード（列は3列、いず修正修正れも実数値、という指定↓）
data = tf.decode_csv(value, [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
#data = tf.stack([data1, data2, data3, data4, data5, data6])
###########################################################


'''#####################Test##################################
 #csvデータを取得）
data_queue2 = tf.train.string_input_producer(["pretest.csv"])

# TextLineReader 生成（1行ずつ読み込む Reader）
reader2 = tf.TextLineReader()
key2, value2 = reader2.read(data_queue2)

# CSVデコード（列は3列、いずれも実数値、という指定↓）
Data = tf.decode_csv(value2, [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
#Data = tf.stack([Data1, Data2, Data3, Data4, Data5, Data6])
###########################################################'''

init = tf.global_variables_initializer()

x = tf.placeholder(tf.float32, [None, 6])#Noneはtargetの数

# 重みとバイアスを保持するVariableを定義する。
# Variableは操作によって値を修正することのできる変数。
W = tf.Variable(tf.zeros([6, 1])) # 重み number of attributes次元の入力を受けて1次元(Memorability)の出力を返す
b = tf.Variable(tf.zeros([1])) # バイアス 1次元の出力に加えられる

# ニューラルネットモデルを定義する。
# 入力xと重みWの行列積(tf.matmul)の出力にバイアスbを加え、ソフトマックスで最終的な出力を決定する。
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 以上でニューラルネットモデルの定義は終わり。
# 以下は、訓練・評価の実装
# (今回のサンプルでは、交差エントロピーで評価して学習しています。)

# 教師データ(正しい答え)を保持するplaceholderを定義する。
y_ = tf.placeholder(tf.float32, [None, 1])

# 交差エントロピーの計算式を定義する。
# 教師データy_とモデルからの出力yの対数をとったものとの積を取り、全体の合計を計算する
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 各ステップにおけるモデルの更新方法を定義する。
# 1ステップごとに0.01の更新率で交差エントロピーが最小になるようにする。
# (ニューラルネットの学習には誤差逆伝搬法が用いられるが、これはモデルが何かで判定しているらしい。)
# (ここでは、y = tf.nn.softmax(...)としたので誤差逆伝搬法がニューラルネットの更新に採用されるようだ。)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#batch_xs, batch_ys = tf.train.batch([data[0:-1], data[-1:]], batch_size = 1)

#batch_size = 1
#min_after_depueue = 10000
#capacity = min_after_depueue + 3*batch_size

####################################################################
##      Training
####################################################################

# Sessionを定義し、すべての変数を初期化する。
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 訓練を行う
    # mnistデータをSessionに渡して訓練を行い、train_stepの定義に合わせてモデルを更新する。
    for i in range(0, 5):
        #batch_xs, batch_ys = tf.train.shuffle_batch([data[0:-1], data[-1:]], batch_size = batch_size, capacity=capacity, min_after_deque=min_after_depueue)
        #x_train, y_train = sess.run([batch_xs, batch_ys])
        print (0)
        print (0)
        print (0)
        print (0)
        print (sess.run(data[0:-1], data[-1:]))
        #sess.run(train_step, feed_dict={x: data[0:-1], y_: data[-1:]})#xs,ysの後にそれぞれ.eval()を入れると無限ループ


    # 性能評価のための評価式を定義する。
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 最終的な性能評価は平均値で決定することを定義する。
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # テストデータとそのラベルを使って性能を評価する。
    #print (sess.run(accuracy, feed_dict={x: Data, y_: M2}))

coord.request_stop()
coord.join(threads)
