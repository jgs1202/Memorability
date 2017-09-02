# -*- coding : utf-8 -*-

import tensorflow as tf


########################学習###############################
# csvデータを取得）
data_queue = tf.train.string_input_producer(["pretarget.csv"])

# TextLineReader 生成（1行ずつ読み込む Reader）
reader = tf.TextLineReader()
key, value = reader.read(data_queue)

# CSVデコード（列は3列、いずれも実数値、という指定↓）
data1, data2, data3, data4, data5, data6, M = tf.decode_csv(value, [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
data = tf.stack([data1, data2, data3, data4, data5, data6])
###########################################################


#####################Test##################################
 #csvデータを取得）
data_queue2 = tf.train.string_input_producer(["pretest.csv"])

# TextLineReader 生成（1行ずつ読み込む Reader）
reader2 = tf.TextLineReader()
key2, value2 = reader2.read(data_queue2)

# CSVデコード（列は3列、いずれも実数値、という指定↓）
Data1, Data2, Data3, Data4, Data5, Data6, M2 = tf.decode_csv(value2, [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
Data = tf.stack([Data1, Data2, Data3, Data4, Data5, Data6])
###########################################################

# 入力情報を持たせるためにplaceholderを定義する。
# shape(第2引数)の次元にNoneを指定すると、どんな長さの次元数であっても対応できる。
x = tf.placeholder("float", [None, 6])#Noneはtargetの数

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
y_ = tf.placeholder("float", [None, 1])

# 交差エントロピーの計算式を定義する。
# 教師データy_とモデルからの出力yの対数をとったものとの積を取り、全体の合計を計算する
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 各ステップにおけるモデルの更新方法を定義する。
# 1ステップごとに0.01の更新率で交差エントロピーが最小になるようにする。
# (ニューラルネットの学習には誤差逆伝搬法が用いられるが、これはモデルが何かで判定しているらしい。)
# (ここでは、y = tf.nn.softmax(...)としたので誤差逆伝搬法がニューラルネットの更新に採用されるようだ。)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# すべての変数を初期化するための準備をする。
init = tf.initialize_all_variables()

# Sessionを定義し、すべての変数を初期化する。
sess = tf.Session()
sess.run(init)

summary_writer = tf.summary.FileWriter('log', graph=sess.graph)

# 訓練を行う
# mnistデータをSessionに渡して訓練を行い、train_stepの定義に合わせてモデルを更新する。
for i in range(1000):
    batch_xs, batch_ys = tf.train.batch([data,M], batch_size = 100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 性能評価のための評価式を定義する。
# 入力データに対するモデルの出力yと教師データy_が一致しているか確認する。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 最終的な性能評価は平均値で決定することを定義する。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# テストデータとそのラベルを使って性能を評価する。
print (sess.run(accuracy, feed_dict={x: Data, y_: M2}))
