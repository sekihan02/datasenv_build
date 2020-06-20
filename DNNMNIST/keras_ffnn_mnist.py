# coding: utf-8
"""
kerasによるNNの実装

使用パッケージのバージョン
kerasのバージョンは2.2.4です
TensorFlowのバージョンは1.13.0-rc1です

ネットワークの内容
  入力層 = 784（画像データ28*28）
  隠れ層 = 200
  出力層 = 10
  学習率 = 0.1
  学習回 = 10

"""
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist

def main():
	try:
		import keras
		kerasExists = True
	except ImportError:
		kerasExists = False

	if kerasExists == True:
		print('kerasのバージョンは{0}です'.format(keras.__version__))
	else:
		print('kerasがインストールされていないか、まだ設定が済んでいません')

	try:
		import tensorflow
		tensorflowExists = True
	except ImportError:
		tensorflowExists = False

	if tensorflowExists == True:
		print('TensorFlowのバージョンは{0}です'.format(tensorflow.__version__))
	else:
		print('TensorFlowがインストールされていないか、まだ設定が済んでいません')



	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	print('データの確認')
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

	"""
	データの表示確認
	"""
	# グラフエリアのサイズを3×3にする
	plt.ﬁgure(ﬁgsize=(13, 4))
	plt.tight_layout()    # 文字配置をいい感じに
	plt.gray()            # グレースケール
	for id in range(3):
		plt.subplot(1, 3, id+1)
		# 784個のRGB値を28*28おｎ行列に変換する
		img = X_train[id, :, :].reshape(28, 28)
		# 色相を反転させて黒でプロット
		plt.pcolor(255 - img)
		# 画像の正解値をプロット
		plt.text(24, 26, '%d' % y_train[id],
				color='b', fontsize=14)
		plt.ylim(27, 0)
	plt.show()

	"""
	データの前処理
	"""
	# 訓練データ
	# 60000*28*28の2次元配列を60000*784の行列に変換
	X_train = X_train.reshape(60000, 784)
	# 訓練データを浮動小数点型に変換
	X_train = X_train.astype('float32')
	# シグモイド関数が出力できる範囲にするため、データを255で割り0から1の範囲に変換
	X_train = X_train / 255

	correct = 10             # 正解ラベルの数

	# 正解ラベルをワンホット表現に変換
	y_train = np_utils.to_categorical(y_train, correct)

	# テストデータ
	# 10000*28*28の2次元配列を60000*784の行列に変換
	X_test = X_test.reshape(10000, 784)
	# テストデータを浮動小数点型に変換
	X_test = X_test.astype('float32')
	# シグモイド関数が出力できる範囲にするため、データを255で割り0から1の範囲に変換
	X_test = X_test / 255
	# 正解ラベルをワンホット表現に変換
	y_test = np_utils.to_categorical(y_test, correct)


	"""
	ネットワークの構築
	"""
	"""
	ニューラルネットワークの構築
	"""
	from keras.models import Sequential
	from keras.layers import Dense, Activation
	from keras.optimizers import SGD

	model = Sequential()                 # Sequentialオブジェクトの生成
	"""
	隠れ層
	"""
	model.add(
		Dense(
			200,                         # 隠れ層のニューロン数 200
			input_dim=784,               # 入力層のニューロン数 784
			activation='relu'            # 活性化関数はReLU
		)
	)
	"""
	出力層
	"""
	model.add(
		Dense(
			10,                           # 出力層のニューロン数は10
			activation='softmax'          # 活性化関数は'softmax'
		)
	)
	"""
	モデルのコンパイル
	"""
	learning_rate = 0.1                   # 学習率
	model.compile(                        # オブジェクトのコンパイル
		loss='categorical_crossentropy',  # 損失の基準は交差エントロピー誤差
		optimizer=SGD(lr=learning_rate),  # 学習方法をSGDにする
		metrics=['accuracy']              # 学習評価として正解率を指定
	)

	model.summary()                       # ニューラルネットワークのサマリーの出力

	"""
	学習を行い結果を出力する
	"""
	history = model.fit(
		X_train,
		y_train,
		epochs=10,                  # 学習回数
		batch_size=100,             # 勾配計算に用いるミニバッチの数
		verbose=1,                  # 学習の進捗状況を出力する
		validation_data=(
			X_test, y_test          # テストデータの指定
		)
	)

	"""
	正解率と損失をグラフにする
	"""
	plt.figure(figsize=(10, 4))    # 図のサイズ
	plt.tight_layout()             # 良い感じに間隔を開ける

	# 1*2のグリッドの左の領域に損失
	plt.subplot(1, 2, 1)
	# 損失のプロット

	# 訓練データの損失（誤り率）をプロット
	plt.plot(history.history['loss'], label='loss', color='black', ls='-', marker='^')
	# テストデータの損失（誤り率）をプロット
	plt.plot(history.history['val_loss'], label='test', color='red', ls='--')
	plt.ylim(0, 1)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(loc='best')
	plt.grid()

	# 1*2のグリッドの右の領域に正解率
	plt.subplot(1, 2, 2)
	# 訓練データの正解率のプロット
	plt.plot(history.history['acc'], label='acc', color='black', ls='-', marker='o')
	# テストデータの正解率のプロット
	plt.plot(history.history['val_acc'], label='test', color='red', ls='--')
	plt.ylim(0, 1)
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(loc='best')
	plt.grid()
	plt.show()

	"""
	モデルの保存
	"""
	model.save('MNIST.h5')
	print('Model Save complete')

	"""
	モデルをロード
	"""
	from keras.models import load_model

	# 学習済モデルをロード
	model = load_model('MNIST.h5')

	"""
	テストデータでモデルの評価を行う
	"""
	score = model.evaluate(X_test, y_test, verbose=True)
	print('evalute loss: ', score[0])
	print('evaluate acc: ', score[1])

if __name__ == '__main__':
    main()