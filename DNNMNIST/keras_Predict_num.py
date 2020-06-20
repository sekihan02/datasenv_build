# coding: utf-8
'''
kerasモデルによる手書き数字認識

使用パッケージのバージョン
kerasのバージョンは2.2.4です
TensorFlowのバージョンは1.13.0-rc1です
'''
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import np_utils

def main():
	"""
	モデルをロード
	"""
	# 学習済モデルをロード
	model = load_model('MNIST.h5')

	"""
	データの前処理
	用意した画像を28×28サイズに加工しつつ、numpy配列に変換し
	前処理を実施
	"""
	# 予測したい画像の保存先
	Xt = []

	# img = cv2.imread('2.jpg', 0)					# PCカメラで保存した画像
	# img = cv2.imread('p1.jpg', 0)					# スマホカメラで保存した画像
	# img = cv2.imread('0.png', 0)					# ペイントで作成した画像
	# img = cv2.imread('1.png', 0)					# ペイントで作成した画像

	img = cv2.imread('number_2.png', 0)			# いらすとや画像
	# img = cv2.imread('number_3.png', 0)			# いらすとや画像

	img = cv2.resize(img,(28, 28), cv2.INTER_CUBIC)

	Xt.append(img)
	Xt = np.array(Xt)

	# 1*28*28の2次元配列を1*784の行列に変換
	Xt = Xt.reshape(1, 784)
	# テストデータを浮動小数点型に変換
	Xt = Xt.astype('float32')
	# シグモイド関数が出力できる範囲にするため、データを255で割り0から1の範囲に変換
	Xt = Xt / 255
	"""
	予測の実行
	"""
	result_predict = model.predict(Xt)
	result_predict_classes = model.predict_classes(Xt)
	result_predict_proba = model.predict_proba(Xt)
	print()
	print('各出力の予測結果: ', result_predict[0])
	print()
	# np.argmax()の引数にndarrayを指定し、最大値となる要素のインデックスを取得
	print('出力予測: ', np.argmax(result_predict[0]))
	print()
	print('クラス(カテゴリ)予測: ', result_predict_classes[0])
	print('クラス確率の予測の確率: ', result_predict_proba[0])
	print()

if __name__ == '__main__':
    main()