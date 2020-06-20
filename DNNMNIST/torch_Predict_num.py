# coding: utf-8
'''
pythorchモデルによる手書き数字認識

使用パッケージのバージョン
pythorchのバージョンは1.5.0+cpuです
'''
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset    # データ関連のユーティリティクラスのインポート

# 定数
INPUT_FEATURES = 28 * 28  # 入力層のニューロン数
LAYER1_NEURONS = 200      # 隠れ層のニューロン数
OUTPUT_RESULTS = 10       # 出力層のニューロン数

# 変数 活性化関数
activation = torch.nn.ReLU()     # 活性化関数（隠れ層）ReLU関数    変更可
acti_out = torch.nn.Softmax()    # 活性化関数（出力層）Softmax関数 変更不可

# モデルの定義
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()

		"""層の定義
		"""
		# 隠れ層
		self.layer1 = nn.Linear(
			INPUT_FEATURES,      # 入力層のユニット数
			LAYER1_NEURONS       # 次の層への出力ユニット数
		)
		# 出力層
		self.layer_out = nn.Linear(
			LAYER1_NEURONS,      # 入力ユニット数
			OUTPUT_RESULTS       # 出力結果への出力ユニット数
		)

	def forward(self, x):
		"""フォワードパスの定義
		"""
		# 出力＝活性化関数（第n層（入力））の形式
		x = activation(self.layer1(x))     # 活性化関数は変数として定義
		x = self.layer_out(x)
		# x = acti_out(self.layer_out(x))    # 活性化関数はSoftmaxで固定
		return x

def main():

	"""
	データの前処理
	用意した画像を28×28サイズに加工しつつ、numpy配列に変換し
	前処理を実施
	"""
	# 予測したい画像の保存先
	Xt = []

	img = cv2.imread('2.jpg', 0)					# PCカメラで保存した画像
	# img = cv2.imread('p1.jpg', 0)					# スマホカメラで保存した画像
	# img = cv2.imread('0.png', 0)					# ペイントで作成した画像
	# img = cv2.imread('1.png', 0)					# ペイントで作成した画像

	# img = cv2.imread('number_2.png', 0)			# いらすとや画像
	# img = cv2.imread('number_3.png', 0)			# いらすとや画像

	# 画像を28*28にサイズ変換
	img = cv2.resize(img,(28, 28), cv2.INTER_CUBIC)

	Xt.append(img)									# リストに追加
	Xt = np.array(Xt)								# リストをnumpyに変換

	"""
	Numpyデータをテンソルに変換する
	"""
	t_X_test = torch.from_numpy(Xt).float()
	"""
	データローダー（loader）の作成
	"""
	# 定数（学習方法の設計時）
	BATCH_SIZE = 42        # バッチサイズ
	loader_test = DataLoader(t_X_test, batch_size=BATCH_SIZE)

	"""
	学習済モデルをロード
	"""
	# モデルのパラメータのロード
	param = torch.load('MNIST_torch.pth')
	model = NeuralNetwork() #読み込む前にクラス宣言が必要
	model.load_state_dict(param)
	# ロードしたモデルの内容を出力
	print(model)

	with torch.no_grad():	# 勾配計算しない設定
		for inputs in loader_test:
			# 28*28の入力データを1*784に変換
			inputs = inputs.reshape(-1, 28 * 28)

			# 評価モードに設定（dropoutなどの挙動が評価用になる）
			model.eval()
			# フォワードプロパゲーションで出力結果を取得
			pred_y = model(inputs)    # フォワードプロパゲーションの結果を取得
			# 出力結果を1まで範囲に変化し、その最大値を取得
			_, disc_y = torch.max(pred_y, 1)

	print()
	# numpyの値に変換して出力
	print('出力の予測結果: ', disc_y.numpy())
	# pythonの値に変換して出力
	print('出力の予測結果: ', disc_y.item())


if __name__ == '__main__':
    main()