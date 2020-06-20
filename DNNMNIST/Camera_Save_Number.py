# coding: utf-8
'''
usbカメラで顔認識
1.キー's'で保存
2.'q'で終了
'''
import cv2
import time

# 認識時に小さい画像は認識しない
MIN_SIZE = (28, 28)

def main():
	capture = cv2.VideoCapture(0)                               # カメラセット

	# 画像サイズの指定(指定する場合にのみ使う)
	ret = capture.set(28, 28)
	# ret = capture.set(3, 480)
	# ret = capture.set(4, 320)


	i = 0														# 保存画像のカウント
	while True:
		start = time.clock()                                    # 開始時刻
		# 画面指定の時のカメラの画像
		# ret, image = capture.read()

		_, image = capture.read()                               # カメラの画像
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # グレースケールに変換

		key = cv2.waitKey(1) & 0xFF                             # キー入力待ち１ms

		# 's'が押されたら保存
		if key == ord('s'):
			cv2.imwrite(str(i)+'.jpg',image)
			i += 1
			print('Save Image...' + str(i) + '.jpg')
		# 'q'が押されたら終了
		if key == ord('q'):
			capture.release()
			cv2.destroyAllWindows()
			break

		# 映像処理
		get_image_time = int((time.clock()-start) * 1000)         # 処理時間計測
		# 1フレーム取得するのにかかった時間を表示
		cv2.putText(image, str(get_image_time) + 'ms', (10,10), 1, 1, (255,255,255))
		# continue               # 待機だと重いのでこっちのほうが良い？あってもなくても変わらん

		cv2.imshow('Save_Number_Camera',image)

	# キャプチャの後始末と，ウィンドウをすべて消す
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()