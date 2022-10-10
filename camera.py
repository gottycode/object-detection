import cv2

# 定数定義
ESC_KEY = 27     # Escキー
INTERVAL= 33     # 待ち時間
FRAME_RATE = 30  # fps

ORG_WINDOW_NAME = "org"

DEVICE_ID = 0

# カメラ映像取得
print('カメラ起動')
cap = cv2.VideoCapture(DEVICE_ID)

# 初期フレームの読込
end_flag, c_frame = cap.read()
height, width, channels = c_frame.shape

# ウィンドウの準備
cv2.namedWindow(ORG_WINDOW_NAME)

# 変換処理ループ
while end_flag == True:
    img = c_frame
    # フレーム表示
    cv2.imshow(ORG_WINDOW_NAME, c_frame)

    # Escキーで終了
    key = cv2.waitKey(INTERVAL)
    if key == ESC_KEY:
        break

    # 次のフレーム読み込み
    end_flag, c_frame = cap.read()

# 終了処理
cv2.destroyAllWindows()
cap.release()


