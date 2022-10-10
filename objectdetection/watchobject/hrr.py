"""
https://iatom.hatenablog.com/entry/2020/11/01/152120
"""
import numpy as np
import cv2
import time

class Hrr():
    # Shi-Tomasiのコーナー検出パラメータ
    __feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Lucas-Kanade法のパラメータ
    __lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self) -> None:
        self.fullbody_detector = cv2.CascadeClassifier(r"..\cascades\data\haarcascade_fullbody.xml")

        # ランダムに色を100個生成（値0～255の範囲で100行3列のランダムなndarrayを生成）
        self.color = np.random.randint(0, 255, (100, 3))


    def first_frame(self,frame):

        # # 最初のフレームの処理
        # end_flag, frame = cap.read()
        # グレースケール変換
        self.gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 追跡に向いた特徴
        self.feature_prev = cv2.goodFeaturesToTrack(
                                            self.gray_prev,
                                            mask = None,
                                            **Hrr.__feature_params)
        # 元の配列と同じ形にして0を代入
        self.mask = np.zeros_like(frame)

    def frame(self,frame):

        # グレースケールに変換
        self.gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #時間取得
        start = time.time()  
        # 全身の人を検出 
        # minSize:物体が取り得る最小サイズ。これよりも小さい物体は無視される
        # minNeighbors:物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む
        body = self.fullbody_detector.detectMultiScale(
                                    self.gray_next,
                                    scaleFactor=1.1,
                                    minNeighbors=3,
                                    minSize=(40, 40))
        #時間取得
        end = time.time()  
        # 検出時間を表示  
    #    print("{} : {:4.1f}ms".format("detectTime", (end - start) * 1000))

        # オプティカルフロー検出
        # オプティカルフローとは物体やカメラの移動によって生じる隣接フレーム間の物体の動きの見え方のパターン
        self.feature_next, status, err = cv2.calcOpticalFlowPyrLK(
            self.gray_prev, 
            self.gray_next,
            self.feature_prev, 
            None, 
            **Hrr.__lk_params)

        # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）
        good_prev = self.feature_prev[status == 1]
        good_next = self.feature_next[status == 1]


        print(self.feature_prev.shape)
        print(self.feature_next.shape)

        # オプティカルフローを描画
        for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
            prev_x, prev_y = prev_point.ravel()
            next_x, next_y = next_point.ravel()
            # print(next_point)
            # print(prev_x, prev_y)
            # print(next_x, next_y)
            # print(self.mask.shape)
            # print(frame.shape)
            
            self.mask = cv2.line(self.mask, (next_x, next_y), (prev_x, prev_y), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (next_x, next_y), 5, self.color[i].tolist(), -1)
        img = cv2.add(frame, self.mask)
        
        # 人検出した数表示のため変数初期化
        human_cnt = 0
        # 人検出した部分を長方形で囲う
        for (x, y, w, h) in body:
            cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,0),2)
            # 人検出した数を加算
            human_cnt += 1

        # 人検出した数を表示
        cv2.putText(img, "Human Cnt:{}".format(int(human_cnt)),(10,550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)


        # # ウィンドウに表示
        # cv2.imshow('human_view', img)


        # 次のフレーム、ポイントの準備
        self.gray_prev = self.gray_next.copy()
        self.feature_prev = self.good_next.reshape(-1, 1, 2)
        # end_flag, frame = cap.read()

        return img

    def end(self):

        # 終了処理
        cv2.destroyAllWindows()
        # cap.release()


if __name__ == '__main__':
    # サンプル画像
    cap = cv2.VideoCapture('..\\data\\768x576.avi')

    # # 最初のフレームの処理
    end_flag, frame = cap.read()

    hrr = Hrr()
    hrr.first_frame(frame)

    while(end_flag):

        img = hrr.frame(frame)    
        # ESCキー
        k = cv2.waitKey(1)
        if k == 27:
            break

        end_flag, frame = cap.read()

        # ウィンドウに表示
        cv2.imshow('human_view', img)

    hrr.end()
    cap.release()
