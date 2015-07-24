# -*- coding:utf-8 -*-
import cv2
import sys
import os.path
import caffe
from caffe.proto import caffe_pb2
import numpy as np
from datetime import datetime

x_old,y_old = -1, -1
click_counter = 0


def read_list(mojilist, inputfile):
    txt = open(inputfile)

    for moji in txt:
        moji = moji.rstrip("\n")
        mojilist.append(moji)


def create_firstcanvas(img):
    img.fill(255)
    cv2.putText(img, "Please write Hiragana with mouse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1, cv2.CV_AA)
    cv2.putText(img, "Here", (75, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (240, 240, 240), 3, cv2.CV_AA)


def draw(event, x, y, flags, param):
    color=(0,0,0)

    global x_old,y_old,click_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        x_old = x
        y_old = y
        if click_counter == 0:
            img.fill(255)
            click_counter += 1
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img,(x,y),15,color,-1)
        cv2.line(img, (x_old, y_old), (x, y), color, 30)
        x_old = x
        y_old = y


def handwrite(img, callback_fun):
    global click_counter

    cv2.namedWindow('Handwriting Recognition')
    cv2.setMouseCallback('Handwriting Recognition', callback_fun)

    print "\nウィンドウにマウスでひらがなを書いてださい"
    print "書き終わったら Type \"S\""
    print "書き間違えたら Type \"C\""
    print "やめたかったら Type \"Esc\""

    while True:
        cv2.imshow('Handwriting Recognition',img)
        # whileで回っているので一応初期化
        k = 0
        k = cv2.waitKey(1)&0xFF

        if k == ord('s'):
            # print "s"
            cv2.imwrite("moji.jpg", img)
            break
        elif k == ord('c'):
            # print "c"
            create_firstcanvas(img)
            click_counter = 0
        elif k == 27:
            # print "ESC"
            break

    return k


def caffe_classify():
    mean_blob = caffe_pb2.BlobProto()
    with open('../tegaki_mean.binaryproto') as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(
    mean_blob.data,
    dtype=np.float32).reshape(
        (mean_blob.channels,
        mean_blob.height,
        mean_blob.width))
    classifier = caffe.Classifier(
        '../tegaki_cifar10_quick.prototxt',
        '../tegaki_cifar10_quick_with_dropout_iter_10000.caffemodel',
        mean=mean_array,
        raw_scale=255)

    image = caffe.io.load_image("moji.jpg")
    predictions = classifier.predict([image], oversample=False)
    # 確率が入ったリストをリターン
    return predictions


def save_filename(moji):
    d = datetime.now()
    filename = moji +"-" + d.strftime('%Y') + "-" + d.strftime('%m') + "-" + d.strftime('%d') + "-" + d.strftime('%H') + "h" + d.strftime('%M') + "m" + d.strftime('%S')  + "s.jpg"
    return filename


if __name__ == "__main__":

    # ローマ字とひらがなのリストをよみこみ（OpenCVが日本語書き込みできないため，ローマ字も）
    romajilist = []
    hiraganalist = []
    read_list(romajilist, "../characterlist/romaji.txt")
    read_list(hiraganalist, "../characterlist/hiragana.txt")

    # 512x512の白色の窓作成
    img = np.zeros((512,512,3),np.uint8)

    while True:
        # 最初のメッセージを書く
        create_firstcanvas(img)
        # 白い窓に字を書かせる
        exit_flag = handwrite(img, draw)
        if exit_flag == 27:
            break

        predictions = caffe_classify()
        sorted_prediction_ind = sorted(range(len(predictions[0])),key=lambda x:predictions[0][x],reverse=True)

        for i in range(3):
            print str(i) + "  -> " + hiraganalist[sorted_prediction_ind[i]] + " (" + str(round(predictions[0,sorted_prediction_ind[i]]*100,2)) + "%)"

        print "\nもう1回やる          Type \"A\"   "
        print "結果を保存してやめる Type \"S\"   "
        print "やめる               Type \"Esc\" \n"

        # cv2.putText()は日本語は出力できないのでローマ字で表記
        first = str(romajilist[sorted_prediction_ind[0]]) + " (" + str(round(predictions[0,sorted_prediction_ind[0]]*100,2)) + "%)"
        cv2.putText(img,first,(96, 480), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.CV_AA)

        cv2.imshow('Handwriting Recognition',img)
        # whileで回っているので一応初期化
        e = 0
        e = cv2.waitKey(0)&0xFF

        if e == ord('s'):
            print "Save image !"
            filename = "../image/" + save_filename(str(romajilist[sorted_prediction_ind[0]]))
            cv2.imwrite(filename, img)
            break;
        elif e == ord('a'):
            click_counter = 0
            print "Play agein !"
        elif e == 27:
            print "Exit !"
            break;
    # end while

    # 最後読み込みに使ったファイル削除
    if os.path.isfile("moji.jpg"):
        os.remove("moji.jpg")
    # 窓削除
    cv2.destroyAllWindows()
