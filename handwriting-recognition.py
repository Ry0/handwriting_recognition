# -*- coding:utf-8 -*-
import cv2
import sys
import os.path
import caffe
from caffe.proto import caffe_pb2
import numpy as np
from datetime import datetime

x_old,y_old=-1,-1


def draw_circle(event, x, y, flags, param):
    color=(0,0,0)

    global x_old,y_old

    if event == cv2.EVENT_LBUTTONDOWN:
        x_old = x
        y_old = y
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img,(x,y),15,color,-1)
        cv2.line(img, (x_old, y_old), (x, y), color, 30)
        x_old = x
        y_old = y
    elif event == cv2.EVENT_RBUTTONDOWN:
        rightclick = 1



def read_list(romajilist, hiraganalist):
    romaji = open("romaji.txt")
    hiragana = open("hiragana.txt")

    for moji in romaji:
        moji = moji.rstrip("\n")
        romajilist.append(moji)
    for moji in hiragana:
        moji = moji.rstrip("\n")
        hiraganalist.append(moji)


if __name__ == "__main__":

    romajilist = []
    hiraganalist = []
    read_list(romajilist, hiraganalist)

    img = np.zeros((512,512,3),np.uint8)

    while True:
        img.fill(255)
        cv2.namedWindow('Handwriting Recognition')

        cv2.setMouseCallback('Handwriting Recognition',draw_circle)

        print "\nウィンドウにマウスでひらがなを書いてださい"
        print "書き終わったら \"s\""
        print "書き間違えたら \"c\""

        while True:
            cv2.imshow('Handwriting Recognition',img)

            k = cv2.waitKey(1)&0xFF

            if k == ord('s'):
                # print "s"
                cv2.imwrite("moji.png", img)
                break
            elif k == ord('c'):
                # print "c"
                img.fill(255)
            elif k == 27:
                print "ESC"
                break

        if k == 27:
            break

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

        image = caffe.io.load_image("moji.png")
        predictions = classifier.predict([image], oversample=False)
        pred = np.argmax(predictions)
        sorted_prediction_ind = sorted(range(len(predictions[0])),key=lambda x:predictions[0][x],reverse=True)
        first = str(romajilist[sorted_prediction_ind[0]]) + " (" + str(round(predictions[0,sorted_prediction_ind[0]]*100,2)) + "%)"
        # second = str(romajilist[sorted_prediction_ind[1]]) + " (" + str(round(predictions[0,sorted_prediction_ind[1]]*100,2)) + "%)"
        # third = str(romajilist[sorted_prediction_ind[2]]) + " (" + str(round(predictions[0,sorted_prediction_ind[2]]*100,2)) + "%)"
        # print(predictions)
        # print(pred)
        print "first  -> " + hiraganalist[sorted_prediction_ind[0]] + " (" + str(round(predictions[0,sorted_prediction_ind[0]]*100,2)) + "%)"
        print "second -> " + hiraganalist[sorted_prediction_ind[1]] + " (" + str(round(predictions[0,sorted_prediction_ind[1]]*100,2)) + "%)"
        print "third  -> " + hiraganalist[sorted_prediction_ind[2]] + " (" + str(round(predictions[0,sorted_prediction_ind[2]]*100,2)) + "%)"

        print "\nType \"a\"   もう1回やる"
        print "Type \"s\"   結果を保存してやめる"
        print "Type \"Esc\" やめる\n"

        cv2.putText(img,first,(110, 480), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.CV_AA)
        # print "write"
        cv2.imshow('Handwriting Recognition',img)
        e = cv2.waitKey(0)&0xFF

        if e == ord('s'):
            print "Save image !"
            cv2.imwrite("./image/moji_result.png", img)
            break;
        elif e == ord('a'):
            print "Play agein !"
        elif e == 27:
            print "Exit !"
            break;

    cv2.destroyAllWindows()
    if os.path.isfile("moji.png"):
        os.remove("moji.png")