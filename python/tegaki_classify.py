# -*- coding:utf-8 -*-
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import sys

def input_arg(argvs, argc):
    if (argc != 2):   # 引数が足りない場合は、その旨を表示
        print 'Usage: # python %s srcdirectory outputdirectory' % argvs[0]
        quit()        # プログラムの終了

    print 'Filename = %s' % argvs[1]
    # 引数でとったディレクトリの文字列をリターン
    return argvs


if __name__ == "__main__":
    argvs = sys.argv   # コマンドライン引数を格納したリストの取得
    argc = len(argvs)  # 引数の個数
    image_path = input_arg(argvs, argc)
    image_src = image_path[1]

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
        '../tegaki_cifar10_quick_iter_10000.caffemodel',
        mean=mean_array,
        raw_scale=255)

    image = caffe.io.load_image(image_src)
    predictions = classifier.predict([image], oversample=False)
    pred = np.argmax(predictions)
    sorted_prediction_ind = sorted(range(len(predictions[0])),key=lambda x:predictions[0][x],reverse=True)
    first = str(sorted_prediction_ind[0]) + " (" + str(round(predictions[0,sorted_prediction_ind[0]]*100,2)) + "%)"
    second = str(sorted_prediction_ind[1]) + " (" + str(round(predictions[0,sorted_prediction_ind[1]]*100,2)) + "%)"
    third = str(sorted_prediction_ind[2]) + " (" + str(round(predictions[0,sorted_prediction_ind[2]]*100,2)) + "%)"
    # print(predictions)
    print(pred)
    print "first  -> " + first
    print "second -> " + second
    print "third  -> " + third
