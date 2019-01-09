# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import xlwt
import xlrd
from xlutils.copy import copy
import webbrowser

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = os.listdir("./input_images")
        HumanNames.sort()
###################################
        SI=[] #建list
        SI.append([]) #多層list
        SI.append([])
        SI.append([])
        SI.append([])
        timenow=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for nameSI in HumanNames: #將人名 出席狀態(未出席) 出席時間(未知時間) 簽到日期 寫入list
            SI[0].append(nameSI)
            SI[1].append("0")
            SI[2].append("none")
            SI[3].append(timenow)

        f=open('signin.txt','w+')#打開檔案
        f.write(timenow+'')#將時間改為本次偵測時間
        f.close()#關閉檔案
###################################
        print('Loading feature extraction model')
        modeldir = './pre_model/20170512-110547.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = './classifier/my_training.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        #video_capture = cv2.VideoCapture(0)
        video_capture = cv2.VideoCapture('rtmp://140.128.197.154:1935/live1')
        c = 0

#
        # #video writer
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(1920,1080))

        print('Start Recognition!')

        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (1024,768), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval
#
            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue
                        if(i>len(cropped)):
                            print('Running')
                            break
                        else:
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            #cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            print(predictions)
                            best_class_indices = np.argmax(predictions, axis=1)
                            print(best_class_indices)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            print(best_class_probabilities)
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

###################################################################################
                        #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        print('result: ', best_class_indices[0])
                        print(best_class_indices)
                        print(HumanNames)
                        peoplenum = 0
                        filename = 'signin.xls'
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:
                                print(H_i)#cmd只顯示有出現在視窗的人名
                                result_names = HumanNames[best_class_indices[0]]
                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 144, 30), thickness=1, lineType=2)
                                if SI[1][peoplenum]=="0":#偵測到的人進行簽到
                                    SI[1][peoplenum]="1"
                                    SI[2][peoplenum]= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                
                            peoplenum+=1
                        print('\n\n>>> If you want to stop the sign in program, press "q" at the wabcam window to close. ')
                        print('>>> If you can\'t stop, check whether you are in the English Input mode or not. \n\n')
                else:
                    print('Unable to align')
######################################################################################

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            str = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, str, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\n===========================================================================================\n')
                print('You have pressed "q" to stop detecting.')
                
                ###########存取偵測結果於表格#############                
                rb = xlrd.open_workbook(filename)
                rs = rb.sheet_by_index(0)
                wb=copy(rb)
                ws=wb.get_sheet(0)

                i = 0
                for names in SI[0]:#列出/刷新全部人名
                    ws.write(i,0,names)#行,列,資料
                    i=i+1
                            
                j = 0
                for comeyn in SI[1]:#列出/刷新全部簽到情形
                    if comeyn == "0":
                        ws.write(j,1,"absent")
                    elif comeyn == "1":
                        ws.write(j,1,"attend")
                    j=j+1

                k = 0
                for sitime_p in SI[2]:#列出/刷新出席時間
                    ws.write(k,2,sitime_p)
                    k=k+1

                l = 0
                for sitime_m in SI[3]:#列出/刷新簽到時間
                    ws.write(l,3,sitime_m)
                    l=l+1

                wb.save(filename)#儲存檔案
                print('Sign in data has updated.')

                ###########上傳資料置資料庫#############
                
                #開啟網頁並上傳表格至資料庫
                webbrowser.open("localhost/face/signin_now.php", 0, False) 

                break

        video_capture.release()
        # #video writer
        out.release()
        cv2.destroyAllWindows()
