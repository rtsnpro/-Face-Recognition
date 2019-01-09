# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
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

        #第一個預訓練模型---> mtcnn ---->人臉檢測
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy')#第二個參數存放模型所在目錄

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
        for x in range(100): #將辨識人名 辨識值 列出
            SI[0].append('siname')
            SI[1].append('sinum')

###################################
        

        #第二個預訓練模型---> facenet ---->人臉識別，主要是輸出512維的特徵值，作為第三個模型的輸入
        print('Loading feature extraction model')
        modeldir = './pre_model/20180408-102900.pb'
        facenet.load_model(modeldir) #加載模型，模型位於modeldir目錄下

        #獲取輸入和輸出 tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        #第三個預訓練模型 ---> facenet ----> 人臉識別分類
        classifier_filename = './classifier/my_training.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        #video_capture = cv2.VideoCapture('rtmp://140.128.197.154:1935/live1')
        c = 0
        counter = 1
        # #video writer
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=10, frameSize=(1024,768))

        print('Start Recognition!')
        prevTime = 0
        
        j=0
        while True:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval
            counter += 1
            if (counter % 12 == 0):
                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]


                    #檢測出人臉框和5個特徵點
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]#人臉數目
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

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            print(predictions)  #對所有有訓練的資料進行比對的相似度
                            print(type(predictions))
                            best_class_indices = np.argmax(predictions, axis=1) 
                            print(best_class_indices)  #取最佳相似度的序號
                            print(type(best_class_indices))
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            print(best_class_probabilities)  #最佳相似度的值
                            print(type(best_class_probabilities))
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face


                            ###################################################################################
                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            print('result: ', best_class_indices[0])  
                            print(best_class_indices)
                            print(type(best_class_indices))
                            print(HumanNames)   #顯示所有有訓練的人名
                            print(type(HumanNames))


                            #判斷畫面應顯示人名
                            if(best_class_probabilities < 0.3):
                                cv2.putText(frame, 'Unknown', (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)  #偵測畫面顯示未知人名
                                print('Unknown')
                                SI[0][j]= 'Unknown'
                                SI[1][j]= best_class_probabilities.astype(float)
                                print('\n'+ str(j+1)) 

                            else:
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        print(H_i)  #顯示有最佳相似度的人名
                                        SI[0][j]= H_i
                                        result_names = HumanNames[best_class_indices[0]]
                                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)  #偵測畫面顯示已知人名

                                        SI[1][j]= best_class_probabilities.astype(float)
                                        print('\n'+ str(j+1))                 
                            j=j+1

                    else:
                        print('Unable to align')
                    


######################################################################################


                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                st1 = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, st1, (text_fps_x, text_fps_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                # c+=1
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if j >= 100: 
                    break
        
        filename = 'ttt.xls'
        rb = xlrd.open_workbook(filename)
        rs = rb.sheet_by_index(0)
        wb=copy(rb)
        ws=wb.get_sheet(0)

        i = 0
        for siname in SI[0]:#列出人名
            ws.write(i,0,str(siname))#行,列,資料
            i=i+1
                            
        j = 0
        for sinum in SI[1]:#列出比較值
            ws.write(j,1,str(sinum))#行,列,資料
            j=j+1

        wb.save(filename)


        #for k in range(100)):
        #    print('\n'+ str(k+1)+"\t"+str(SI[0][k])+"\t"+str(SI[1][k]))

        video_capture.release()
        # #video writer
        out.release()
        cv2.destroyAllWindows()
