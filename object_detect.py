import pickle

import numpy as np
import time
import cv2
import os, csv
from gensim.models import KeyedVectors

word2vec = KeyedVectors.load_word2vec_format('glove.word2vec.300d.txt')

LABELS = open("coco.names").read().strip().split("\n")
np.random.seed(666)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# 导入 YOLO 配置和权重文件并加载网络：
net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
# 获取 YOLO 未连接的输出图层
layer = net.getUnconnectedOutLayersNames()

with open('id_objfeature.pkl', 'rb') as f:
    id_objects = pickle.load(f)

root = 'FB_pic'
dirlist = os.listdir(root)
for i in dirlist:
    # piclist = os.listdir(os.path.join(root, i))
    # for j in piclist:
    #     # print(j)
    id = "FB"+i.split('.')[0]
    print(id)
    path = os.path.join(root, i)
    # try:
    image = cv2.imread(path)
    # 获取图片尺寸
    (H, W) = image.shape[:2]
    # 从输入图像构造一个 blob，然后执行 YOLO 对象检测器的前向传递，给我们边界盒和相关概率
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    # 前向传递，获得信息
    layerOutputs = net.forward(layer)
    # 用于得出检测时间
    end = time.time()
    print("YOLO took {:.6f} seconds".format(end - start))

    objects = np.zeros(300)
    count = 0
    # 循环提取每个输出层
    for output in layerOutputs:
        # 循环提取每个框
        for detection in output:
            # 提取当前目标的类 ID 和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            object = LABELS[classID].split(' ')
            if confidence > 0:
                # print(object, confidence)
                for ob in object:
                    objects += word2vec[ob] * confidence
                    count += 1

    if count > 0:
        objects = objects / count
    # print(objects)
    if id in id_objects.keys():
        id_objects[id] = (id_objects[id] + objects) / 2
    else:
        id_objects[id] = objects
    # except AttributeError:
    #     print('AttributeError')
    #     pass

# root = 'Twitter_pic_1209'
# dirlist = os.listdir(root)
# for i in dirlist:
#     piclist = os.listdir(os.path.join(root, i))
#     for j in piclist:
#         # print(j)
#         id = j.split('_')[0]
#         print(id)
#         path = os.path.join(os.path.join(root, i), j)
#         try:
#             image = cv2.imread(path)
#             # 获取图片尺寸
#             (H, W) = image.shape[:2]
#             # 从输入图像构造一个 blob，然后执行 YOLO 对象检测器的前向传递，给我们边界盒和相关概率
#             blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
#                                          swapRB=True, crop=False)
#             net.setInput(blob)
#             start = time.time()
#             # 前向传递，获得信息
#             layerOutputs = net.forward(layer)
#             # 用于得出检测时间
#             end = time.time()
#             print("YOLO took {:.6f} seconds".format(end - start))
#
#             objects = np.zeros(300)
#             count = 0
#
#             # 循环提取每个输出层
#             for output in layerOutputs:
#                 # 循环提取每个框
#                 for detection in output:
#                     # 提取当前目标的类 ID 和置信度
#                     scores = detection[5:]
#                     classID = np.argmax(scores)
#                     confidence = scores[classID]
#                     object = LABELS[classID].split(' ')
#                     if confidence > 0:
#                         # print(object, confidence)
#                         for ob in object:
#                             objects += word2vec[ob] * confidence
#                             count += 1
#             if count > 0:
#                 objects = objects / count
#             # print(objects)
#             if id in id_objects.keys():
#                 id_objects[id] = (id_objects[id] + objects) / 2
#             else:
#                 id_objects[id] = objects
#
#         except AttributeError:
#             print('AttributeError')
#             pass

with open('id_objfeature1.pkl', 'wb') as f:
    pickle.dump(id_objects, f)

# with open('id_objects_new_literature.csv', 'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     for line in l:
#         writer.writerow(line)