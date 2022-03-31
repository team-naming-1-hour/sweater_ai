from tkinter import image_names
from unittest import result
from venv import create
import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#쿠다코어 사용 가능 확인 및 쿠다 코어 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


#YOLO PARAMS
yolo_df2_params = {   "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
"weights_path" : "yolo/weights/yolov3-df2_15000.weights",
"class_path":"yolo/df2cfg/df2.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET
dataset = 'modanet'

if dataset == 'df2': #deepfashion2
	yolo_params = yolo_df2_params

if dataset == 'modanet':
	yolo_params = yolo_modanet_params


#Classes
# modanet 데이터셋을 이용해 학습시킨 모델의 경우 아래와 같이 의상을 분류합니다
# bag
# belt
# boots
# footwear
# outer
# dress
# sunglasses
# pants
# top
# shorts
# skirt
# headwear
# scarf/tie
classes = load_classes(yolo_params["class_path"])
en2kor_classes = ['가방','벨트','신발','발싸개','아우터','드레스','선글라스','바지','상의','반바지','치마','모자','스카프/넥타이']
#
model = 'yolo'


if model == 'yolo': # YOLOv3 모델을 불러옵니다. 추후 다른 모델을 사용하게 되는 경우 여기에 elif를 추가해 모델을 불러옵니다.
	detectron = YOLOv3Predictor(params=yolo_params)

#Faster RCNN / RetinaNet / Mask RCNN

INPUT_PATH = "./targets"
OUTPUT_PATH = "./results"

if not os.path.exists(INPUT_PATH): # 인풋 폴더 경로가 잘못된 경우
	print('{} does not exists..'.format(INPUT_PATH))
	exit();
if not os.path.exists(OUTPUT_PATH): # 아웃풋 폴더 경로가 잘못된 경우
	print('{} does not exists..'.format(OUTPUT_PATH))
	exit();
		
image_list = os.listdir(INPUT_PATH)
result_txt = open(OUTPUT_PATH+'/result.txt','w',encoding='utf-8')
result_txt.write("{} 개의 이미지 분석 결과\n\n".format(len(image_list)))
find_cloth_count = 0
for image_name in image_list: # INPUT_PATH의 각 이미지에 대해
	try:
		img = cv2.imread(INPUT_PATH+'/'+image_name)
		detections = detectron.get_detections(img) # YOLOv3를 이용한 의상 예측
		create_folder(OUTPUT_PATH+'/'+image_name) # 이미지 이름에 해당하는 폴더 생성
		cv2.imwrite(OUTPUT_PATH+'/'+image_name + '/' + image_name,img) # 원본사진 저장
		each_result_txt = open(OUTPUT_PATH+'/'+image_name+'/result.txt','w',encoding='utf-8')
		each_result_txt.write("{} 분석 결과\n\n".format(image_name))
		detection_count = 0
		if len(detections) > 0 : # 감지된 의상이 1개 이상인 경우
			find_cloth_count+=1;
			detections.sort(reverse=False ,key = lambda x:x[4])
			for x1, y1, x2, y2, cls_conf, cls_pred in detections:
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				x1 = 0 if x1<0 else x1
				x2 = 0 if x2<0 else x2
				y1 = 0 if y1<0 else y1
				y2 = 0 if y2<0 else y2

				if classes[int(cls_pred)] == 'outer':
					# 아우터인 경우
					detection_count+=1;
					each_result_txt.write("%s: Conf: %.5f\n" % (en2kor_classes[int(cls_pred)], cls_conf)) # 결과 작성
					cropped_image = img[y1:y2, x1:x2]
					cv2.imwrite(OUTPUT_PATH+'/'+image_name+'/outer.jpg',cropped_image)
				elif classes[int(cls_pred)] == 'top':
					# 상의인 경우
					detection_count+=1;
					each_result_txt.write("%s: Conf: %.5f\n" % (en2kor_classes[int(cls_pred)], cls_conf))
					cropped_image = img[y1:y2, x1:x2]		
					cv2.imwrite(OUTPUT_PATH+'/'+image_name+'/top.jpg',cropped_image)
				elif classes[int(cls_pred)] == 'shorts':
					# 반바지인인 경우
					detection_count+=1;
					each_result_txt.write("%s: Conf: %.5f\n" % (en2kor_classes[int(cls_pred)], cls_conf))
					cropped_image = img[y1:y2, x1:x2]					
					cv2.imwrite(OUTPUT_PATH+'/'+image_name+'/shorts.jpg',cropped_image)
				elif classes[int(cls_pred)] == 'pants':
					# 하의인인 경우
					detection_count+=1;
					each_result_txt.write("%s: Conf: %.5f\n" % (en2kor_classes[int(cls_pred)], cls_conf))
					cropped_image = img[y1:y2, x1:x2]					
					cv2.imwrite(OUTPUT_PATH+'/'+image_name+'/pants.jpg',cropped_image)
				elif classes[int(cls_pred)] == 'dress':
					# 원피스인 경우
					detection_count+=1;
					each_result_txt.write("%s: Conf: %.5f\n" % (en2kor_classes[int(cls_pred)], cls_conf))
					cropped_image = img[y1:y2, x1:x2]					
					cv2.imwrite(OUTPUT_PATH+'/'+image_name+'/dress.jpg',cropped_image)
				elif classes[int(cls_pred)] == 'skirt':
					# 치마인 경우
					detection_count+=1;
					each_result_txt.write("%s: Conf: %.5f\n" % (en2kor_classes[int(cls_pred)], cls_conf))
					cropped_image = img[y1:y2, x1:x2]					
					cv2.imwrite(OUTPUT_PATH+'/'+image_name+'/skirt.jpg',cropped_image)
		else: # 의상을 감지하지 못한 경우
			print("의상 감지 실패")
		each_result_txt.write("\n {} 개의 의상 탐지".format(detection_count))
		each_result_txt.close()
	except:
		print(image_name+" error")

result_txt.write("{} 개 의상 분류 성공".format(find_cloth_count))
result_txt.close();
