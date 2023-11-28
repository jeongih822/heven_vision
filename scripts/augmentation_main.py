import cv2
import os
import numpy as np
from img_aug import Img_aug #데이터 증강 class를 불러옴


file_list = os.listdir('../train/images')

file_name = []
for file in file_list:
    if file.count(".") == 1:
        name = file.split('.')[0]
        file_name.append(int(name))

# augmentation 이전 image 개수
whole_num_of_image = max(file_name)

# 데이터 증강 class 선언
aug = Img_aug(whole_num_of_image+1)		

# 증강결과로 출력되는 원본 이미지 1개당 augmented 이미지의 개수 선언
augment_num = 2	    

# data augmentation 이후 저장되는 경로
save_path = '../train/images/'

# data augmentation 
for i in range(whole_num_of_image + 1):
    jpg_path = f'../train/images/{i+1}.jpeg'
    img = cv2.imread(jpg_path)
    images_aug = aug.seq.augment_images([img for i in range(augment_num)])
    
    for num,aug_img in enumerate(images_aug) :
        cv2.imwrite(save_path+f'{aug.cnt}.jpeg',aug_img)
        print('Image %d complete' %aug.cnt)
        aug.cnt += 1

print('Complete augmenting images')