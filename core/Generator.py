import os
import json
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CocoDetection
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from tqdm import tqdm

from torch.utils.data.dataloader import default_collate

# Albumentations를 사용한 데이터 변환 정의
def get_transform(img_size=512, is_normal=True):
    if is_normal:
       return A.Compose([
            A.Resize(img_size, img_size),  # 이미지 크기 조정             
            ToTensorV2()  # albumentations에서 제공하는 PyTorch Tensor 변환
        ], bbox_params=A.BboxParams(format='coco',
                                    label_fields=['category_id', 'class_labels']))        
    else:
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # 50% 확률로 수평 뒤집기
            A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 조정
            A.Resize(img_size, img_size),  # 이미지 크기 조정
            ToTensorV2()  # albumentations에서 제공하는 PyTorch Tensor 변환
        ], bbox_params=A.BboxParams(format='coco',
                                    label_fields=['category_id', 'class_labels']))

# CocoDetection 클래스를 상속받아 Custom Dataset 클래스 정의
class CustomCocoGeneration:
    def __init__(self, img_root: str, coco_ann_path: str):
        self.augmentations = get_transform()
        self.coco_dict = self.load_coco_data(coco_ann_path)
        self.img_root = img_root
        self.category_list = self.get_category_list()
        self.down_ratio = 4
    
    def __len__(self):
        return len(self.coco_dict['images'])
    
    def load_coco_data(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        assert set(data.keys()) == set(['images', 'annotations', 'categories'])
        return data
    
    def get_img_info(self, index):
        return self.coco_dict['images'][index]
    
    def get_category_list(self):
        categories_list = []
        for idx, cate in enumerate(self.coco_dict['categories']):
            if idx == 0:
                if cate['id'] != 0:
                    categories_list.append("none")
            categories_list.append(cate['name'])
        return categories_list
    
    def get_annotation(self, image_id, img_size):
        """
        bboxes = [
        [23, 74, 295, 388, 'dog'],
        [377, 294, 252, 161, 'cat'],
        [333, 421, 49, 49, 'sports ball'],
                                            ]
        """
        ann_list = []
        category_id_list = []
        category_name_list = []
        for ann_info in self.coco_dict['annotations']:
            if ann_info['image_id'] == image_id: 
                x, y, w, h = ann_info['bbox']
                # x = np.clip(ann_info['bbox'][0], 0, img_size[1]-2)
                # y = np.clip(ann_info['bbox'][1], 0, img_size[0]-2)
                # w = np.clip(ann_info['bbox'][2], 0, img_size[1]-2)
                # h = np.clip(ann_info['bbox'][3], 0, img_size[0]-2)  
                # if x+w > img_size[1]:
                #     w = img_size[1] - x
                # if y+h > img_size[0]:
                #     h = img_size[0] - y
                # if sum([x, y, w, h]) == 0:
                #     continue          
                ann_list.append([x, y, w, h])
                #ann_list.append(ann_info['bbox'])
                category_id_list.append(ann_info['category_id']+1)
                category_name_list.append(self.category_list[ann_info['category_id']])
                
        return {'bboxes': ann_list, 
                'category_id': category_id_list, 
                'class_labels': category_name_list}
        
    def create_heatmap(self, bboxes, category_id_list, img_size):
        num_classes = len(self.category_list) + 1
        output_h = img_size[0] // self.down_ratio
        output_w = img_size[1] // self.down_ratio
        max_objs = len(bboxes)
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        indices = torch.zeros(max_objs, dtype=torch.int64)
        width_height = torch.zeros((max_objs, 2), dtype=torch.float32)
        off_set = torch.zeros((max_objs, 2), dtype=torch.float32)
        regression_mask = torch.zeros((max_objs,), dtype=torch.uint8)    
        if max_objs == 0:
            return hm   
  
        for k in range(max_objs):
            bbox = bboxes[k]
            cls_id = category_id_list[k] 
            x1, y1, w, h = map(int, bbox)
            ct = torch.tensor([x1 + w / 2, y1 + h / 2], dtype=torch.float32)
            ct_int = ct.to(torch.int32)
            rescale_x1 = np.clip(x1 / self.down_ratio, 0, output_w)
            rescale_y1 = np.clip(y1 / self.down_ratio, 0, output_h)
            rescale_w = np.clip(w / self.down_ratio, 0, output_w)
            rescale_h = np.clip(h / self.down_ratio, 0, output_h)
            rescale_cx = rescale_x1 + rescale_w / 2
            rescale_cy = rescale_y1 + rescale_h / 2
            peak_point = [rescale_cx, rescale_cy]
                
            radius = self.gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(10, int(radius))                 
            hm[cls_id] = self.draw_msra_gaussian(hm[cls_id], peak_point, radius)
            width_height[k] = torch.tensor([1.0 * w, 1.0 * h])
            indices[k] = ct_int[1] * output_w + ct_int[0]
            off_set[k] = ct - ct_int.to(torch.float32)
            regression_mask[k] = 1
        return {'hm': hm, 'indices': indices,
                'width_height': width_height,
                'off_set': off_set,
                'regression_mask': regression_mask}

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)       
            
    def draw_msra_gaussian(self, heatmap, center, sigma):
        tmp_size = sigma * 3
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        w, h = heatmap.shape[0], heatmap.shape[1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
            return heatmap
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
        img_x = max(0, ul[0]), min(br[0], h)
        img_y = max(0, ul[1]), min(br[1], w)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return heatmap            

    def __getitem__(self, index):
        img_info = self.get_img_info(index)
        ann_dict = self.get_annotation(img_info['id'], (img_info['height'], img_info['width']))
        img_path = os.path.join(self.img_root, img_info['file_name'])
        if not os.path.exists(img_path):
            msg = f'Image file not found: {img_path}'
            raise FileNotFoundError(msg)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(ann_dict['bboxes']) == 0:
            augmented = self.augmentations(image=img,
                                           bboxes=[[1, 2, 3, 4]],
                                           category_id=[0],
                                           class_labels=[0]) 
            augmented['bboxes'] = [[0, 0, 0, 0]]
            augmented['category_id'] = [0]
            augmented['class_labels'] = ["none"]
        else:
            try:
                augmented = self.augmentations(image=img,
                                            bboxes=ann_dict['bboxes'],
                                            category_id=ann_dict['category_id'],
                                            class_labels=ann_dict['class_labels'])
            except:
                augmented = self.augmentations(image=img,
                                            bboxes=[[1, 2, 3, 4]],
                                            category_id=[0],
                                            class_labels=[0]) 
                augmented['bboxes'] = [[0, 0, 0, 0]]
                augmented['category_id'] = [0]
                augmented['class_labels'] = ["none"]
            
        img = augmented['image'] # (C, H, W)
        np_box = np.array(augmented['bboxes'], dtype=np.float32).reshape(-1, 4)
        hm_dict = self.create_heatmap(augmented['bboxes'],
                                    augmented['category_id'],
                                    img.shape[1:])
        #히트맵 상에서의 위치를 표현하기 위한 인덱스
        max_objs = len(augmented['bboxes'])


        gt = {'boxes': np_box,
              'labels': augmented['category_id'],
              'names': augmented['class_labels'],
              'hm': hm_dict['hm'],
              'indices': hm_dict['indices'],
              'width_height': hm_dict['width_height'],
              'off_set': hm_dict['off_set'],
              'regression_mask': hm_dict['regression_mask']}
        #print(gt)
            
        return img, gt        

# 데이터셋과 데이터 로더 설정
img_folder = '/media/data1/anti_uav/ly261666/3rd_Anti-UAV/master/data_files/coco_format/train/imgs'
ann_file = '/media/data1/anti_uav/ly261666/3rd_Anti-UAV/master/data_files/coco_format/train/annotations.json'

transform = get_transform()  # 데이터 변환 가져오기

coco_dataset = CustomCocoGeneration(img_folder, ann_file)
data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=64)

# 데이터 로더를 통해 데이터를 로드하는 예시
count = 1
tmp_image_path = "tmp/"
vis = False
os.makedirs(tmp_image_path, exist_ok=True)
total_size = coco_dataset.__len__()
for imgs, annotations in tqdm(data_loader, total=total_size):
    if vis:
        for batch in range(imgs.shape[0]):
            img = imgs[batch].permute(1, 2, 0).numpy()
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
            bboxes = annotations['boxes'][batch]
            labels = annotations['labels'][0][batch]
            names = annotations['names'][0][batch]
           
            # x1, y1, w, h = map(int, bboxes[0])
            # cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
            # cv2.putText(img, names, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imwrite(tmp_image_path + f"image_{batch}.jpg", img)
            # hm = annotations['hm'][batch]
            # plt.imshow(hm[0], cmap='hot', interpolation='nearest')
            # plt.colorbar()  # 색상 바 추가      
            # plt.savefig(tmp_image_path + f"hm_{batch}.jpg")  
            # plt.clf()

        


    

