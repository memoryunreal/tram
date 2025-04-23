import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import argparse
import numpy as np
from glob import glob
import cv2
import torch.nn.functional as F
from pycocotools import mask as masktool
from segment_anything import SamPredictor, sam_model_registry

from lib.pipeline import video2frames
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov', help='input video')
parser.add_argument("--object", type=str, default='none', help='target object to detect and segment')
parser.add_argument("--thresh", type=float, default=0.25, help='detection confidence threshold')
parser.add_argument("--min_size", type=int, default=100, help='minimum size of detection')
parser.add_argument("--manual_box", type=str, default=None, help='manually specify bounding box as x1,y1,x2,y2')
parser.add_argument("--track", action='store_true', help='track object across frames using first frame box')
parser.add_argument("--exclude_person", action='store_true', help='exclude person detections')
args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
mask_folder = f'{seq_folder}/masks'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

##### Extract Frames #####
print('Extracting frames ...')
nframes = video2frames(file, img_folder)

##### Detection + SAM Segmentation #####
print(f'Detect and Segment {args.object} ...')
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

# Parse manual box if provided
manual_box = None
if args.manual_box:
    try:
        manual_box = np.array([float(x) for x in args.manual_box.split(',')]).reshape(1, 4)
        print(f"Using manual bounding box: {manual_box[0]}")
    except:
        print("Error parsing manual box. Format should be x1,y1,x2,y2")
        manual_box = None

# Initialize tracking box
tracking_box = None

# ViTDet for detection
cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = args.thresh
detector = DefaultPredictor_Lazy(detectron2_cfg)

# SAM for segmentation
sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
_ = sam.to('cuda')
predictor = SamPredictor(sam)

# Object ID to detect (based on COCO dataset)
object_id_map = {
    'person': 0, 
    'bicycle': 1, 
    'car': 2, 
    'motorcycle': 3,
    'airplane': 4, 
    'bus': 5, 
    'train': 6, 
    'truck': 7,
    'boat': 8, 
    'traffic light': 9, 
    'fire hydrant': 10, 
    'stop sign': 11,
    'parking meter': 12, 
    'bench': 13, 
    'bird': 14, 
    'cat': 15,
    'dog': 16, 
    'horse': 17, 
    'sheep': 18, 
    'cow': 19,
    'elephant': 20, 
    'bear': 21, 
    'zebra': 22, 
    'giraffe': 23,
    'backpack': 24, 
    'umbrella': 25, 
    'handbag': 26, 
    'tie': 27,
    'suitcase': 28, 
    'frisbee': 29, 
    'skis': 30, 
    'snowboard': 31,
    'sports ball': 32, 
    'kite': 33, 
    'baseball bat': 34, 
    'baseball glove': 35,
    'skateboard': 36, 
    'surfboard': 37, 
    'tennis racket': 38, 
    'bottle': 39,
    'wine glass': 40, 
    'cup': 41, 
    'fork': 42, 
    'knife': 43,
    'spoon': 44, 
    'bowl': 45, 
    'banana': 46, 
    'apple': 47,
    'sandwich': 48, 
    'orange': 49, 
    'broccoli': 50, 
    'carrot': 51,
    'hot dog': 52, 
    'pizza': 53, 
    'donut': 54, 
    'cake': 55,
    'chair': 56, 
    'couch': 57, 
    'potted plant': 58, 
    'bed': 59,
    'dining table': 60, 
    'toilet': 61, 
    'tv': 62, 
    'laptop': 63,
    'mouse': 64, 
    'remote': 65, 
    'keyboard': 66, 
    'cell phone': 67,
    'microwave': 68, 
    'oven': 69, 
    'toaster': 70, 
    'sink': 71,
    'refrigerator': 72, 
    'book': 73, 
    'clock': 74, 
    'vase': 75,
    'scissors': 76, 
    'teddy bear': 77, 
    'hair drier': 78, 
    'toothbrush': 79,
    # Custom objects
    'pikaqiu': 15,  # Using cat class ID for pikachu (closest match)
    'pikachu': 15,  # Alternate spelling
}

# Get target object ID
target_object = args.object.lower()
exclude_classes = []
if args.exclude_person:
    exclude_classes.append(0)  # 排除人类 (Person ID = 0)
    print("Excluding person detections")

if target_object in object_id_map:
    target_object_id = object_id_map[target_object]
    print(f"Detecting {target_object} (class ID: {target_object_id})")
else:
    # 如果对象名称不在预定义映射中，则尝试使用所有适当的检测
    print(f"Object '{target_object}' not specifically mapped, will detect all objects except excluded classes")
    target_object_id = None

# Simple IoU function for tracking
def compute_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou

# Process each frame
for frame_idx, imgpath in enumerate(imgfiles):
    print(f"Processing frame {frame_idx+1}/{len(imgfiles)}: {os.path.basename(imgpath)}")
    img_cv2 = cv2.imread(imgpath)
    
    # Use manual box for the first frame or all frames
    if manual_box is not None and (frame_idx == 0 or not args.track):
        boxes = manual_box
        # Add dummy confidence score
        boxes = np.hstack([boxes, np.ones((boxes.shape[0], 1))])
    # Use tracking box if enabled
    elif tracking_box is not None and args.track:
        boxes = tracking_box
    # Otherwise use detection
    else:
        # Detection with ViTDet
        with torch.no_grad():
            det_out = detector(img_cv2)
            det_instances = det_out['instances']
            
            # Filter detections to target object if specified
            if target_object_id is not None:
                valid_idx = (det_instances.pred_classes == target_object_id) & (det_instances.scores > args.thresh)
            else:
                # 如果没有特定对象ID，使用所有检测，但排除指定类别
                valid_idx = det_instances.scores > args.thresh
                if exclude_classes:
                    for class_id in exclude_classes:
                        valid_idx = valid_idx & (det_instances.pred_classes != class_id)
                
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            confs = det_instances.scores[valid_idx].cpu().numpy()
            classes = det_instances.pred_classes[valid_idx].cpu().numpy()
            boxes = np.hstack([boxes, confs[:, None]])
            
            # 打印类别信息，帮助调试
            if len(boxes) > 0:
                print(f"Detected classes: {classes}")
            
            # Filter by minimum size
            if args.min_size:
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                valid = np.stack([w, h]).max(axis=0) > args.min_size
                boxes = boxes[valid]
            
            # 只保留置信度最高的一个框
            if len(boxes) > 1:
                # 按置信度排序并只保留最高的一个
                highest_conf_idx = np.argmax(boxes[:, 4])
                boxes = boxes[highest_conf_idx:highest_conf_idx+1]
                print(f"Selected highest confidence box: {boxes[0, :4]} (score: {boxes[0, 4]:.3f})")
            
        # Update tracking box if tracking is enabled
        if len(boxes) > 0 and args.track and tracking_box is None:
            # Initialize tracking with the first detected box
            tracking_box = boxes[0:1, :].copy()
            print(f"Initialized tracking with box: {tracking_box[0, :4]}")
        elif len(boxes) > 0 and args.track and tracking_box is not None:
            # Find the best matching box based on IoU
            best_iou = 0
            best_idx = -1
            for i, box in enumerate(boxes[:, :4]):
                iou = compute_iou(tracking_box[0, :4], box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou > 0.5:  # IoU threshold for tracking
                tracking_box = boxes[best_idx:best_idx+1, :].copy()
                print(f"Updated tracking box with IoU {best_iou:.2f}: {tracking_box[0, :4]}")
    
    # Segmentation with SAM if boxes detected
    if len(boxes) > 0:
        predictor.set_image(img_cv2, image_format='BGR')
        
        # 处理单个边界框
        bb = torch.tensor(boxes[:, :4]).cuda()
        bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])  
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=bb,
            multimask_output=False
        )
        
        # 直接使用第一个（也是唯一一个）掩码
        mask = masks.cpu().squeeze()
        # 如果维度大于2，就再squeeze一次
        if len(mask.shape) > 2:
            mask = mask.squeeze(0)
        
        # 保存掩码
        mask_filename = f"{mask_folder}/{frame_idx+1:08d}.png"
        mask_np = mask.numpy().astype(np.uint8) * 255
        cv2.imwrite(mask_filename, mask_np)
        
        # 可视化结果
        vis_img = img_cv2.copy()
        vis_img[mask_np > 0] = vis_img[mask_np > 0] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
        # 绘制边界框
        box = boxes[0, :4]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(vis_img, pt1, pt2, (0, 255, 0), 2)
        
        vis_filename = f"{mask_folder}/{frame_idx+1:08d}_vis.jpg"
        cv2.imwrite(vis_filename, vis_img)
        
        print(f"Saved mask to {mask_filename} (using highest confidence box)")
    else:
        # Save empty mask
        mask_filename = f"{mask_folder}/{frame_idx+1:08d}.png"
        empty_mask = np.zeros((img_cv2.shape[0], img_cv2.shape[1]), dtype=np.uint8)
        cv2.imwrite(mask_filename, empty_mask)
        print(f"No {target_object} detected in frame, saved empty mask")

print(f"Processing completed. Masks saved to {mask_folder}") 