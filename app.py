import os
import sys
import cv2
import numpy as np
import torch
import gradio as gr
import shutil
from glob import glob
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing required libraries with error handling
try:
    from segment_anything import SamPredictor, sam_model_registry
    from pycocotools import mask as masktool
    from lib.pipeline import video2frames
    from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
except ImportError as e:
    logger.error(f"Error importing required libraries: {e}")
    logger.error("Make sure you have the required packages installed in your 'my' conda environment")
    logger.error("Required packages: segment_anything, pycocotools, detectron2")
    logger.error(traceback.format_exc())
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))

def find_pretrained_models():
    """Check if pretrained models exist and provide information if not"""
    sam_model_path = "data/pretrain/sam_vit_h_4b8939.pth"
    detector_cfg_path = "data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py"
    
    # Check if the data/pretrain directory exists
    if not os.path.exists("data/pretrain"):
        os.makedirs("data/pretrain", exist_ok=True)
        logger.error(f"Created data/pretrain directory, but model files are missing")
        return False, f"Missing model files. Please download pretrained models to {os.path.abspath('data/pretrain/')}"
    
    # Check if the SAM model exists
    if not os.path.exists(sam_model_path):
        logger.error(f"SAM model not found at {sam_model_path}")
        return False, f"Missing SAM model. Please download to {os.path.abspath(sam_model_path)}"
    
    # Check if the detector config exists
    if not os.path.exists(detector_cfg_path):
        logger.error(f"Detector config not found at {detector_cfg_path}")
        return False, f"Missing detector config. Please download to {os.path.abspath(detector_cfg_path)}"
    
    return True, "Pretrained models found"

def initialize_models():
    # Check for pretrained models
    models_ok, message = find_pretrained_models()
    if not models_ok:
        return message, None, None
    
    try:
        # Initialize SAM
        logger.info("Initializing SAM model...")
        sam_path = os.path.abspath("data/pretrain/sam_vit_h_4b8939.pth")
        sam = sam_model_registry["vit_h"](checkpoint=sam_path)
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            device = torch.device("cuda")
            _ = sam.to(device)
        else:
            logger.warning("CUDA not available, using CPU (this might be slow)")
            device = torch.device("cpu")
            _ = sam.to(device)
            
        sam_predictor = SamPredictor(sam)
        
        # Initialize ViTDet for detection
        logger.info("Initializing detector model...")
        cfg_path = os.path.abspath('data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py')
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        return "Models initialized successfully!", sam_predictor, detector
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        logger.error(traceback.format_exc())
        return f"Error initializing models: {str(e)}", None, None

def process_input(input_path, current_seq, img_folder, mask_folder, results_info):
    if not input_path:
        return None, "Please provide an input video or folder of images", current_seq, img_folder, mask_folder, results_info, None, None
    
    try:
        # Determine sequence name
        if os.path.isfile(input_path):
            # Input is a video file
            file = input_path
            seq = os.path.basename(file).split('.')[0]
        else:
            # Input is a directory
            file = None
            seq = os.path.basename(input_path)
        
        current_seq = seq
        
        # Create necessary folders
        seq_folder = f'results/{seq}'
        img_folder = f'{seq_folder}/images'
        mask_folder = f'{seq_folder}/masks'
        painted_folder = f'{seq_folder}/painted_image'
        
        os.makedirs(seq_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(painted_folder, exist_ok=True)
        
        # Process input
        output_video_path = f'{seq_folder}/tram_input.mp4'
        
        if file and os.path.isfile(file):
            # Extract frames from video
            logger.info(f"Extracting frames from {file}")
            nframes = video2frames(file, img_folder)
            results_info = f"Video processed: {file}\nFrames extracted: {nframes}\nSaved to: {img_folder}"
            
            # Get video info
            video = cv2.VideoCapture(file)
            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video.release()
            
            # Create a copy of the input video
            shutil.copy(file, output_video_path)
        else:
            # Copy images from folder
            logger.info(f"Processing images from folder {input_path}")
            img_files = sorted(glob(os.path.join(input_path, '*.jpg')) + 
                            glob(os.path.join(input_path, '*.png')))
            
            if not img_files:
                return None, "No image files found in the provided folder", current_seq, img_folder, mask_folder, results_info, None, None
            
            # Copy images to img_folder
            for i, img_file in enumerate(img_files):
                shutil.copy(img_file, f'{img_folder}/{i+1:08d}.jpg')
            
            # Get image dimensions from first image
            first_img = cv2.imread(img_files[0])
            height, width = first_img.shape[:2]
            
            # Create video from images
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            for img_file in img_files:
                img = cv2.imread(img_file)
                video_writer.write(img)
            
            video_writer.release()
            nframes = len(img_files)
            
            results_info = f"Images processed: {len(img_files)}\nSaved to: {img_folder}"
        
        # Update results info with video details
        results_info += f"\nVideo created: {output_video_path}\nResolution: {width}x{height}\nFrames: {nframes}\nFPS: {fps}"
        
        # Log info about the sequence
        logger.info(f"Processed sequence: {seq}")
        logger.info(f"Image folder: {img_folder}")
        logger.info(f"Number of frames: {nframes}")
        
        # 直接读取第一帧并转换为RGB格式
        first_frame_path = f'{img_folder}/{0+1:08d}.jpg'
        if os.path.exists(first_frame_path):
            first_frame = cv2.imread(first_frame_path)
            if first_frame is not None:
                # BGR到RGB的转换
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                return output_video_path, results_info, current_seq, img_folder, mask_folder, results_info, first_frame, gr.update(maximum=nframes)
        
        # 如果没有找到第一帧，则返回None
        return output_video_path, results_info, current_seq, img_folder, mask_folder, results_info, None, gr.update(maximum=nframes)
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        logger.error(traceback.format_exc())
        return None, f"Error processing input: {str(e)}", current_seq, img_folder, mask_folder, results_info, None

def get_frame(frame_index, seq_manager):
    """获取指定帧图像"""
    if seq_manager is None:
        logger.warning("无效的序列管理器")
        return None, None
    
    # 确保frame_index是整数
    frame_index = int(frame_index)
    logger.info(f"获取帧 {frame_index} 图像")
    
    img_folder = f'results/{seq_manager}/images'
    if not os.path.exists(img_folder):
        logger.warning(f"图像文件夹不存在: {img_folder}")
        return None, None
    
    # 文件名是从1开始的索引，与slider一致
    img_path = f"{img_folder}/{frame_index:08d}.jpg"
    logger.info(f"加载图像: {img_path}")
    
    if os.path.exists(img_path):
        # 读取图像并转换BGR到RGB
        image = cv2.imread(img_path)
        if image is not None:
            # 转换BGR到RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"成功加载图像: {img_path}")
            return image, img_path
        else:
            logger.warning(f"无法读取图像: {img_path}")
    else:
        logger.warning(f"图像不存在: {img_path}")
        
        # 尝试在图像文件夹中查找所有图像文件
        image_files = sorted(glob(f"{img_folder}/*.jpg"))
        if image_files:
            logger.info(f"找到 {len(image_files)} 个图像文件")
            logger.info(f"第一个图像文件: {image_files[0]}")
            if frame_index <= len(image_files):
                img_path = image_files[frame_index-1]  # 调整索引，从0到n-1
                logger.info(f"尝试加载替代图像: {img_path}")
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image, img_path
    
    return None, None

def detect_and_track(frame_index, target_object, min_size, exclude_person, current_seq, results_info, detector, sam_predictor):
    if not current_seq or detector is None or sam_predictor is None:
        return "Please initialize models and input first", None, results_info
    
    try:
        seq_folder = f'results/{current_seq}'
        img_folder = f'{seq_folder}/images'
        mask_folder = f'{seq_folder}/masks'
        painted_folder = f'{seq_folder}/painted_image'
        
        # Object ID mapping (same as in estimate_object.py)
        object_id_map = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
            'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
            'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11,
            'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15,
            'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19,
            'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23,
            'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27,
            'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31,
            'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35,
            'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39,
            'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43,
            'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47,
            'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51,
            'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55,
            'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59,
            'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
            'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67,
            'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71,
            'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75,
            'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79,
            'pikaqiu': 15, 'pikachu': 15,
        }
        
        target_object = target_object.lower()
        exclude_classes = []
        if exclude_person:
            exclude_classes.append(0)
        
        if target_object in object_id_map:
            target_object_id = object_id_map[target_object]
        else:
            target_object_id = None
        
        # Process all frames
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
        tracking_box = None
        
        logger.info(f"Processing {len(imgfiles)} frames for object detection and segmentation")
        
        # Storage for boxes, masks, and tracks for all frames
        all_boxes = []
        all_masks = []
        all_tracks = []
        
        for frame_idx, imgpath in enumerate(imgfiles):
            img_cv2 = cv2.imread(imgpath)
            
            # Detection with ViTDet
            with torch.no_grad():
                det_out = detector(img_cv2)
                det_instances = det_out['instances']
                
                # Filter detections
                if target_object_id is not None:
                    valid_idx = (det_instances.pred_classes == target_object_id) & (det_instances.scores > 0.25)
                else:
                    valid_idx = det_instances.scores > 0.25
                    if exclude_classes:
                        for class_id in exclude_classes:
                            valid_idx = valid_idx & (det_instances.pred_classes != class_id)
                
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                confs = det_instances.scores[valid_idx].cpu().numpy()
                classes = det_instances.pred_classes[valid_idx].cpu().numpy()
                boxes = np.hstack([boxes, confs[:, None]])
                
                # Filter by minimum size
                if min_size:
                    w = boxes[:, 2] - boxes[:, 0]
                    h = boxes[:, 3] - boxes[:, 1]
                    valid = np.stack([w, h]).max(axis=0) > min_size
                    boxes = boxes[valid]
                
                # Keep only the highest confidence detection
                if len(boxes) > 1:
                    highest_conf_idx = np.argmax(boxes[:, 4])
                    boxes = boxes[highest_conf_idx:highest_conf_idx+1]
            
            # Segmentation with SAM if boxes detected
            if len(boxes) > 0:
                sam_predictor.set_image(img_cv2, image_format='BGR')
                
                device = next(sam_predictor.model.parameters()).device
                bb = torch.tensor(boxes[:, :4]).to(device)
                bb = sam_predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])  
                masks, scores, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=bb,
                    multimask_output=False
                )
                
                mask = masks.cpu().squeeze()
                if len(mask.shape) > 2:
                    mask = mask.squeeze(0)
                
                # Save mask
                mask_filename = f"{mask_folder}/{frame_idx+1:08d}.png"
                mask_np = mask.numpy().astype(np.uint8) * 255
                cv2.imwrite(mask_filename, mask_np)
                
                # Create visualization
                vis_img = img_cv2.copy()
                vis_img[mask_np > 0] = vis_img[mask_np > 0] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
                
                # Draw bounding box
                box = boxes[0, :4]
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                cv2.rectangle(vis_img, pt1, pt2, (0, 255, 0), 2)
                
                # Save visualization
                vis_filename = f"{painted_folder}/{frame_idx+1:08d}.jpg"
                cv2.imwrite(vis_filename, vis_img)
                
                # Store boxes and masks for saving
                all_boxes.append(boxes[0])
                
                # Convert mask to COCO RLE format for saving
                h, w = mask_np.shape
                rle = masktool.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for saving
                all_masks.append(rle)
                
                # Simple tracking ID (1-indexed)
                track_id = 1
                all_tracks.append(track_id)
            else:
                # Save empty mask
                mask_filename = f"{mask_folder}/{frame_idx+1:08d}.png"
                empty_mask = np.zeros((img_cv2.shape[0], img_cv2.shape[1]), dtype=np.uint8)
                cv2.imwrite(mask_filename, empty_mask)
                
                # Save original image as visualization
                vis_filename = f"{painted_folder}/{frame_idx+1:08d}.jpg"
                cv2.imwrite(vis_filename, img_cv2)
                
                # Add empty data
                if frame_idx > 0 and len(all_boxes) > 0:
                    all_boxes.append(all_boxes[-1])  # Use previous box
                    all_masks.append(all_masks[-1])  # Use previous mask
                    all_tracks.append(all_tracks[-1])  # Use previous track
                else:
                    # Add dummy values if this is the first frame
                    dummy_box = np.array([0, 0, 10, 10, 0.1])  # x1,y1,x2,y2,conf
                    all_boxes.append(dummy_box)
                    
                    # Empty mask in RLE format
                    h, w = img_cv2.shape[:2]
                    empty_rle = masktool.encode(np.asfortranarray(empty_mask))
                    empty_rle['counts'] = empty_rle['counts'].decode('utf-8')
                    all_masks.append(empty_rle)
                    
                    # Dummy track ID
                    all_tracks.append(0)
        
        # Save boxes, masks, and tracks
        np.save(f'{seq_folder}/boxes.npy', np.array(all_boxes))
        np.save(f'{seq_folder}/masks.npy', np.array(all_masks))
        np.save(f'{seq_folder}/tracks.npy', np.array(all_tracks))
        
        # Update results info
        results_info += f"\n\nDetection and Segmentation completed:\nMasks saved to: {mask_folder}\nVisualization saved to: {painted_folder}"
        results_info += f"\nBoxes, masks, and tracks saved to: {seq_folder}"
        
        # Return the visualization for the selected frame
        vis_img_path = f"{painted_folder}/{frame_index:08d}.jpg"
        if os.path.exists(vis_img_path):
            return results_info, vis_img_path, results_info
        else:
            return results_info, None, results_info
    except Exception as e:
        logger.error(f"Error in detect_and_track: {e}")
        logger.error(traceback.format_exc())
        return f"Error during detection and tracking: {str(e)}", None, results_info

def run_masked_droid_slam(static_camera, current_seq, results_info):
    if not current_seq:
        return "Please initialize input first", None, results_info
    
    try:
        # Import required functions for camera calibration
        from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world
        
        # Set up paths
        seq_folder = f'results/{current_seq}'
        img_folder = f'{seq_folder}/images'
        camera_folder = f'{seq_folder}/camera'
        os.makedirs(camera_folder, exist_ok=True)
        
        # Check if necessary files exist
        masks_file = f'{seq_folder}/masks.npy'
        boxes_file = f'{seq_folder}/boxes.npy'
        tracks_file = f'{seq_folder}/tracks.npy'
        
        required_files = [masks_file, boxes_file, tracks_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}\n"
            error_msg += "Please run Detect+SAM+DEVA-Track-Anything first"
            logger.error(error_msg)
            return error_msg, None, results_info
        
        # Load masks and convert from RLE to binary format
        masks_ = np.load(masks_file, allow_pickle=True)
        
        logger.info("Running masked DROID-SLAM...")
        logger.info(f"Static camera mode: {static_camera}")
        
        # Convert masks from RLE to binary format
        masks = np.array([masktool.decode(m) for m in masks_])
        masks = torch.from_numpy(masks)
        
        # Get image files
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
        
        # Calibrate camera intrinsics
        logger.info("Calibrating camera intrinsics...")
        cam_int, is_static = calibrate_intrinsics(img_folder, masks, is_static=static_camera)
        
        # Run metric SLAM
        logger.info("Running metric SLAM...")
        cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int, is_static=is_static)
        
        # Align camera to world
        logger.info("Aligning camera to world coordinates...")
        wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)
        
        # Save camera parameters
        camera = {
            'pred_cam_R': cam_R.numpy(),
            'pred_cam_T': cam_T.numpy(),
            'world_cam_R': wd_cam_R.numpy(),
            'world_cam_T': wd_cam_T.numpy(),
            'img_focal': cam_int[0],
            'img_center': cam_int[2:],
            'spec_focal': spec_f
        }
        
        camera_path = f'{seq_folder}/camera.npy'
        np.save(camera_path, camera)
        
        # Update results info
        camera_info = "Static camera" if is_static else "Dynamic camera"
        results_info += f"\n\nRun masked DROID-SLAM with {camera_info}:\nCamera data saved to: {camera_path}"
        results_info += f"\nCamera intrinsics: focal={cam_int[0]:.2f}, center={cam_int[2:]}"
        results_info += f"\nDetected {'static' if is_static else 'moving'} camera"
        
        return results_info, None, results_info
    except Exception as e:
        logger.error(f"Error in run_masked_droid_slam: {e}")
        logger.error(traceback.format_exc())
        return f"Error running masked DROID-SLAM: {str(e)}", None, results_info

def sam_click_handler(img, points, point_labels, current_mask,
                      sam_predictor, current_image, current_seq, point_mode, evt: gr.SelectData):
    """Handler for segmentation clicks"""
    if sam_predictor is None:
        return img, "Please initialize models first", points, point_labels, current_mask
    
    if img is None and current_image is None:
        return img, "Please load an image first", points, point_labels, current_mask
    
    try:
        # Get the image to segment - either from the provided img or from the current_image path
        if img is not None:
            # Use the image directly if provided (numpy array from Gradio)
            current_image_np = img
            logger.info(f"Using provided image: shape={current_image_np.shape}, type={type(current_image_np)}")
        elif isinstance(current_image, str):
            # Load from file path if provided as string
            current_image_np = cv2.cvtColor(cv2.imread(current_image), cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image from path: {current_image}")
        else:
            return img, "Invalid image format", points, point_labels, current_mask
        
        # Get coordinates from event
        x = evt.index[0]
        y = evt.index[1]
        
        # Use the point_mode from the state
        label = point_mode
        logger.info(f"Adding {('negative', 'positive')[label]} point at ({x}, {y})")
        
        # Add new point to existing points
        points = points + [[x, y]]
        point_labels = point_labels + [label]
        
        # Run SAM with the updated points
        input_points = np.array(points)
        input_labels = np.array(point_labels)
        
        # Set the image in SAM - ensure format is RGB
        sam_predictor.set_image(current_image_np, image_format='RGB')
        
        # Generate mask
        masks, scores, _ = sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        
        # Get the mask result
        current_mask = masks[0]
        
        # Create visualization
        vis_img = current_image_np.copy()
        
        # Apply mask with transparency
        mask_np = current_mask.astype(np.uint8) * 255
        vis_img[mask_np > 0] = vis_img[mask_np > 0] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
        
        # Draw points with different colors for positive/negative
        for i, (px, py) in enumerate(points):
            color = (0, 255, 0) if point_labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis_img, (int(px), int(py)), 5, color, -1)
        
        # Save visualization temporarily
        temp_path = f"results/{current_seq}/temp_sam_vis.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        point_type = "Positive" if label == 1 else "Negative"
        return vis_img, f"Points: {len(points)} (Added {point_type} point at {x},{y})", points, point_labels, current_mask
        
    except Exception as e:
        logger.error(f"Error in sam_click_handler: {e}")
        logger.error(traceback.format_exc())  # Log the full traceback
        return img, f"Error adding point: {str(e)}", points, point_labels, current_mask


def clear_sam_points(current_image, current_seq):
    if current_image is None:
        return None, "Please load an image first", [], [], None
    
    try:
        current_image = cv2.imread(current_image)
        # Save original image temporarily
        temp_path = f"results/{current_seq}/temp_original.jpg"
        cv2.imwrite(temp_path, current_image)
        
        return temp_path, "Points cleared", [], [], None
    except Exception as e:
        logger.error(f"Error in clear_sam_points: {e}")
        return None, f"Error clearing points: {str(e)}", [], [], None

def save_object_mask(current_seq, current_mask, current_image):
    if current_seq is None or current_mask is None or current_image is None:
        return "Please create a mask first"
    
    try:
        current_image = cv2.imread(current_image)
        # Save mask
        mask_path = f"results/{current_seq}/object_mask.png"
        cv2.imwrite(mask_path, current_mask.astype(np.uint8) * 255)
        
        # Create object image with white background
        object_img = current_image.copy()
        white_bg = np.ones_like(object_img) * 255
        
        # Apply mask
        mask_3ch = np.stack([current_mask] * 3, axis=2)
        object_img = object_img * mask_3ch + white_bg * (1 - mask_3ch)
        
        # Save object image
        object_path = f"results/{current_seq}/object_ori.png"
        cv2.imwrite(object_path, object_img)
        
        return f"Mask saved to: {mask_path}\nObject saved to: {object_path}"
    except Exception as e:
        logger.error(f"Error in save_object_mask: {e}")
        return f"Error saving object mask: {str(e)}"

# Gradio interface
def create_interface():
    try:
        with gr.Blocks() as app:
            # State variables
            current_seq_state = gr.State(None)
            img_folder_state = gr.State(None)
            mask_folder_state = gr.State(None)
            sam_predictor_state = gr.State(None)
            # sam_predictor_state_2 = gr.State(None)
            detector_state = gr.State(None)
            points_state = gr.State([])
            point_labels_state = gr.State([])
            current_mask_state = gr.State(None)
            current_image_state = gr.State(None)
            results_info_state = gr.State("")
            point_mode_state = gr.State(1)       # 1=positive(默认)  0=negative
            
            gr.Markdown("# TRAM Video Processing Pipeline")
            
            # Input processing section
            with gr.Row():
                with gr.Column(scale=1):
                    input_path = gr.Textbox(label="Input Video or Image Folder Path")
                    
                    # Move video output here (before Process Input button)
                    video_output = gr.Video(label="Processed Video")
                    
                    process_button = gr.Button("Process Input")
                    
                    # Move initialize models button to after process input
                    init_button = gr.Button("Initialize Models")
                
                with gr.Column(scale=1):
                    # Combine initialization status and processing info
                    info_text = gr.Textbox(label="Processing Information", lines=12)
            

            
            # Frame selection and processing
            with gr.Row():
                with gr.Column():
                    # Update frame slider to show proper max value based on sequence
                    
                    frame_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Frame Selection")
                    frame_image = gr.Image(label="Selected Frame")
                    
                    with gr.Row():
                        detect_button = gr.Button("Detect+SAM+DEVA-Track-Anything")
                        target_object = gr.Textbox(label="Target Object", value="person")
                        exclude_person = gr.Checkbox(label="Exclude Person")
                    
                    with gr.Row():
                        droid_button = gr.Button("Run masked DROID-SLAM")
                        static_camera = gr.Checkbox(label="Static Camera")
                
                with gr.Column():
                    processed_image = gr.Image(label="Processed Frame")

           
            
            # SAM interactive segmentation
            gr.Markdown("## Interactive Segmentation")
            
            with gr.Accordion("Interactive Segmentation", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Replace buttons with Radio component
                        point_mode_radio = gr.Radio(
                            choices=["Positive", "Negative"],
                            value="Positive",
                            label="Point Type",
                            interactive=True
                        )
                        
                        # Map Radio values to point mode (1=positive, 0=negative)
                        def update_point_mode(value):
                            mode = 1 if value == "Positive" else 0
                            logger.info(f"Point mode set to: {value} ({mode})")
                            return mode
                        
                        point_mode_radio.change(
                            update_point_mode,
                            inputs=[point_mode_radio],
                            outputs=[point_mode_state]
                        )
                        
                        # Add a clear message
                        gr.Markdown("*Click points to segment objects. Positive (green) points mark object, Negative (red) points mark background.*")
                        
                        # Put SAM status and segmentation image in the same row
                        with gr.Row():
                            with gr.Column(scale=2):  # Give more space to the image
                                sam_image = gr.Image(
                                    label="Segment with clicks",
                                    type="numpy",
                                    interactive=True
                                )
                            
                            with gr.Column(scale=1):  # Less space for the status
                                sam_status = gr.Textbox(
                                    label="Status",
                                    value="Click Initialize button to start",
                                    interactive=False,
                                    lines=5
                                )
                        
                        with gr.Row():
                            clear_button = gr.Button("Clear Points")
                            save_button = gr.Button("Save Object Mask")
                        
                        # Connect SAM click handler
                        sam_image.select(
                            sam_click_handler,
                            inputs=[
                                sam_image, points_state, point_labels_state, current_mask_state,
                                sam_predictor_state, current_image_state, current_seq_state, point_mode_state
                            ],
                            outputs=[
                                sam_image, sam_status, points_state, point_labels_state, current_mask_state
                            ]
                        )

            # Connect to frame selection to load the same image in interactive mode
            frame_slider.change(
                lambda frame_idx, seq: (
                    logger.info(f"Frame slider changed to {frame_idx}"),
                    get_frame(frame_idx, seq)
                )[1],  # Return only the get_frame result
                inputs=[frame_slider, current_seq_state], 
                outputs=[frame_image, current_image_state]
            ).then(
                # Update sam_image to show current frame
                lambda img: (
                    logger.info(f"Updating segmentation image: {type(img)}"),
                    # Also clear points when frame changes
                    img
                ), 
                inputs=[current_image_state],
                outputs=[sam_image]
            ).then(
                # Clear points when frame changes
                lambda: ([], [], None),
                outputs=[points_state, point_labels_state, current_mask_state]
            )

            # Process input and initialize models events
            process_button.click(
                process_input, 
                inputs=[
                    input_path, 
                    current_seq_state, 
                    img_folder_state, 
                    mask_folder_state, 
                    results_info_state
                ], 
                outputs=[
                    video_output, 
                    info_text, 
                    current_seq_state, 
                    img_folder_state, 
                    mask_folder_state, 
                    results_info_state,
                    frame_image,
                    frame_slider
                ]
            )
 

            
            # Initialize models with proper state handling to prevent redundant initialization
            initialized = gr.State(False)
            
            def init_models(initialized_state, info):
                if initialized_state:
                    return "Models already initialized.", info, initialized_state, None, None
                else:
                    try:
                        status, sam_pred, detector = initialize_models()
                        new_info = f"{status}\n\n{info}" if info else status
                        return status, new_info, True, sam_pred, detector
                    except Exception as e:
                        logger.error(f"Error initializing models: {e}")
                        return f"Error initializing models: {str(e)}", info, False, None, None
            
            init_button.click(
                init_models,
                inputs=[
                    initialized,
                    info_text
                ],
                outputs=[
                    sam_status, # Display status in SAM status field
                    info_text,
                    initialized,
                    sam_predictor_state, 
                    detector_state
                ]
            )
            
            # Ensure current_image_state changes update sam_image
            current_image_state.change(
                lambda img: img,
                inputs=[current_image_state],
                outputs=[sam_image]
            )
            
            # Process buttons for object detection and tracking
            detect_button.click(
                detect_and_track, 
                inputs=[
                    frame_slider, 
                    target_object, 
                    gr.Number(value=100), 
                    exclude_person, 
                    current_seq_state, 
                    results_info_state, 
                    detector_state, 
                    sam_predictor_state
                ], 
                outputs=[info_text, processed_image, results_info_state]
            )
            
            droid_button.click(
                run_masked_droid_slam, 
                inputs=[static_camera, current_seq_state, results_info_state], 
                outputs=[info_text, processed_image, results_info_state]
            )
            
            # Clear points button
            clear_button.click(
                lambda image, seq: (image, "Points cleared", [], [], None),
                inputs=[current_image_state, current_seq_state],
                outputs=[sam_image, sam_status, points_state, point_labels_state, current_mask_state]
            )
            
            # Save object mask button
            save_button.click(
                save_object_mask, 
                inputs=[current_seq_state, current_mask_state, current_image_state], 
                outputs=sam_status
            )
            
            # Add example inputs
            gr.Examples(
                examples=[
                    ["example_video.mov"],
                    ["/nvme-ssd1/lizhe/code/tram/data/pikaqiu/"]
                ],
                inputs=input_path
            )
        
        return app
    except Exception as e:
        logger.error(f"Error creating interface: {e}")
        logger.error(traceback.format_exc())
        raise

# Create and launch the app
try:
    app = create_interface()

    if __name__ == "__main__":
        # Print parameter explanation
        print("\n\n" + "="*80)
        print("TRAM APP PARAMETER EXPLANATION")
        print("="*80)
        print("Detection Confidence Threshold: 0.25")
        print("  - This is the minimum confidence score (0.0-1.0) for an object to be detected")
        print("  - Higher values make detection more selective, lower values detect more objects")
        
        print("\nMinimum Size: 100 pixels")
        print("  - Objects smaller than this size (in pixels) will be filtered out")
        print("  - Increase to focus on larger objects, decrease to include smaller objects")
        
        print("\nPoint Coordinates (X, Y)")
        print("  - Pixel coordinates where you want to add points for SAM segmentation")
        print("  - Values should be within the image dimensions")
        
        print("\nPoint Types:")
        print("  - Positive: Indicates pixels that belong to the object (green points)")
        print("  - Negative: Indicates pixels that don't belong to the object (red points)")
        
        print("\nStatic Camera Checkbox")
        print("  - Enable if the camera doesn't move in the scene")
        print("  - Affects camera parameter calculation algorithms")
        print("  - Enables more precise 3D reconstruction for static scenes")
        print("="*80)
        
        logger.info("Starting Gradio app...")
        app.launch(debug=True, share=False)
except Exception as e:
    logger.error(f"Error starting app: {e}")
    logger.error(traceback.format_exc()) 