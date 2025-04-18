import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os
import argparse
import traceback

# 设置环境路径
sys.path.insert(0, os.path.dirname(__file__))

def test_specific_frame(img_path):
    """测试特定帧的ZoeDepth推理"""
    print(f"Testing ZoeDepth on image: {img_path}")
    
    # 验证文件存在
    if not os.path.exists(img_path):
        print(f"Error: Image file not found: {img_path}")
        return False
    

    # 加载ZoeDepth模型
    print("Loading ZoeDepth model...")
    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    model_zoe_n.eval()
    model_zoe_n = model_zoe_n.to('cuda')
    print("Model loaded successfully")
    
    # 读取图片
    print(f"Reading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to read image with OpenCV")
        return False
    
    print(f"Original image shape: {img.shape}, dtype: {img.dtype}")
    
    # BGR转RGB
    img_rgb = img[:,:,::-1].copy()
    print(f"RGB converted image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
    
    # 验证无无效值
    if np.isnan(img_rgb).any():
        print("Warning: Image contains NaN values")
    if np.isinf(img_rgb).any():
        print("Warning: Image contains Inf values")
    
    # 转换为PIL图像
    print("Converting to PIL image...")
    try:
        img_pil = Image.fromarray(img_rgb)
        print(f"PIL Image size: {img_pil.size}, mode: {img_pil.mode}")
    except Exception as e:
        print(f"Error converting to PIL image: {e}")
        print("Trying alternative conversion...")
        # 尝试转换为uint8类型
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
            print(f"Converted to uint8, new shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        img_pil = Image.fromarray(img_rgb)
    
    # 执行模型推理
    print("Running model inference...")

    # ZoeDepth内部使用的height和width值需要转换为Python内置int类型
    # 直接修改ZoeDepth源码可能更简单，但这里采用一个变通方法
    
    # 将PIL图像转为固定尺寸的图像，尺寸采用Python内置int
    target_width, target_height = 512, 384  # 使用ZoeDepth默认尺寸
    resized_img = img_pil.resize((target_width, target_height))
    print(f"Resized to fixed size: {resized_img.size}")
    
    # 使用调整大小后的图像进行推理
    pred_depth = model_zoe_n.infer_pil(resized_img)
    print("Inference successful!")
    print(f"Output depth shape: {pred_depth.shape}, dtype: {pred_depth.dtype}")
    print(f"Depth range: min={pred_depth.min()}, max={pred_depth.max()}")
    return True
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug ZoeDepth on a specific frame')
    parser.add_argument('--image', type=str, help='Path to specific image file')
    parser.add_argument('--folder', type=str, help='Path to images folder')
    parser.add_argument('--index', type=int, default=0, help='Frame index (if folder is provided)')
    
    args = parser.parse_args()
    
    if args.image:
        img_path = args.image
    elif args.folder:
        import glob
        imgfiles = sorted(glob.glob(f'{args.folder}/*.jpg'))
        if not imgfiles:
            print(f"No images found in folder: {args.folder}")
            sys.exit(1)
        if args.index >= len(imgfiles):
            print(f"Index {args.index} out of range. Found {len(imgfiles)} images.")
            sys.exit(1)
        img_path = imgfiles[args.index]
    else:
        # 默认路径
        img_path = "/nvme-ssd1/lizhe/code/tram/results/example_video/images/0000.jpg"
    
    success = test_specific_frame(img_path)
    
    if success:
        print("\nDebug test completed successfully")
    else:
        print("\nDebug test failed") 