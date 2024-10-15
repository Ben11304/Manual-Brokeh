import cv2
import torch
import numpy as np

import numpy as np
import pydensecrf.densecrf as dcrf

def apply_crf(image, depth):
    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)

    # Chuẩn bị U và pairwise
    U = np.stack([depth.flatten(), 1 - depth.flatten()], axis=0)
    U = -np.log(U + 1e-6)
    U = U.reshape((2, h * w))
    d.setUnaryEnergy(U)

    # Thêm tính năng màu sắc
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)

    # Thực hiện suy luận
    Q = d.inference(5)
    depth_crf = np.argmax(Q, axis=0).reshape((h, w))

    return depth_crf.astype('float32')




def load_model(device="cpu"):
    model_type = "DPT_Large"  # MiDaS v3 - Large (highest accuracy, slowest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device(device)
    midas.to(device)
    midas.eval()
    return midas, model_type

def depth_map(img, midas, model_type="DPT_Large", device="cpu"):
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # Chuẩn hóa bản đồ độ sâu
    depth_min = output.min()
    depth_max = output.max()
    depth_map = (output - depth_min) / (depth_max - depth_min)
    depth_map = 1.0 - depth_map 
    # depth_map_uint8 = (depth_map * 255).astype('uint8')
    # depth_map_smooth = cv2.bilateralFilter(depth_map_uint8, d=3, sigmaColor=75, sigmaSpace=75)
    # depth_map_smooth = depth_map_smooth.astype('float32') / 255.0

    # depth_map_uint8 = (depth_map * 255).astype('uint8')

    # # Áp dụng Gaussian Blur
    # depth_map_smooth = cv2.GaussianBlur(depth_map_uint8, (5, 5), 0)

    # # Chuyển đổi lại về dạng float và chuẩn hóa
    # depth_map_smooth = depth_map_smooth.astype('float32') / 255.0

    # # depth_processed=apply_crf(img, depth_map)
    return depth_map

def depth_transform(depth_map, focus_point):
    depth_map_transform = abs(depth_map - focus_point)
    depth_map_transform /= depth_map_transform.max()  # Chuẩn hóa về [0, 1]
    return depth_map_transform

def blur(depth_map, image, num_levels=5, focus_point=0.5, max_kernel_size=71):
    depth_map = depth_transform(depth_map, focus_point)
    # Tạo danh sách các ngưỡng
    levels = np.linspace(0, 1, num_levels + 1)
    # Tạo danh sách lưu các mặt nạ và độ mờ tương ứng
    masks = []
    blur_values = []

    for i in range(num_levels):
        # Xác định mặt nạ cho mức hiện tại
        lower = levels[i]
        upper = levels[i + 1]
        mask = cv2.inRange(depth_map, lower, upper)
        masks.append(mask)
        
        # Xác định giá trị độ mờ cho mức hiện tại (kernel size)
        kernel_size = int(1 + (max_kernel_size - 1) * (i / (num_levels - 1)))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Đảm bảo kernel_size là số lẻ
        blur_values.append(kernel_size)

    blurred_images = []

    for kernel_size in blur_values:
        # Áp dụng Gaussian blur với kernel_size
        if kernel_size >= 3:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            blurred = image.copy()
        blurred_images.append(blurred)

    # Khởi tạo ảnh kết quả với giá trị 0
    result = np.zeros_like(image)

    for mask, blurred in zip(masks, blurred_images):
        # Tạo mặt nạ 3 kênh
        mask_3ch = cv2.merge([mask, mask, mask])
        
        # Áp dụng mặt nạ lên ảnh làm mờ và thêm vào ảnh kết quả
        masked_blur = cv2.bitwise_and(blurred, mask_3ch)
        result = cv2.add(result, masked_blur)

    # Tạo mặt nạ tổng hợp từ các mặt nạ đã tạo
    combined_mask = np.zeros_like(depth_map, dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Tạo mặt nạ nghịch đảo cho vùng không được làm mờ
    inverse_mask = cv2.bitwise_not(combined_mask)
    inverse_mask_3ch = cv2.merge([inverse_mask, inverse_mask, inverse_mask])

    # Áp dụng mặt nạ nghịch đảo lên ảnh gốc và thêm vào kết quả
    masked_original = cv2.bitwise_and(image, inverse_mask_3ch)
    result = cv2.add(result, masked_original)

    return result
