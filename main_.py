import utilis_
import cv2
import os

# def get_all_image_files(root_folder):
#     # Các định dạng ảnh thường gặp
#     image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff','.JPG'}
#     image_files = []
    
#     # Sử dụng os.walk để đệ quy qua tất cả các thư mục và file
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             # Kiểm tra xem file có phải là file ảnh không
#             if os.path.splitext(filename)[1].lower() in image_extensions:
#                 image_files.append(os.path.join(dirpath, filename))
#     return image_files

# image_paths=get_all_image_files("./image")
# model=utilis.load_model("cpu")


# for image in image_paths:
#     depth_map=utilis.depth_map(image,model,"cpu")
#     result=utilis.blur(depth_map,image,8,1)
#     name=image.split('/')[-1]
#     name=name.split('.')[0]
#     cv2.imwrite(f'ouput/{name}.jpg', result)




def main():
    image_path = './image/Image 5.jpeg'
    device = "cpu"
    midas = utilis_.load_model(device)
    depth_map_data = utilis_.depth_map(image_path, midas, device)
    window_name = 'Depth Blur'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Max Kernel Size', window_name, 31, 101, lambda x: None)
    cv2.createTrackbar('Num Levels', window_name, 2, 20, lambda x: None)

    num_levels = 10
    max_kernel_size = 51
    focus_point = 0.5  
    updated= True
    def mouse_callback(event, x, y, flags, param):
        nonlocal focus_point, updated
        if event == cv2.EVENT_LBUTTONDOWN:
            # Lấy giá trị độ sâu tại điểm được nhấp
            focus_point = depth_map_data[y, x]
            print(f'Focus point updated to: {focus_point:.4f}')
            updated = True  # Đánh dấu cần cập nhật ảnh
    cv2.setMouseCallback(window_name, mouse_callback)
    while True:
        max_kernel_trackbar = cv2.getTrackbarPos('Max Kernel Size', window_name)
        num_levels = cv2.getTrackbarPos('Num Levels', window_name)
        max_kernel_size = max_kernel_trackbar
        if max_kernel_size % 2 == 0:
            max_kernel_size += 1 
        if max_kernel_size < 3:
            max_kernel_size = 3  
        if num_levels < 1:
            num_levels = 1  


        # tính năng click điểm focus
        if updated:
            result = utilis_.blur(depth_map_data, image_path, num_levels=num_levels, focus_point=focus_point, max_kernel_size=max_kernel_size)
            updated = False 
        cv2.imshow(window_name, result)

        # Nhấn phím 'q' để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

