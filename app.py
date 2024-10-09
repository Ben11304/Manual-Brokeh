from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import utilis
import torch
import uuid
from PIL import Image
import pillow_heif
from shutil import copyfile

app = Flask(__name__)

# Thư mục lưu trữ tệp tải lên và kết quả
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Tải mô hình MiDaS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas, model_type = utilis.load_model(device)

# Biến cache để lưu trữ depth_map và img
cache = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra xem tệp được tải lên hay không
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Lấy phần mở rộng của tệp
            filename = str(uuid.uuid4())
            img_extension = os.path.splitext(file.filename)[1].lower()

            if img_extension == '.heic' or img_extension == '.heif':
                # Đọc tệp bằng pillow_heif
                heif_file = pillow_heif.read_heif(file.read())
                image_pil = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                )
                # Chuyển đổi và lưu dưới dạng JPEG
                filename = filename + '.jpg'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_pil.save(filepath, "JPEG")
                img_original = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            else:
                # Lưu tệp tải lên
                filename = filename + img_extension
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Đọc ảnh bằng OpenCV
                img_original = cv2.imread(filepath)

            # Kiểm tra ảnh có được đọc thành công không
            if img_original is None:
                return "Không thể đọc ảnh.", 400

            # Tính toán depth_map và lưu vào cache
            depth_map = utilis.depth_map(img_original, midas, model_type, device)
            cache[filename] = {'depth_map': depth_map, 'img': img_original}

            # Lưu ảnh gốc vào thư mục results với tên result_<filename>
            result_filename = 'result_' + filename
            result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_filepath, img_original)

            return redirect(url_for('select_focus', filename=filename))
    return render_template('index.html')

@app.route('/select_focus/<filename>', methods=['GET', 'POST'])
def select_focus(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    return render_template('select_focus.html', uploaded_image=filename)

@app.route('/process_click', methods=['POST'])
def process_click():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        x = int(data.get('x', -1))
        y = int(data.get('y', -1))
        max_kernel_size = int(data.get('max_kernel_size', 71))
        num_levels = int(data.get('num_levels', 5))
        filename = data.get('filename', '')

        if filename == '':
            return jsonify({'error': 'Filename not provided'}), 400

        # Kiểm tra xem đã có trong cache chưa
        if filename in cache:
            depth_map = cache[filename]['depth_map']
            img_original = cache[filename]['img']
        else:
            return jsonify({'error': 'Depth map not found in cache.'}), 400

        # Kiểm tra tọa độ hợp lệ
        h, w = depth_map.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return jsonify({'error': 'Invalid coordinates'}), 400

        # Lấy giá trị độ sâu tại điểm (x, y)
        focus_point = depth_map[y, x]

        # Áp dụng làm mờ ảnh trên ảnh gốc
        result_img = utilis.blur(depth_map, img_original, num_levels=num_levels, focus_point=focus_point, max_kernel_size=max_kernel_size)

        # Lưu ảnh kết quả vào thư mục results
        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, result_img)

        # Không cập nhật ảnh trong cache để giữ nguyên ảnh gốc

        # Trả về đường dẫn tới ảnh kết quả
        return jsonify({'result_image': url_for('static', filename='results/' + result_filename)})
    except Exception as e:
        # Ghi log lỗi nếu cần
        print(f"Error in process_click: {e}")
        return jsonify({'error': f'Đã xảy ra lỗi trong quá trình xử lý: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
