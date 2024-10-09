from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import utilis
import torch
import uuid

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

# Trang chủ: Form tải lên ảnh
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
            # Lưu tệp tải lên
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('select_focus', filename=filename))
    return render_template('index.html')

# Trang chọn điểm lấy nét
@app.route('/select_focus/<filename>', methods=['GET', 'POST'])
def select_focus(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    if request.method == 'POST':
        x = int(request.form['x'])
        y = int(request.form['y'])
        max_kernel_size = int(request.form.get('max_kernel_size', 71))
        num_levels = int(request.form.get('num_levels', 5))

        # Đọc ảnh và tính toán bản đồ độ sâu
        img = cv2.imread(filepath)
        depth_map = utilis.depth_map(img, midas, model_type, device)

        # Lấy giá trị độ sâu tại điểm (x, y)
        focus_point = depth_map[y, x]

        # Áp dụng làm mờ ảnh
        result_img = utilis.blur(depth_map, img, num_levels=num_levels, focus_point=focus_point, max_kernel_size=max_kernel_size)

        # Lưu ảnh kết quả
        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, result_img)

        return render_template('result.html', result_image=result_filename)

    return render_template('select_focus.html', uploaded_image=filename)

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
