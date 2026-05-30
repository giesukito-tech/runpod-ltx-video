# Sử dụng bản build chuẩn từ NVIDIA có sẵn CUDA 12.1 và Ubuntu xịn
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Cài đặt Python, FFMPEG hệ thống và các công cụ nền để xử lý video
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cấu hình thư mục làm việc bên trong Docker
WORKDIR /

# Copy danh sách thư viện vào và tiến hành cài đặt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy file chạy chính app.py của anh vào trong Docker
COPY app.py .

# Lệnh kích hoạt khi RunPod Serverless gọi worker thức dậy
CMD [ "python3", "-u", "/app.py" ]
