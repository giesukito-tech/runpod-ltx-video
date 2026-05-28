# Sử dụng Base Image chứa sẵn môi trường CUDA 12.1 và Python 3.10 chính thức từ NVIDIA
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Thiết lập chế độ cài đặt không tương tác để tránh bị dừng hỏi xác nhận giữa chừng
ENV DEBIAN_FRONTEND=noninteractive

# Cập nhật hệ thống và cài đặt các công cụ cốt lõi (Git để kéo repo, FFmpeg để xử lý video)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Nâng cấp công cụ cài đặt pip lên phiên bản mới nhất
RUN pip3 install --no-cache-dir --upgrade pip

# Cài đặt PyTorch và các thư viện xử lý đồ họa tương thích sâu với driver CUDA 12.1 của RunPod
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài đặt bộ công cụ điều khiển Serverless của RunPod và các thư viện AI từ Hugging Face
RUN pip3 install --no-cache-dir runpod transformers diffusers accelerate sentencepiece

# 🚀 TRIỂN KHAI REPO GỐC: Kéo mã nguồn LTX-Video chính thức từ GitHub của Lightricks về máy ảo
RUN git clone https://github.com/Lightricks/LTX-Video.git /app/LTX-Video

# Di chuyển thư mục làm việc vào thẳng bên trong kho chứa của LTX-Video
WORKDIR /app/LTX-Video

# Tiến hành cài đặt Repo LTX-Video dưới dạng một thư viện hệ thống (Editable mode)
RUN pip3 install -e .

# Sao chép file app.py (Trạm gác nhận lệnh API) anh vừa tạo trên GitHub vào đúng thư mục chạy
COPY app.py /app/LTX-Video/app.py

# Ra lệnh cho container khởi chạy luồng Python xử lý Serverless ngay khi RunPod cấp phát GPU
CMD ["python3", "-u", "app.py"]
