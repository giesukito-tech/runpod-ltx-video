import runpod
import torch
import os
import base64

# Bọc lót an toàn tuyệt đối để xử lý dứt điểm lỗi ImportError
try:
    from diffusers import LTXVideoPipeline
except ImportError:
    print("⚠️  [Cảnh báo] Không tìm thấy LTXVideoPipeline ở package tổng, tiến hành nạp trực tiếp từ module con...")
    from diffusers.pipelines.ltx_video.pipeline_ltx_video import LTXVideoPipeline

from diffusers.utils import export_to_video

# Biến toàn cục để giữ Model trong bộ nhớ VRAM (Phục vụ cho các lượt gọi Hot Start tiếp theo)
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("📥 [RunPod] Đang tải mô hình LTX-Video từ Hugging Face vào VRAM...")
        
        # Chỉ định ID mô hình chính thức của Lightricks
        model_id = "Lightricks/LTX-Video"
        
        # Cấu hình tối ưu để mô hình chạy mượt và nét trên các dòng card xịn (RTX 4090, L40S)
        pipe = LTXVideoPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        print("🚀 [RunPod] Đã nạp mô hình LTX-Video thành công!")

def handler(job):
    """ Hàm trạm gác tự động kích hoạt khi có request API gửi lên từ n8n/Python """
    global pipe
    
    # Lấy các tham số cấu hình đầu vào do người dùng truyền lên
    job_input = job.get('input', {})
    prompt = job_input.get('prompt', 'A beautiful cinematic animation of wildlife')
    negative_prompt = job_input.get('negative_prompt', 'low quality, blurry, worst quality, distorted')
    
    # Tự động đồng bộ các thông số kích thước, số khung hình, số bước khử nhiễu
    width = int(job_input.get('width', 704))
    height = int(job_input.get('height', 480))
    num_frames = int(job_input.get('num_frames', 24))
    steps = int(job_input.get('steps', 25))
    guidance_scale = float(job_input.get('guidance_scale', 3.0))
    seed = job_input.get('seed', 42)
    
    # Đảm bảo mô hình đã được nạp sẵn vào card đồ họa
    load_model()
    
    # Thiết lập seed để kiểm soát tính ngẫu nhiên của khung hình
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"🎬 [RunPod] Đang render video với câu lệnh: '{prompt}'...")
    
    # Kích hoạt bộ não LTX-Video tính toán hình ảnh
    with torch.inference_mode():
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np" # Xuất ra định dạng mảng numpy để đóng gói thành video
        ).frames[0]
        
    # Đường dẫn file tạm trên Linux để chứa video trước khi mã hóa
    temp_video_path = "/tmp/output_result.mp4"
    
    # Xuất các khung hình đã render thành file video .mp4 hoàn chỉnh (mặc định 8 khung hình/giây)
    export_to_video(video_frames, output_video_path=temp_video_path, fps=8)
    
    print("📦 [RunPod] Render xong! Đang mã hóa file video sang định dạng chuỗi Base64...")
    
    # Đọc file video nhị phân và chuyển đổi thành chuỗi Base64 để bắn qua cổng API
    with open(temp_video_path, "rb") as video_file:
        encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
        
    # Dọn dẹp file tạm trên hệ thống để tránh đầy ổ cứng của server
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    print("✨ [RunPod] Đóng gói thành công! Đang gửi trả dữ liệu về máy Client...")
    
    # Trả về cấu trúc JSON chuẩn có chứa video dạng mã hóa cho đầu nhận
    return {"video": encoded_string}

if __name__ == "__main__":
    # Kích hoạt lắng nghe cổng RunPod Serverless
    runpod.serverless.start({"handler": handler})
