# Wav-KAN-SEUNet
Kết hợp wav-KAN vào UNet để giải bài toán khử nhiễu âm thanh (Speech Enhancement).

## 1. Cài đặt môi trường (Environment Setup)

Dự án yêu cầu Python (khuyến nghị >= 3.8). Thực hiện cài đặt các thư viện cần thiết thông qua file `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Lưu ý:** Thư viện `pysepm` được cài đặt trực tiếp từ GitHub để tính toán một số metrics nâng cao. Ngoài ra, bạn có thể cần cài đặt phiên bản `torch` và `torchaudio` sao cho phù hợp với phiên bản CUDA trên máy của bạn (nếu dùng GPU). Ví dụ:
```bash
# Thay thế URL tùy theo phiên bản CUDA (VD: cu118, cu121)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 2. Hướng dẫn chạy train mô hình (Training)

Để huấn luyện mô hình, bạn sử dụng script `train.py`. Bạn cần chỉ định đường dẫn đến thư mục chứa dữ liệu clean và noisy cho tập train. Khuyến nghị nên cung cấp thêm tập validation để mô hình có thể lưu lại checkpoint tốt nhất và áp dụng Early Stopping.

**Ví dụ lệnh chạy cơ bản:**
```bash
python train.py \
    --wavelet_type mexican_hat \
    --clean_dir path/to/train/clean \
    --noisy_dir path/to/train/noisy \
    --val_clean_dir path/to/val/clean \
    --val_noisy_dir path/to/val/noisy \
    --epochs 50 \
    --batch_size 8 \
    --lr 0.001
```

**Danh sách các tham số chính:**
- `--wavelet_type`: Loại wavelet sử dụng trong mạng KAN (`mexican_hat`, `morlet`, `dog`). Mặc định là `mexican_hat`.
- `--clean_dir`: Thư mục chứa audio gốc (clean) của tập train. *(Bắt buộc)*
- `--noisy_dir`: Thư mục chứa audio có nhiễu (noisy) của tập train. *(Bắt buộc)*
- `--val_clean_dir`: Thư mục chứa audio clean của tập validation.
- `--val_noisy_dir`: Thư mục chứa audio noisy của tập validation.
- `--epochs`: Số lượng epochs để train. Mặc định `10`.
- `--batch_size`: Kích thước batch. Mặc định `8`.
- `--lr`: Learning rate (tốc độ học). Mặc định `0.001`.
- `--checkpoint`: Đường dẫn tới file model `.pth` nếu bạn muốn resume training từ một checkpoint cũ.
- `--patience`: Số lượng epoch tối đa không có sự cải thiện trên độ đo Val_PESQ trước khi kích hoạt Early Stopping. Mặc định `10`.

*Lưu ý: Trong quá trình train, các checkpoint (như `model_best.pth`, `latest_model.pth`) và file log quá trình huấn luyện (`training_log.csv`) sẽ được tự động lưu vào thư mục `checkpoints/`.*

## 3. Đánh giá mô hình (Evaluate)

Để đánh giá mô hình đã train trên tập test (hoặc tập validation), bạn sử dụng script `evaluate.py`. Quá trình đánh giá sẽ tính toán các metrics quan trọng trong xử lý giọng nói: **PESQ, STOI, CSIG, CBAK, COVL** cùng với độ phức tạp của mô hình (**Parameters** và **GFLOPs**).

**Ví dụ lệnh chạy:**
```bash
python evaluate.py \
    --model_path checkpoints/model_best.pth \
    --wavelet_type mexican_hat \
    --clean_dir path/to/test/clean \
    --noisy_dir path/to/test/noisy \
    --batch_size 8
```

**Danh sách các tham số:**
- `--model_path`: Đường dẫn tới file model (`.pth`) đã train. *(Bắt buộc)*
- `--wavelet_type`: Loại wavelet sử dụng phải khớp với cấu hình lúc train. Mặc định `mexican_hat`.
- `--clean_dir`: Thư mục chứa audio clean của tập test. *(Bắt buộc)*
- `--noisy_dir`: Thư mục chứa audio noisy của tập test. *(Bắt buộc)*
- `--batch_size`: Kích thước batch khi thực hiện đánh giá. Mặc định `8`.
