import os

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoiceBankDataset
from model import WavKAN_UNet
from utils import AudioUtils


# ==========================================
# 4. Training Pipeline
# ==========================================
def train_model():
    # Cấu hình
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 8  # Tùy chỉnh theo VRAM
    EPOCHS = 10
    LR = 0.001

    # --- DUMMY DATA ---
    # (Để bạn chạy thử ngay mà không cần tải dataset 2GB)
    # Nếu có dataset thật, hãy thay đổi paths bên dưới
    print("Đang tạo Dummy Data để kiểm tra luồng...")
    os.makedirs('dummy_data/clean', exist_ok=True)
    os.makedirs('dummy_data/noisy', exist_ok=True)
    for i in range(20):
        # Tạo sóng sin + nhiễu
        sr = 16000
        t = torch.linspace(0, 2, sr * 2)
        clean = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        noise = torch.randn_like(clean) * 0.5
        noisy = clean + noise
        torchaudio.save(f'dummy_data/clean/file_{i}.wav', clean, sr)
        torchaudio.save(f'dummy_data/noisy/file_{i}.wav', noisy, sr)

    clean_dir = 'dummy_data/clean'
    noisy_dir = 'dummy_data/noisy'
    # ------------------

    dataset = VoiceBankDataset(clean_dir, noisy_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = WavKAN_UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss()  # L1 Loss thường tốt cho spectrogram

    print(f"Bắt đầu train trên {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for noisy_spec, clean_spec in tqdm(dataloader):
            noisy_spec = noisy_spec.to(DEVICE)  # (B, 1, F, T)
            clean_spec = clean_spec.to(DEVICE)

            # Forward
            mask = model(noisy_spec)

            # Predicted Clean Spectrogram = Mask * Noisy
            pred_spec = mask * noisy_spec

            # Loss
            loss = criterion(pred_spec, clean_spec)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    # Test Inference thử 1 mẫu
    print("\nĐang thử nghiệm Inference...")
    model.eval()
    with torch.no_grad():
        sample_noisy, sample_clean = dataset[0]
        input_tensor = sample_noisy.unsqueeze(0).to(DEVICE)  # (1, 1, F, T)

        mask = model(input_tensor)
        pred_mag = mask * input_tensor

        # Để tái tạo âm thanh, ta cần Phase gốc của tín hiệu nhiễu
        # (Load lại file để lấy phase - bước này chỉ minh họa logic)
        path = dataset.noisy_files[0]
        audio, _ = torchaudio.load(path)
        # Cắt ngắn như trong dataset __getitem__
        audio = audio[:, :32000]  # Giả sử max_len 2s

        # Lấy phase gốc
        _, phase, _ = AudioUtils.audio_to_spec(audio.to(DEVICE))

        # Lưu ý: pred_mag đang có shape (1, 1, F, T), cần squeeze
        pred_mag = pred_mag.squeeze(0).squeeze(0)

        # Reconstruct
        # Cần resize phase nếu kích thước bị đổi do padding của mạng (nếu có)
        clean_audio_recon = AudioUtils.spec_to_audio(pred_mag, phase)

        torchaudio.save('output_denoised.wav', clean_audio_recon.cpu(), 16000)
        print("Đã lưu file kết quả: output_denoised.wav")


if __name__ == "__main__":
    train_model()
