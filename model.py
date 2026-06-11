import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Wav-KAN Layer (Lõi Wavelet)
# ==========================================

class WavKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, wavelet_type='mexican_hat'):
        super(WavKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Các tham số có thể học được: Weights, Translation (dịch), Dilation (co giãn)
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.translation = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.dilation = nn.Parameter(torch.rand(input_dim, output_dim) * 0.1 + 0.5)  # Khởi tạo dương

        self.wavelet_type = wavelet_type

    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        # Broadcasting để tính toán song song
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.output_dim)  # (B, In, Out)

        translation = self.translation.unsqueeze(0)  # (1, In, Out)
        dilation = self.dilation.unsqueeze(0)  # (1, In, Out)

        # Chuẩn hóa đầu vào cho wavelet: t = (x - b) / a
        t = (x_expanded - translation) / (dilation + 1e-5)  # Tránh chia cho 0

        # Chọn loại Wavelet
        if self.wavelet_type == 'mexican_hat':
            # Mexican Hat: (1 - t^2) * exp(-t^2 / 2)
            basis = (1 - t ** 2) * torch.exp(-t ** 2 / 2)
        elif self.wavelet_type == 'morlet':
            # Morlet (phiên bản thực): cos(5t) * exp(-t^2 / 2)
            basis = torch.cos(5 * t) * torch.exp(-t ** 2 / 2)
        else:
            # Gaussian Derivative (DOG) đơn giản
            basis = -t * torch.exp(-t ** 2 / 2)

        # Tương tự phép nhân ma trận nhưng có basis function
        # y = sum(w * basis)
        y = torch.sum(basis * self.weights.unsqueeze(0), dim=1)  # (B, Out)

        return y


class ConvBlock(nn.Module):
    """Khối Convolution cơ bản cho Encoder/Decoder"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 2. Wav-KAN U-Net Architecture
# ==========================================

class WavKAN_UNet(nn.Module):
    def __init__(self, wavelet_type='mexican_hat'):
        super(WavKAN_UNet, self).__init__()

        # --- Encoder (Downsampling) ---
        self.enc1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck với Wav-KAN ---
        # Tại đây kích thước feature map đã nhỏ, ta dùng Wav-KAN để học global context
        # Giả sử input ảnh Spectrogram chuẩn hóa về kích thước chia hết cho 8 (ví dụ 256x256 -> 32x32)
        self.bottleneck_conv = nn.Conv2d(128, 256, 3, padding=1)

        # Wav-KAN Bridge: Xử lý thông tin channels
        # Thay vì chỉ dùng Dense Layer thường, ta dùng WavKANLayer
        self.wav_kan = WavKANLayer(256, 256, wavelet_type=wavelet_type)

        # --- Decoder (Upsampling) ---
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 256 do nối skip connection (128+128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Output layer: Dự đoán Mask (giá trị từ 0 đến 1)
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, 1, F, T)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck_conv(p3)  # (B, 256, F/8, T/8)

        # Apply Wav-KAN trên chiều channel (như 1x1 Conv nhưng xịn hơn)
        # Cần reshape: (B, C, H, W) -> (B * H * W, C)
        batch, c, h, w = b.shape
        b_flat = b.permute(0, 2, 3, 1).contiguous().view(-1, c)

        # WavKAN Transform
        b_processed = self.wav_kan(b_flat)

        # Reshape lại
        b_out = b_processed.view(batch, h, w, c).permute(0, 3, 1, 2)

        # Decoder
        u3 = self.up3(b_out)
        # Crop hoặc Pad nếu kích thước không khớp do padding lẻ (đơn giản hóa ở đây giả sử khớp)
        if u3.shape != e3.shape:
            u3 = F.interpolate(u3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.shape != e2.shape:
            u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape != e1.shape:
            u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Output Mask
        mask = self.sigmoid(self.final(d1))
        return mask
