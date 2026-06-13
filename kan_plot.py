import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================================
# 1. CẤU HÌNH ĐỊNH DẠNG CHUẨN LUẬN VĂN KHOA HỌC
# ==========================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# ==========================================
# 2. ĐỌC DỮ LIỆU TỪ FILE CSV
# ==========================================
try:
    df = pd.read_csv('training_log.csv')
except FileNotFoundError:
    print("Không tìm thấy file 'training_log.csv'.")
    exit()

epochs = df['Epoch']
train_loss = df['Train_Loss']
val_pesq = df['Val_PESQ']

# ==========================================
# 3. VẼ ĐỒ THỊ 1: TRAIN LOSS
# ==========================================
fig_loss, ax_loss = plt.subplots(figsize=(8, 5))

color_loss = '#1f77b4'  # Xanh dương
ax_loss.plot(epochs, train_loss, color=color_loss, linestyle='-', linewidth=2,
             marker='o', markersize=4, label='Train Loss (MAE)')

ax_loss.set_xlabel('Epoch (Chu kỳ huấn luyện)', fontweight='bold')
ax_loss.set_ylabel('Train Loss', fontweight='bold')
ax_loss.grid(True, linestyle='--', alpha=0.6)
ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax_loss.legend(loc='upper right', frameon=True, edgecolor='black')

fig_loss.tight_layout()

# Lưu đồ thị Loss
fig_loss.savefig('train_loss_plot.png', dpi=300, format='png', bbox_inches='tight')
fig_loss.savefig('train_loss_plot.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.close(fig_loss) # Đóng figure để giải phóng bộ nhớ

# ==========================================
# 4. VẼ ĐỒ THỊ 2: VALIDATION PESQ
# ==========================================
fig_pesq, ax_pesq = plt.subplots(figsize=(8, 5))

color_pesq = '#d62728'  # Đỏ đậm
ax_pesq.plot(epochs, val_pesq, color=color_pesq, linestyle='--', linewidth=2,
             marker='s', markersize=4, label='Validation PESQ')

ax_pesq.set_xlabel('Epoch (Chu kỳ huấn luyện)', fontweight='bold')
ax_pesq.set_ylabel('PESQ Score', fontweight='bold')
ax_pesq.grid(True, linestyle='--', alpha=0.6)
ax_pesq.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax_pesq.legend(loc='lower right', frameon=True, edgecolor='black')

fig_pesq.tight_layout()

# Lưu đồ thị PESQ
fig_pesq.savefig('val_pesq_plot.png', dpi=300, format='png', bbox_inches='tight')
fig_pesq.savefig('val_pesq_plot.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.close(fig_pesq)

print("Đã xuất thành công 4 file:")
print("- train_loss_plot.png / .pdf")
print("- val_pesq_plot.png / .pdf")