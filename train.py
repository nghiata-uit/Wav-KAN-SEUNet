import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoiceBankDataset
from model import WavKAN_UNet
from utils import AudioUtils
from metrics import evaluate_batch

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

# ==========================================
# 4. Training Pipeline
# ==========================================
def train_model():
    parser = argparse.ArgumentParser(description="Train Wav-KAN-SEUNet")
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat', help='Wavelet type: mexican_hat, morlet, dog')
    parser.add_argument('--clean_dir', type=str, required=True, help='Path to clean train audio directory')
    parser.add_argument('--noisy_dir', type=str, required=True, help='Path to noisy train audio directory')
    parser.add_argument('--val_clean_dir', type=str, default=None, help='Path to clean val audio directory')
    parser.add_argument('--val_noisy_dir', type=str, default=None, help='Path to noisy val audio directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_file', type=str, default='training_log.csv', help='Name of the log file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint to resume training')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr

    # Tạo thư mục checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    log_file_path = os.path.join('checkpoints', args.log_file)

    print(f"Loading training dataset from {args.clean_dir} and {args.noisy_dir}...")
    dataset = VoiceBankDataset(args.clean_dir, args.noisy_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataloader = None
    if args.val_clean_dir and args.val_noisy_dir:
        print(f"Loading validation dataset from {args.val_clean_dir} and {args.val_noisy_dir}...")
        val_dataset = VoiceBankDataset(args.val_clean_dir, args.val_noisy_dir)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        print("Warning: Validation directories not provided. Validation metrics will not be computed.")

    model = WavKAN_UNet(wavelet_type=args.wavelet_type).to(DEVICE)
    
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Starting from scratch.")
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss()  # L1 Loss thường tốt cho spectrogram

    print(f"Bắt đầu train trên {DEVICE}...")

    # Tính toán GFLOPs
    if get_model_complexity_info is not None:
        try:
            # Tính toán GFLOPs với kích thước dummy input. STFT của 2s audio 16kHz thường là (257, 251)
            macs, params = get_model_complexity_info(model, (1, 257, 251), as_strings=True,
                                                    print_per_layer_stat=False, verbose=False)
            print(f"Model GFLOPs (MACs): {macs}, Params: {params}")
        except Exception as e:
            print(f"Error computing GFLOPs: {e}")
            macs, params = "N/A", "N/A"
    else:
        macs, params = "N/A", "N/A"
        print("ptflops not installed. Skipping GFLOPs calculation.")

    best_pesq = -1.0
    log_data = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # Lưu ý: dataloader giờ trả về 4 giá trị thay vì 2
        for noisy_spec, clean_spec, noisy_audio, clean_audio in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Evaluation phase
        val_pesq = float('nan')
        val_stoi = float('nan')
        val_csig = float('nan')
        val_cbak = float('nan')
        val_covl = float('nan')

        if val_dataloader is not None:
            model.eval()
            
            all_pesq = []
            all_stoi = []
            all_csig = []
            all_cbak = []
            all_covl = []
            
            with torch.no_grad():
                for noisy_spec, clean_spec, noisy_audio, clean_audio in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                    noisy_spec = noisy_spec.to(DEVICE)
                    
                    mask = model(noisy_spec)
                    pred_spec = mask * noisy_spec
                    
                    # Convert spectrograms back to audio to calculate metrics
                    # We need the phase of the noisy signal
                    pred_spec_np = pred_spec.squeeze(1).cpu().numpy() # (B, F, T)
                    noisy_audio = noisy_audio.to(DEVICE)
                    
                    batch_pred_audio = []
                    for i in range(pred_spec.shape[0]):
                        # Reconstruct phase using the noisy audio phase
                        mag, phase, _ = AudioUtils.audio_to_spec(noisy_audio[i])
                        # Use predicted magnitude, original phase
                        pred_mag_i = pred_spec[i, 0] # (F, T)
                        # Reconstruct
                        recon_audio = AudioUtils.spec_to_audio(pred_mag_i, phase.squeeze(0))
                        batch_pred_audio.append(recon_audio.squeeze(0))
                        
                    batch_pred_audio = torch.stack(batch_pred_audio) # (B, Time)
                    
                    # Tính metrics trên batch này
                    mets = evaluate_batch(clean_audio.cpu(), batch_pred_audio.cpu())
                    
                    if 'PESQ' in mets and not np.isnan(mets['PESQ']): all_pesq.append(mets['PESQ'])
                    if 'STOI' in mets and not np.isnan(mets['STOI']): all_stoi.append(mets['STOI'])
                    if 'CSIG' in mets and not np.isnan(mets['CSIG']): all_csig.append(mets['CSIG'])
                    if 'CBAK' in mets and not np.isnan(mets['CBAK']): all_cbak.append(mets['CBAK'])
                    if 'COVL' in mets and not np.isnan(mets['COVL']): all_covl.append(mets['COVL'])
            
            if len(all_pesq) > 0: val_pesq = np.mean(all_pesq)
            if len(all_stoi) > 0: val_stoi = np.mean(all_stoi)
            if len(all_csig) > 0: val_csig = np.mean(all_csig)
            if len(all_cbak) > 0: val_cbak = np.mean(all_cbak)
            if len(all_covl) > 0: val_covl = np.mean(all_covl)
            
            print(f"Validation Metrics - PESQ: {val_pesq:.3f}, STOI: {val_stoi:.3f}, "
                  f"CSIG: {val_csig:.3f}, CBAK: {val_cbak:.3f}, COVL: {val_covl:.3f}")
            
            # Checkpoint: Save model if PESQ is higher
            if not np.isnan(val_pesq) and val_pesq > best_pesq:
                print(f"PESQ improved from {best_pesq:.3f} to {val_pesq:.3f}. Saving best model...")
                best_pesq = val_pesq
                torch.save(model.state_dict(), os.path.join('checkpoints', 'model_best.pth'))
                torch.save(model.state_dict(), os.path.join('checkpoints', f'model_best_epoch_{epoch+1}.pth'))
        else:
            # If no validation set, just save the model at each epoch or keep the latest
            torch.save(model.state_dict(), os.path.join('checkpoints', 'latest_model.pth'))

        # Ghi log metrics
        epoch_log = {
            'Epoch': epoch + 1,
            'Train_Loss': avg_loss,
            'Val_PESQ': val_pesq,
            'Val_STOI': val_stoi,
            'Val_CSIG': val_csig,
            'Val_CBAK': val_cbak,
            'Val_COVL': val_covl,
            'GFLOPs': macs,
            'Params': params
        }
        log_data.append(epoch_log)
        pd.DataFrame(log_data).to_csv(log_file_path, index=False)

if __name__ == "__main__":
    train_model()
