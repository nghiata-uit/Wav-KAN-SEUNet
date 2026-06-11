import glob
import os

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from utils import AudioUtils


# ==========================================
# 3. Dataset & Utils (VoiceBank-DEMAND)
# ==========================================
class VoiceBankDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, sample_rate=16000, max_len_sec=2):
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.sr = sample_rate
        self.max_len = int(sample_rate * max_len_sec)  # Cắt đoạn 2s để train batch

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load audio
        clean_audio, sr = torchaudio.load(self.clean_files[idx])
        noisy_audio, sr = torchaudio.load(self.noisy_files[idx])

        # Resample nếu cần
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            clean_audio = resampler(clean_audio)
            noisy_audio = resampler(noisy_audio)

        # Cắt hoặc Pad độ dài cố định
        if clean_audio.shape[1] > self.max_len:
            start = torch.randint(0, clean_audio.shape[1] - self.max_len, (1,))
            clean_audio = clean_audio[:, start:start + self.max_len]
            noisy_audio = noisy_audio[:, start:start + self.max_len]
        else:
            padding = self.max_len - clean_audio.shape[1]
            clean_audio = F.pad(clean_audio, (0, padding))
            noisy_audio = F.pad(noisy_audio, (0, padding))

        # Chuyển sang Spectrogram
        clean_mag, _, _ = AudioUtils.audio_to_spec(clean_audio)
        noisy_mag, _, _ = AudioUtils.audio_to_spec(noisy_audio)

        # STFT output là (1, Freq, Time), đã có channel = 1
        return noisy_mag, clean_mag, noisy_audio, clean_audio
