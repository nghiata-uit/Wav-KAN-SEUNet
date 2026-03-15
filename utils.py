import torch

class AudioUtils:
    @staticmethod
    def audio_to_spec(audio, n_fft=512, hop_length=128):
        # audio: (1, samples)
        window = torch.hann_window(n_fft).to(audio.device)
        stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                          window=window, return_complex=True)
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        return mag, phase, stft

    @staticmethod
    def spec_to_audio(mag, phase, n_fft=512, hop_length=128):
        # Reconstruct complex spectrogram
        complex_spec = mag * torch.exp(1j * phase)
        window = torch.hann_window(n_fft).to(mag.device)
        audio = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, window=window)
        return audio