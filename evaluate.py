import os
import argparse
import numpy as np

import torch
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

def evaluate_model():
    parser = argparse.ArgumentParser(description="Evaluate Wav-KAN-SEUNet")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth model file')
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat', help='Wavelet type: mexican_hat, morlet, dog')
    parser.add_argument('--clean_dir', type=str, required=True, help='Path to clean test audio directory')
    parser.add_argument('--noisy_dir', type=str, required=True, help='Path to noisy test audio directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.batch_size

    print(f"Loading test dataset from {args.clean_dir} and {args.noisy_dir}...")
    dataset = VoiceBankDataset(args.clean_dir, args.noisy_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading model from {args.model_path} with wavelet type: {args.wavelet_type}")
    model = WavKAN_UNet(wavelet_type=args.wavelet_type).to(DEVICE)
    
    # Load model weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # 1. Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    macs = "N/A"
    params_str = "N/A"
    
    # 2. Calculate GFLOPs
    if get_model_complexity_info is not None:
        try:
            # STFT of 2s 16kHz audio typically results in (257, 251) depending on hop_length
            # Let's use (1, 257, 251) as dummy input shape for ptflops
            macs, params_str = get_model_complexity_info(model, (1, 257, 251), as_strings=True,
                                                    print_per_layer_stat=False, verbose=False)
        except Exception as e:
            print(f"Error computing GFLOPs with ptflops: {e}")
    else:
        print("ptflops not installed. Skipping GFLOPs calculation. Run `pip install ptflops` to enable.")

    all_pesq = []
    all_stoi = []
    all_csig = []
    all_cbak = []
    all_covl = []

    print(f"Evaluating on {DEVICE}...")
    with torch.no_grad():
        for noisy_spec, clean_spec, noisy_audio, clean_audio in tqdm(dataloader, desc="Evaluating"):
            noisy_spec = noisy_spec.to(DEVICE)
            
            # Forward pass
            mask = model(noisy_spec)
            pred_spec = mask * noisy_spec
            
            # Convert spectrograms back to audio to calculate metrics
            noisy_audio = noisy_audio.to(DEVICE)
            
            batch_pred_audio = []
            for i in range(pred_spec.shape[0]):
                # Reconstruct phase using the noisy audio phase
                mag, phase, _ = AudioUtils.audio_to_spec(noisy_audio[i])
                # Use predicted magnitude, original phase
                pred_mag_i = pred_spec[i, 0] # (F, T)
                # Reconstruct audio
                recon_audio = AudioUtils.spec_to_audio(pred_mag_i, phase.squeeze(0))
                batch_pred_audio.append(recon_audio.squeeze(0))
                
            batch_pred_audio = torch.stack(batch_pred_audio) # (B, Time)
            
            # Calculate metrics for the current batch
            mets = evaluate_batch(clean_audio.cpu(), batch_pred_audio.cpu())
            
            if 'PESQ' in mets and not np.isnan(mets['PESQ']): all_pesq.append(mets['PESQ'])
            if 'STOI' in mets and not np.isnan(mets['STOI']): all_stoi.append(mets['STOI'])
            if 'CSIG' in mets and not np.isnan(mets['CSIG']): all_csig.append(mets['CSIG'])
            if 'CBAK' in mets and not np.isnan(mets['CBAK']): all_cbak.append(mets['CBAK'])
            if 'COVL' in mets and not np.isnan(mets['COVL']): all_covl.append(mets['COVL'])

    # Aggregate results
    results = {
        'PESQ': np.mean(all_pesq) if len(all_pesq) > 0 else float('nan'),
        'STOI': np.mean(all_stoi) if len(all_stoi) > 0 else float('nan'),
        'CSIG': np.mean(all_csig) if len(all_csig) > 0 else float('nan'),
        'CBAK': np.mean(all_cbak) if len(all_cbak) > 0 else float('nan'),
        'COVL': np.mean(all_covl) if len(all_covl) > 0 else float('nan'),
    }

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total Parameters : {total_params:,}")
    if get_model_complexity_info is not None:
        print(f"GFLOPs (MACs)    : {macs}")
    print("-" * 40)
    print(f"PESQ  : {results['PESQ']:.4f}")
    print(f"STOI  : {results['STOI']:.4f}")
    print(f"CSIG  : {results['CSIG']:.4f}")
    print(f"CBAK  : {results['CBAK']:.4f}")
    print(f"COVL  : {results['COVL']:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()
