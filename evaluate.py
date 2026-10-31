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
    parser.add_argument('--kan_type', type=str, default=None, choices=['wavelet', 'bspline'],
                        help='KAN type in bottleneck (auto-detect from checkpoint if not specified)')
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat', help='Wavelet type: mexican_hat, morlet, dog (used if kan_type=wavelet)')
    parser.add_argument('--bspline_grid', type=int, default=5, help='Grid size for B-spline KAN')
    parser.add_argument('--bspline_order', type=int, default=3, help='Spline order k for B-spline KAN')
    parser.add_argument('--clean_dir', type=str, required=True, help='Path to clean test audio directory')
    parser.add_argument('--noisy_dir', type=str, required=True, help='Path to noisy test audio directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.batch_size

    print(f"Loading test dataset from {args.clean_dir} and {args.noisy_dir}...")
    dataset = VoiceBankDataset(args.clean_dir, args.noisy_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    # Tự động phát hiện kan_type từ checkpoint
    kan_type = args.kan_type
    if kan_type is None:
        keys = list(state_dict.keys())
        # B-spline (pykan) thường có các key như base_fun.mask, spline_grid, ...
        if any('spline_grid' in k or 'coef' in k and 'kan' in k.lower() for k in keys):
            kan_type = 'bspline'
        elif any(k.startswith('bottleneck_kan.') or k.startswith('wav_kan.') for k in keys):
            kan_type = 'wavelet'
        else:
            kan_type = 'wavelet'

    print(f"Detected/using kan_type: {kan_type}")
    if kan_type == 'wavelet':
        print(f"Wavelet type: {args.wavelet_type}")
    else:
        print(f"B-spline config: grid={args.bspline_grid}, k={args.bspline_order}")

    model = WavKAN_UNet(
        kan_type=kan_type,
        wavelet_type=args.wavelet_type,
        grid_size=args.bspline_grid,
        spline_order=args.bspline_order
    ).to(DEVICE)

    # Tương thích ngược với checkpoint cũ
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('wav_kan.'):
            new_key = k.replace('wav_kan.', 'bottleneck_kan.', 1)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Strict load failed, trying strict=False: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Tính toán tham số
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = "N/A"

    if get_model_complexity_info is not None and kan_type != 'bspline':
        try:
            macs, _ = get_model_complexity_info(model, (1, 257, 251), as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        except Exception as e:
            print(f"Error computing GFLOPs with ptflops: {e}")
    elif kan_type == 'bspline':
        print("Skipping GFLOPs for bspline (pykan not supported by ptflops).")

    all_pesq, all_stoi, all_csig, all_cbak, all_covl = [], [], [], [], []

    print(f"Evaluating on {DEVICE}...")
    with torch.no_grad():
        for noisy_spec, clean_spec, noisy_audio, clean_audio in tqdm(dataloader, desc="Evaluating"):
            noisy_spec = noisy_spec.to(DEVICE)
            mask = model(noisy_spec)
            pred_spec = mask * noisy_spec
            noisy_audio = noisy_audio.to(DEVICE)

            batch_pred_audio = []
            for i in range(pred_spec.shape[0]):
                mag, phase, _ = AudioUtils.audio_to_spec(noisy_audio[i])
                pred_mag_i = pred_spec[i, 0]
                recon_audio = AudioUtils.spec_to_audio(pred_mag_i, phase.squeeze(0))
                batch_pred_audio.append(recon_audio.squeeze(0))

            batch_pred_audio = torch.stack(batch_pred_audio)
            mets = evaluate_batch(clean_audio.cpu(), batch_pred_audio.cpu())

            for k, lst in [('PESQ', all_pesq), ('STOI', all_stoi), ('CSIG', all_csig), ('CBAK', all_cbak), ('COVL', all_covl)]:
                if k in mets and not np.isnan(mets[k]):
                    lst.append(mets[k])

    results = {
        'PESQ': np.mean(all_pesq) if all_pesq else float('nan'),
        'STOI': np.mean(all_stoi) if all_stoi else float('nan'),
        'CSIG': np.mean(all_csig) if all_csig else float('nan'),
        'CBAK': np.mean(all_cbak) if all_cbak else float('nan'),
        'COVL': np.mean(all_covl) if all_covl else float('nan'),
    }

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"KAN type         : {kan_type}")
    if kan_type == 'wavelet':
        print(f"Wavelet type     : {args.wavelet_type}")
    else:
        print(f"B-spline         : grid={args.bspline_grid}, k={args.bspline_order}")
    print(f"Total Parameters : {total_params:,}")
    if macs != "N/A":
        print(f"GFLOPs (MACs)    : {macs}")
    print("-" * 40)
    print(f"PESQ  : {results['PESQ']:.4f}")
    print(f"STOI  : {results['STOI']:.4f}")
    print(f"CSIG  : {results['CSIG']:.4f}")
    print(f"CBAK  : {results['CBAK']:.4f}")
    print(f"COVL  : {results['COVL']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_model()
