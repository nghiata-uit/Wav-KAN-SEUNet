import numpy as np
import torch
import warnings

try:
    from pesq import pesq
except ImportError:
    print("Warning: pesq not installed. Run `pip install pesq`")
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    print("Warning: pystoi not installed. Run `pip install pystoi`")
    stoi = None

try:
    import pysepm
except ImportError:
    print("Warning: pysepm not installed. Run `pip install https://github.com/schmiph2/pysepm/archive/master.zip`")
    pysepm = None

def compute_metrics(clean_audio, enhanced_audio, fs=16000):
    """
    Computes PESQ, STOI, CSIG, CBAK, COVL metrics.
    clean_audio, enhanced_audio: numpy arrays of shape (T,)
    fs: sampling rate
    """
    # Validate inputs
    if len(clean_audio) != len(enhanced_audio):
        min_len = min(len(clean_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]

    metrics = {}

    # PESQ
    if pesq is not None:
        try:
            # PESQ requires WB for 16kHz
            # Handle potential pesq errors for silence
            metrics['PESQ'] = pesq(fs, clean_audio, enhanced_audio, 'wb')
        except Exception as e:
            metrics['PESQ'] = float('nan')
    else:
        metrics['PESQ'] = float('nan')

    # STOI
    if stoi is not None:
        try:
            metrics['STOI'] = stoi(clean_audio, enhanced_audio, fs, extended=False)
        except Exception as e:
            metrics['STOI'] = float('nan')
    else:
        metrics['STOI'] = float('nan')

    # CSIG, CBAK, COVL
    if pysepm is not None:
        try:
            # We catch warnings because pysepm might complain about singular matrices or log(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                csig, cbak, covl = pysepm.composite(clean_audio, enhanced_audio, fs)
            metrics['CSIG'] = csig
            metrics['CBAK'] = cbak
            metrics['COVL'] = covl
        except Exception as e:
            metrics['CSIG'] = float('nan')
            metrics['CBAK'] = float('nan')
            metrics['COVL'] = float('nan')
    else:
        metrics['CSIG'] = float('nan')
        metrics['CBAK'] = float('nan')
        metrics['COVL'] = float('nan')

    return metrics

def evaluate_batch(clean_batch, enhanced_batch, fs=16000):
    """
    Evaluates a batch of audio signals.
    clean_batch, enhanced_batch: torch Tensors of shape (B, T) or (B, 1, T)
    """
    batch_metrics = {'PESQ': [], 'STOI': [], 'CSIG': [], 'CBAK': [], 'COVL': []}
    
    clean_np = clean_batch.detach().cpu().numpy()
    enhanced_np = enhanced_batch.detach().cpu().numpy()

    for i in range(clean_np.shape[0]):
        c = clean_np[i].squeeze()
        e = enhanced_np[i].squeeze()
        
        mets = compute_metrics(c, e, fs)
        
        for k, v in mets.items():
            if not np.isnan(v):
                batch_metrics[k].append(v)
                
    # Return average metrics for the batch
    avg_metrics = {}
    for k, v in batch_metrics.items():
        if len(v) > 0:
            avg_metrics[k] = np.mean(v)
        else:
            avg_metrics[k] = 0.0
            
    return avg_metrics
