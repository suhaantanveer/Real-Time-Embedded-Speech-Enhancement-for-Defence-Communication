"""
train.py
Training pipeline for TinySpeechEnhancer:
Trains the tiny causal GRU model on generated noisy-clean speech pairs.
"""

import os
import argparse
import torch
import torch.optim as optim
from model import TinySpeechEnhancer
from features import FeatureExtractor
from losses import CompositeEnhancementLoss, compute_si_snr
from dataset_loader import get_dataloaders

def train(manifest_path: str, epochs: int = 15, batch_size: int = 4, lr: float = 1e-3, 
          output_dir: str = "checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training TinySpeechEnhancer on device: {device}")
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size)
    if len(train_loader) == 0:
        print("[ERROR] No training samples found. Run mix_dataset.py first!")
        return
        
    # 2. Instantiate Model & Components
    model = TinySpeechEnhancer().to(device)
    extractor = FeatureExtractor()
    criterion = CompositeEnhancementLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"[INFO] Model Parameters: {model.get_parameter_count():,}")
    
    best_val_sisnr = -999.0
    best_model_path = os.path.join(output_dir, "best_tiny_enhancer.pth")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_sisnr = 0.0
        
        for noisy_audio, clean_audio in train_loader:
            noisy_audio = noisy_audio.to(device)
            clean_audio = clean_audio.to(device)
            target_len = noisy_audio.shape[-1]
            
            optimizer.zero_grad()
            
            # Extract STFT and 22-band log energies
            log_energies, stft_complex = extractor.extract_features(noisy_audio)
            
            # Neural network predicts 22-band spectral gains
            gains, _ = model(log_energies)
            
            # Reconstruct enhanced audio via iSTFT
            est_audio = extractor.apply_gains_and_istft(stft_complex, gains, target_len=target_len)
            
            # Compute composite loss
            loss, sisnr = criterion(est_audio, clean_audio)
            loss.backward()
            
            # Gradient clipping for recurrent stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_sisnr += sisnr.item()
            
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_sisnr = total_train_sisnr / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_sisnr_sum = 0.0
        with torch.no_grad():
            for noisy_audio, clean_audio in val_loader:
                noisy_audio = noisy_audio.to(device)
                clean_audio = clean_audio.to(device)
                target_len = noisy_audio.shape[-1]
                
                log_energies, stft_complex = extractor.extract_features(noisy_audio)
                gains, _ = model(log_energies)
                est_audio = extractor.apply_gains_and_istft(stft_complex, gains, target_len=target_len)
                
                sisnr = torch.mean(compute_si_snr(est_audio, clean_audio))
                val_sisnr_sum += sisnr.item()
                
        avg_val_sisnr = val_sisnr_sum / max(1, len(val_loader))
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:7.3f} | Train SI-SNR: {avg_train_sisnr:+6.2f} dB | Val SI-SNR: {avg_val_sisnr:+6.2f} dB")
        
        # Save best checkpoint
        if avg_val_sisnr > best_val_sisnr:
            best_val_sisnr = avg_val_sisnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_sisnr': avg_val_sisnr,
                'optimizer_state': optimizer.state_dict()
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path}")
            
    print(f"\n[DONE] Training complete. Best Validation SI-SNR: {best_val_sisnr:.2f} dB")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinySpeechEnhancer")
    parser.add_argument("--manifest", type=str, default="../dataset/data_generated/manifest.json", 
                        help="Path to dataset manifest.json")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()
    
    train(manifest_path=args.manifest, epochs=args.epochs, batch_size=args.batch_size, 
          lr=args.lr, output_dir=args.output_dir)
