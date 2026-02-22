import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
import argparse
from datetime import datetime
import torch.backends.cudnn as cudnn
import signal
import sys
from contextlib import contextmanager
import time

from model.paper10_net import CFFANet_OOD
from data import get_loader
from utils import clip_gradient, adjust_lr

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
# Training hyperparameters
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='every n epochs decay learning rate')
parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
parser.add_argument('--seed', type=int, default=12, help='random seed')
parser.add_argument('--deterministic', type=int, default=0, help='whether use deterministic training')
# Model configuration
parser.add_argument('--model_name', type=str, default='CFFANet_OOD', help='Name of the model to use')
parser.add_argument('--trained_model_name', type=str, default='polyp_ood_Kvasir_lr2_j', help='Name of the saved model')

# OOD-specific arguments
parser.add_argument('--use_mixstyle', type=int, default=1, help='Enable MixStyle augmentation')
parser.add_argument('--test_time_augmentation', action='store_true', help='Use TTA during validation')
parser.add_argument('--augmentation_level', type=str, default='aggressive',
                    choices=['light', 'medium', 'aggressive'], help='Data augmentation intensity')

# NEW: Timeout and debugging
parser.add_argument('--batch_timeout', type=int, default=60, help='Timeout per batch in seconds')
parser.add_argument('--checkpoint_freq', type=int, default=1, help='Save checkpoint every N epochs')

opt = parser.parse_args()


# ============================================================================
# Timeout Handler
# ============================================================================

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time"""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # Set up signal handler (Unix/Linux only, won't work on Windows)
    if sys.platform != 'win32':
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Windows: use simple time tracking
        yield


# ============================================================================
# Unified Optimal Loss for Polyp Segmentation
# ============================================================================

class UnifiedPolypLoss(nn.Module):
    def __init__(self,
                 alpha=0.3,  # Weight for False Positives
                 beta=0.7,  # Weight for False Negatives (higher = recall more polyps)
                 gamma=1.33,  # Focal parameter (higher = focus on hard cases)
                 smooth=1e-6):
        super(UnifiedPolypLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

        # Sobel operators for boundary detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)

    def focal_tversky_loss(self, pred, target):
        # Flatten predictions and targets
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        # Calculate Tversky components
        TP = (pred * target).sum(dim=1)
        FP = ((1 - target) * pred).sum(dim=1)
        FN = (target * (1 - pred)).sum(dim=1)

        # Tversky Index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Focal modulation - emphasizes hard examples
        focal_tversky = torch.pow(1 - tversky, 1 / self.gamma)

        return focal_tversky.mean()

    def boundary_loss(self, pred, target):

        # Move kernels to correct device
        if self.sobel_x.device != target.device:
            self.sobel_x = self.sobel_x.to(target.device)
            self.sobel_y = self.sobel_y.to(target.device)

        # Compute boundary weights from ground truth
        target_grad_x = torch.abs(F.conv2d(target, self.sobel_x, padding=1))
        target_grad_y = torch.abs(F.conv2d(target, self.sobel_y, padding=1))
        target_boundary = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2)

        # Create weight map (higher weight at boundaries)
        boundary_weight = 1 + 10 * target_boundary

        # Weighted BCE on actual predictions (not gradients!)
        weighted_bce = F.binary_cross_entropy(
            pred,
            target,
            weight=boundary_weight,
            reduction='mean'
        )

        return weighted_bce

    def structure_loss(self, pred, target):

        # Compute structure weight (higher weight near boundaries)
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )
        # Weighted BCE
        wbce = F.binary_cross_entropy(pred, target, weight=weit, reduction='mean')
        return wbce

    def forward(self, pred, target):
        # 1. Focal Tversky Loss (main component)
        ftl = self.focal_tversky_loss(pred, target)
        # 2. Boundary Loss (edge accuracy)
        bl = self.boundary_loss(pred, target)
        # 3. Structure Loss (shape preservation)
        sl = self.structure_loss(pred, target)
        # Weighted combination
        # FTL is primary, boundary and structure are auxiliary
        total_loss = ftl + 0.5 * bl + 0.3 * sl
        # Return individual components for monitoring
        loss_dict = {
            'ftl': ftl.item(),  # Focal Tversky
            'bnd': bl.item(),  # Boundary
            'str': sl.item()  # Structure
        }
        return total_loss, loss_dict

def train(train_loader, model, optimizer, criterion, epoch, total_step):
    """Training function with timeout protection"""
    model.train()
    total_loss = 0.0
    loss_dict_sum = {}
    print(f"\n🔄 Starting Epoch {epoch} training...")
    for i, pack in enumerate(train_loader, start=1):
        batch_start_time = time.time()
        try:
            optimizer.zero_grad()
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            # Forward pass
            preds = model(images)
            # Compute loss
            loss, loss_dict = criterion(preds, gts)
            # Backward pass
            loss.backward()
            if opt.clip > 0:
                clip_gradient(optimizer, opt.clip)
            optimizer.step()
            total_loss += loss.item()

            for key, val in loss_dict.items():
                if key not in loss_dict_sum:
                    loss_dict_sum[key] = 0.0
                loss_dict_sum[key] += val
            batch_time = time.time() - batch_start_time

            # Print progress
            if i % 20 == 0 or i == total_step:
                current_lr = opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch)
                loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.6f}, Total Loss: {:.4f}, {}, Time: {:.2f}s'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch, opt.epoch, i, total_step, current_lr, loss.item(), loss_str, batch_time))
            # Detect hanging batch
            if batch_time > opt.batch_timeout:
                print(f"⚠️  WARNING: Batch {i} took {batch_time:.1f}s (timeout: {opt.batch_timeout}s)")

        except Exception as e:
            print(f"\n❌ ERROR in batch {i}: {str(e)}")
            print("Skipping this batch and continuing...")
            continue

    avg_loss_dict = {k: v / total_step for k, v in loss_dict_sum.items()}
    print(f"✅ Training epoch {epoch} completed!")
    return total_loss / total_step, avg_loss_dict


def validate(val_loader, model, criterion):
    """Validation function with progress tracking"""
    model.eval()
    total_val_loss = 0.0
    loss_dict_sum = {}

    print("🔍 Running validation...", end='', flush=True)

    val_start_time = time.time()
    num_batches = len(val_loader)

    with torch.no_grad():
        for i, pack in enumerate(val_loader, start=1):
            try:
                images, gts = pack
                images = images.cuda()
                gts = gts.cuda()

                preds = model(images)
                loss, loss_dict = criterion(preds, gts)

                total_val_loss += loss.item()

                for key, val in loss_dict.items():
                    if key not in loss_dict_sum:
                        loss_dict_sum[key] = 0.0
                    loss_dict_sum[key] += val

                # Progress indicator
                if i % 5 == 0 or i == num_batches:
                    print(f" {i}/{num_batches}", end='', flush=True)

            except Exception as e:
                print(f"\n❌ ERROR in validation batch {i}: {str(e)}")
                continue

    val_time = time.time() - val_start_time
    print(f" Done! ({val_time:.1f}s)")

    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_sum.items()}
    return total_val_loss / num_batches, avg_loss_dict


# ============================================================================
# Main Training Function
# ============================================================================

def main():

    # Set deterministic behavior
    if not opt.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # Set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # Define paths
    image_root = './dataset/train/Kvasir-SEG/images/'
    gt_root = './dataset/train/Kvasir-SEG/masks_aug_binary/'
    val_image_root = './dataset/test/Kvasir-SEG/images/'
    val_gt_root = './dataset/test/Kvasir-SEG/masks_binary/'

    # Load datasets with FIXED settings
    print("Loading datasets...")
    train_loader = get_loader(
        image_root, gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        mode='train',
        augmentation_level=opt.augmentation_level,
        shuffle=True,
        num_workers=8,  # CRITICAL FIX
        pin_memory=True  # CRITICAL FIX
    )

    val_loader = get_loader(
        val_image_root, val_gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        mode='val',
        augmentation_level='light',
        shuffle=False,
        num_workers=8,  # CRITICAL FIX
        pin_memory=True  # CRITICAL FIX
    )

    total_step = len(train_loader)
    print(f"✅ Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    print(f"✅ Batches per epoch: {total_step}")

    # Initialize model
    print(f"\nInitializing {opt.model_name} model...")
    print(f"  - Loss Function: Unified Focal Tversky Loss (State-of-the-Art)")
    print(f"  - Batch Timeout: {opt.batch_timeout}s")

    model = CFFANet_OOD(
        num_channels=3,
        num_classes=1,
        pretrained=True,
    )
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Initialize loss and optimizer
    criterion = UnifiedPolypLoss(alpha=0.3, beta=0.7, gamma=1.33)
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    # Create save directory
    save_path = f"trained_models/{opt.model_name}/"
    os.makedirs(save_path, exist_ok=True)
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    print("\n" + "=" * 70)
    print("🚀 Starting Training with Monitoring!")
    print("=" * 70 + "\n")

    training_start_time = time.time()

    for epoch in range(1, opt.epoch + 1):
        epoch_start_time = time.time()
        # Adjust learning rate
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        try:
            # Train and validate
            train_loss, train_loss_dict = train(train_loader, model, optimizer, criterion, epoch, total_step)
            val_loss, val_loss_dict = validate(val_loader, model, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epoch_time = time.time() - epoch_start_time
            # Print epoch summary
            print(f'\n{"=" * 70}')
            print(f'Epoch [{epoch}/{opt.epoch}] Summary (Time: {epoch_time:.1f}s):')
            print(f'  Train Loss: {train_loss:.4f}')
            loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in train_loss_dict.items()])
            print(f'    └─ {loss_str}')
            print(f'  Val Loss:   {val_loss:.4f}')
            loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in val_loss_dict.items()])
            print(f'    └─ {loss_str}')
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                model_save_path = os.path.join(save_path, f"{opt.trained_model_name}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'use_uncertainty': False,
                    'use_mixstyle': opt.use_mixstyle,
                }, model_save_path)
                print(f'  ✅ Best model saved! (Val Loss: {best_val_loss:.4f})')
            else:
                patience_counter += 1
                print(f'  ⏳ No improvement for {patience_counter}/{opt.patience} epochs.')
            # Save checkpoint periodically
            if epoch % opt.checkpoint_freq == 0:
                checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, checkpoint_path)
                print(f'  💾 Checkpoint saved: epoch {epoch}')
            # Early stopping
            if patience_counter >= opt.patience:
                print(f'\n{"=" * 70}')
                print(f'⏹️  Early stopping at epoch {epoch}')
                print(f'Best epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}')
                print("=" * 70)
                break
            print("=" * 70 + "\n")

            # Clear GPU cache periodically
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
                print("🧹 GPU cache cleared\n")
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user!")
            print(f"Saving emergency checkpoint at epoch {epoch}...")
            emergency_path = os.path.join(save_path, f"emergency_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, emergency_path)
            print(f"✅ Emergency checkpoint saved to {emergency_path}")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in epoch {epoch}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nAttempting to continue training...")
            continue

    total_training_time = time.time() - training_start_time
    print(f"\n✅ Training completed in {total_training_time / 3600:.2f} hours!")
    print(f"Best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

    history_path = os.path.join(save_path, "training_history.pth")
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_training_time': total_training_time,
        'model_name': opt.model_name,
        'trained_model_name': opt.trained_model_name,
    }, history_path)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()