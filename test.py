import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
import imageio
import json
import cv2

from model.paper10_net import CFFANet_OOD
from data import test_dataset


def postprocess_prediction(pred_prob, threshold=0.5, min_size=200):

    # Binarize
    pred_binary = (pred_prob > threshold).astype(np.uint8)

    # Morphological operations to improve mask quality
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel)  # Remove noise
    pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)  # Fill holes

    # Connected component analysis to remove small polyps
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pred_binary, connectivity=8
    )

    cleaned_binary = np.zeros_like(pred_binary)
    removed_count = 0
    kept_count = 0

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned_binary[labels == i] = 1
            kept_count += 1
        else:
            removed_count += 1

    return cleaned_binary, removed_count, kept_count


def compute_dice(gt, pred):
    """Dice coefficient"""
    gt = (gt > 0.5).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)
    intersection = np.sum(gt * pred)
    return (2 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-5)


def compute_iou(gt, pred):
    """Intersection over Union"""
    gt = (gt > 0.5).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-5)


def compute_metrics(gt, pred):
    """Compute all metrics"""
    gt = (gt > 0.5).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)

    tp = np.sum(gt * pred)
    fp = np.sum((1 - gt) * pred)
    fn = np.sum(gt * (1 - pred))

    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)

    # F2 Score (emphasizes recall)
    beta = 2.0
    f2 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-5)

    return precision, recall, f2


def test_model(model, test_loader, save_path, opt, device):
    """Test model and save results"""

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f2_scores = []

    total_removed = 0
    total_kept = 0

    with torch.no_grad():
        for i in tqdm(range(test_loader.size), desc="Testing"):
            # Load data
            image, gt, name = test_loader.load_data()

            gt_np = np.asarray(gt, np.float32)
            gt_np = gt_np / (gt_np.max() + 1e-8)

            # Forward pass
            image_tensor = image.to(device)
            pred_tensor = model(image_tensor)
            pred_tensor = F.interpolate(pred_tensor, size=gt_np.shape,
                                        mode='bilinear', align_corners=False)

            # Get prediction probability
            pred_prob = pred_tensor.cpu().numpy().squeeze()

            # Post-process: remove small polyps and improve mask quality
            pred_binary, removed_count, kept_count = postprocess_prediction(
                pred_prob,
                threshold=opt.threshold,
                min_size=opt.min_polyp_size
            )

            total_removed += removed_count
            total_kept += kept_count

            # Compute metrics on cleaned prediction
            dice = compute_dice(gt_np, pred_binary)
            iou = compute_iou(gt_np, pred_binary)
            precision, recall, f2 = compute_metrics(gt_np, pred_binary)

            dice_scores.append(dice)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f2_scores.append(f2)

            # Save cleaned prediction
            output_name = os.path.splitext(name)[0] + '.png'
            imageio.imwrite(os.path.join(save_path, 'salmap', output_name),
                            (pred_binary * 255).astype(np.uint8))

    # Calculate average metrics
    metrics = {
        'miou': float(np.mean(iou_scores)),
        'dice': float(np.mean(dice_scores)),
        'recall': float(np.mean(recall_scores)),
        'precision': float(np.mean(precision_scores)),
        'f2_score': float(np.mean(f2_scores)),
        'total_detections_kept': int(total_kept),
        'total_detections_removed': int(total_removed)
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./trained_models/CFFANet_OOD/trained_model.pth')
    parser.add_argument('--testsize', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Segmentation threshold (default: 0.5)')
    parser.add_argument('--min_polyp_size', type=int, default=300,
                        help='Minimum polyp size in pixels (default: 200)')
    parser.add_argument('--output_dir', type=str, default='./1_test_results/')
    parser.add_argument('--dataset_path', type=str, default='./dataset/test/')
    parser.add_argument('--test_datasets', nargs='+', default=['Kvasir-SEG' , 'Kvasir-SEG2', 'CVC-ClinicDB', 'BKAI-IGH'])

    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 50)
    print("Testing Polyp Segmentation Model")
    print("With Post-processing")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Threshold: {opt.threshold}")
    print(f"  Min polyp size: {opt.min_polyp_size} pixels")
    print(f"  Small detections will be REMOVED")

    # Load model
    print(f"\nLoading model: {opt.model_path}")
    checkpoint = torch.load(opt.model_path, map_location=device)

    state_dict = checkpoint.get('model_state_dict', checkpoint)

    model = CFFANet_OOD(
        num_channels=3,
        num_classes=1,
        pretrained=False,
        use_uncertainty=False,
        use_mixstyle=True
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    print("✓ Model loaded successfully")

    # Test on datasets
    all_metrics = {}

    for dataset in opt.test_datasets:
        print(f"\n{'-' * 50}")
        print(f"Testing on: {dataset}")
        print('-' * 50)

        img_root = os.path.join(opt.dataset_path, dataset, 'images/')
        gt_root = os.path.join(opt.dataset_path, dataset, 'masks_binary/')
        save_path = os.path.join(opt.output_dir, dataset)

        # Create output directory
        os.makedirs(os.path.join(save_path, 'salmap'), exist_ok=True)

        if not os.path.exists(img_root):
            print(f"✗ Dataset not found: {img_root}")
            continue

        # Load test data
        test_loader = test_dataset(img_root, gt_root, opt.testsize)
        print(f"Total images: {test_loader.size}")

        # Test
        metrics = test_model(model, test_loader, save_path, opt, device)
        all_metrics[dataset] = metrics

        # Print metrics
        print(f"\nResults:")
        print(f"  mIoU:      {metrics['miou']:.4f}")
        print(f"  Dice:      {metrics['dice']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  F2-Score:  {metrics['f2_score']:.4f}")
        print(f"\nPost-processing:")
        print(f"  Detections kept:    {metrics['total_detections_kept']}")
        print(f"  Detections removed: {metrics['total_detections_removed']} (< {opt.min_polyp_size}px)")

    # Save all metrics
    metrics_file = os.path.join(opt.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print("\n" + "=" * 50)
    print("✓ Testing completed")
    print("=" * 50)
    print(f"\nResults saved to: {opt.output_dir}")
    print(f"  ├─ salmap/      - Cleaned binary predictions")
    print(f"  └─ metrics.json - Accuracy metrics")
    print(f"\nPost-processing applied:")
    print(f"  • Morphological opening (noise removal)")
    print(f"  • Morphological closing (hole filling)")
    print(f"  • Small polyps removed (< {opt.min_polyp_size}px)")
    print()


if __name__ == '__main__':
    main()