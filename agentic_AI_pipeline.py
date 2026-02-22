"""
Agentic AI Framework for Polyp Segmentation and Interpretation
Using LangGraph with 5 Specialized Agents - OpenAI Version
1. Segmentation Agent
2. Confidence Analyzer Agent
3. Quality Assurance Agent
4. Triage Agent
5. Description Agent
"""

import os

# Fix OpenMP error - must be set before importing numpy/torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Suppress warnings
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
import cv2
import imageio
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI  # Changed from langchain_anthropic

from model.paper10_net import CFFANet_OOD
from data import test_dataset

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class PolypAnalysisState(TypedDict):
    """Global state shared across all agents"""
    # Input data
    image_name: str
    image_tensor: torch.Tensor
    ground_truth: np.ndarray
    original_image: np.ndarray

    # Segmentation outputs
    pred_prob: np.ndarray
    pred_binary: np.ndarray

    # Confidence analysis
    confidence_stats: dict

    # Quality assurance
    quality_metrics: dict
    post_processed: bool

    # Triage decision
    triage_decision: str  # "auto_approve", "flag_for_review", "reject"
    triage_reasoning: str
    flag_reasons: dict

    # Clinical description
    clinical_description: str
    interpretation: str

    # Agent messages/logs
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Workflow control
    current_agent: str
    requires_human_review: bool

    # Save paths
    save_path: str


@dataclass
class SegmentationResult:
    """Results from segmentation agent"""
    pred_prob: np.ndarray
    success: bool
    message: str


@dataclass
class ConfidenceAnalysis:
    """Results from confidence analyzer"""
    total_polyp_pixels: int
    mean_confidence: float
    std_confidence: float
    very_high_conf_pct: float
    high_conf_pct: float
    uncertain_pct: float
    low_conf_pct: float
    very_low_conf_pct: float
    confidence_distribution: dict


@dataclass
class QualityAssessment:
    """Results from quality assurance agent"""
    pred_binary: np.ndarray
    dice_score: float
    iou_score: float
    precision: float
    recall: float
    f2_score: float
    removed_detections: int
    kept_detections: int
    quality_passed: bool
    quality_issues: list


@dataclass
class TriageDecision:
    """Results from triage agent"""
    decision: str  # "auto_approve", "flag_for_review", "reject"
    confidence_level: str  # "high", "medium", "low"
    reasoning: str
    requires_human_review: bool
    flag_criteria_met: dict


@dataclass
class ClinicalDescription:
    """Results from description agent"""
    description: str
    interpretation: str
    clinical_recommendations: list
    uncertainty_notes: str


# ============================================================================
# AGENT 1: SEGMENTATION AGENT
# ============================================================================

class SegmentationAgent:
    """Agent responsible for initial polyp detection and segmentation"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.name = "SegmentationAgent"

    def __call__(self, state: PolypAnalysisState) -> PolypAnalysisState:
        """Perform polyp segmentation"""
        print(f"\n{'=' * 80}")
        print(f"🔬 {self.name}: Processing {state['image_name']}")
        print(f"{'=' * 80}")

        try:
            with torch.no_grad():
                # Forward pass
                image_tensor = state['image_tensor'].to(self.device)
                pred_tensor = self.model(image_tensor)

                # Resize to match ground truth
                gt_shape = state['ground_truth'].shape
                pred_tensor = F.interpolate(
                    pred_tensor,
                    size=gt_shape,
                    mode='bilinear',
                    align_corners=False
                )

                # Get probability map
                pred_prob = pred_tensor.cpu().numpy().squeeze()

                result = SegmentationResult(
                    pred_prob=pred_prob,
                    success=True,
                    message=f"Successfully segmented polyp regions. Probability map shape: {pred_prob.shape}"
                )

                print(f"✓ Segmentation complete")
                print(f"  - Output shape: {pred_prob.shape}")
                print(f"  - Value range: [{pred_prob.min():.3f}, {pred_prob.max():.3f}]")
                print(f"  - Mean probability: {pred_prob.mean():.3f}")

        except Exception as e:
            result = SegmentationResult(
                pred_prob=None,
                success=False,
                message=f"Segmentation failed: {str(e)}"
            )
            print(f"✗ Segmentation failed: {str(e)}")

        # Update state
        state['pred_prob'] = result.pred_prob
        state['current_agent'] = self.name
        state['messages'] = [AIMessage(content=result.message, name=self.name)]

        return state


# ============================================================================
# AGENT 2: CONFIDENCE ANALYZER AGENT
# ============================================================================

class ConfidenceAnalyzerAgent:
    """Agent responsible for analyzing prediction confidence and uncertainty"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "ConfidenceAnalyzerAgent"

    def analyze_confidence(self, pred_prob: np.ndarray) -> dict:
        """Analyze confidence distribution of predicted polyp pixels"""
        # Get predicted polyp pixels
        polyp_mask = pred_prob > self.threshold
        total_polyp_pixels = np.sum(polyp_mask)

        if total_polyp_pixels == 0:
            return {
                'total_polyp_pixels': 0,
                'very_low_conf_pixels': 0,
                'low_conf_pixels': 0,
                'uncertain_pixels': 0,
                'high_conf_pixels': 0,
                'very_high_conf_pixels': 0,
                'very_low_conf_pct': 0.0,
                'low_conf_pct': 0.0,
                'uncertain_pct': 0.0,
                'high_conf_pct': 0.0,
                'very_high_conf_pct': 0.0,
                'mean_confidence': 0.0,
                'std_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }

        # Extract confidence values for polyp pixels
        polyp_confidences = pred_prob[polyp_mask]

        # Count pixels in different confidence ranges
        very_low_conf = np.sum((polyp_confidences >= self.threshold) & (polyp_confidences < 0.3))
        low_conf = np.sum((polyp_confidences >= 0.3) & (polyp_confidences < 0.5))
        uncertain = np.sum((polyp_confidences >= 0.5) & (polyp_confidences < 0.7))
        high_conf = np.sum((polyp_confidences >= 0.7) & (polyp_confidences < 0.9))
        very_high_conf = np.sum(polyp_confidences >= 0.9)

        # Calculate statistics
        return {
            'total_polyp_pixels': int(total_polyp_pixels),
            'very_low_conf_pixels': int(very_low_conf),
            'low_conf_pixels': int(low_conf),
            'uncertain_pixels': int(uncertain),
            'high_conf_pixels': int(high_conf),
            'very_high_conf_pixels': int(very_high_conf),
            'very_low_conf_pct': float(very_low_conf / total_polyp_pixels * 100),
            'low_conf_pct': float(low_conf / total_polyp_pixels * 100),
            'uncertain_pct': float(uncertain / total_polyp_pixels * 100),
            'high_conf_pct': float(high_conf / total_polyp_pixels * 100),
            'very_high_conf_pct': float(very_high_conf / total_polyp_pixels * 100),
            'mean_confidence': float(np.mean(polyp_confidences)),
            'std_confidence': float(np.std(polyp_confidences)),
            'min_confidence': float(np.min(polyp_confidences)),
            'max_confidence': float(np.max(polyp_confidences))
        }

    def __call__(self, state: PolypAnalysisState) -> PolypAnalysisState:
        """Analyze confidence of segmentation predictions"""
        print(f"\n{'=' * 80}")
        print(f"📊 {self.name}: Analyzing prediction confidence")
        print(f"{'=' * 80}")

        pred_prob = state['pred_prob']
        conf_stats = self.analyze_confidence(pred_prob)

        # Generate analysis message
        if conf_stats['total_polyp_pixels'] == 0:
            message = "No polyp pixels detected (all predictions below threshold)"
        else:
            message = f"""Confidence Analysis Complete:
  - Total polyp pixels: {conf_stats['total_polyp_pixels']:,}
  - Mean confidence: {conf_stats['mean_confidence']:.3f} ± {conf_stats['std_confidence']:.3f}
  - Very High (0.9-1.0): {conf_stats['very_high_conf_pct']:.1f}%
  - High (0.7-0.9): {conf_stats['high_conf_pct']:.1f}%
  - Uncertain (0.5-0.7): {conf_stats['uncertain_pct']:.1f}%
  - Low (0.3-0.5): {conf_stats['low_conf_pct']:.1f}%
  - Very Low (<0.3): {conf_stats['very_low_conf_pct']:.1f}%"""

        print(message)

        # Update state
        state['confidence_stats'] = conf_stats
        state['current_agent'] = self.name
        state['messages'] = state.get('messages', []) + [
            AIMessage(content=message, name=self.name)
        ]

        return state


# ============================================================================
# AGENT 3: QUALITY ASSURANCE AGENT
# ============================================================================

class QualityAssuranceAgent:
    """Agent responsible for post-processing and quality validation"""

    def __init__(self, threshold: float = 0.5, min_polyp_size: int = 500):
        self.threshold = threshold
        self.min_polyp_size = min_polyp_size
        self.name = "QualityAssuranceAgent"

    def postprocess_prediction(self, pred_prob: np.ndarray) -> tuple:
        """Post-process prediction to remove small false positives"""
        pred_binary = (pred_prob > self.threshold).astype(np.uint8)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel)
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)

        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pred_binary, connectivity=8
        )

        cleaned_binary = np.zeros_like(pred_binary)
        removed_count = 0
        kept_count = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_polyp_size:
                cleaned_binary[labels == i] = 1
                kept_count += 1
            else:
                removed_count += 1

        return cleaned_binary, removed_count, kept_count

    def compute_metrics(self, gt: np.ndarray, pred: np.ndarray) -> dict:
        """Compute segmentation quality metrics"""
        gt = (gt > 0.5).astype(np.float32)
        pred = (pred > 0.5).astype(np.float32)

        # Basic metrics
        tp = np.sum(gt * pred)
        fp = np.sum((1 - gt) * pred)
        fn = np.sum(gt * (1 - pred))

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)

        # Dice coefficient
        intersection = tp
        dice = (2 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-5)

        # IoU
        union = np.sum(gt) + np.sum(pred) - intersection
        iou = intersection / (union + 1e-5)

        # F2 score
        beta = 2.0
        f2 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-5)

        return {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f2_score': float(f2)
        }

    def __call__(self, state: PolypAnalysisState) -> PolypAnalysisState:
        """Perform quality assurance and validation"""
        print(f"\n{'=' * 80}")
        print(f"🔍 {self.name}: Validating segmentation quality")
        print(f"{'=' * 80}")

        # Post-process predictions
        pred_prob = state['pred_prob']
        gt = state['ground_truth']

        pred_binary, removed, kept = self.postprocess_prediction(pred_prob)

        # Compute quality metrics
        metrics = self.compute_metrics(gt, pred_binary)
        metrics['removed_detections'] = removed
        metrics['kept_detections'] = kept

        # Quality assessment
        quality_issues = []
        if metrics['dice'] < 0.5:
            quality_issues.append("Low Dice score (< 0.5)")
        if metrics['recall'] < 0.6:
            quality_issues.append("Low recall (< 0.6)")
        if kept == 0:
            quality_issues.append("No valid detections after post-processing")

        quality_passed = len(quality_issues) == 0

        message = f"""Quality Assurance Complete:
  - Post-processing: Removed {removed} small detections, kept {kept}
  - Dice Score: {metrics['dice']:.4f}
  - IoU: {metrics['iou']:.4f}
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F2 Score: {metrics['f2_score']:.4f}
  - Quality Status: {'✓ PASSED' if quality_passed else '⚠ ISSUES DETECTED'}"""

        if quality_issues:
            message += f"\n  - Issues: {', '.join(quality_issues)}"

        print(message)

        # Update state
        state['pred_binary'] = pred_binary
        state['quality_metrics'] = metrics
        state['post_processed'] = True
        state['current_agent'] = self.name
        state['messages'] = state.get('messages', []) + [
            AIMessage(content=message, name=self.name)
        ]

        return state


# ============================================================================
# AGENT 4: TRIAGE AGENT
# ============================================================================

class TriageAgent:
    """Agent responsible for intelligent routing and human-in-the-loop integration"""

    def __init__(self,
                 very_high_threshold: float = 87.89,
                 high_threshold: float = 6.68,
                 uncertain_threshold: float = 4.96):
        self.very_high_threshold = very_high_threshold
        self.high_threshold = high_threshold
        self.uncertain_threshold = uncertain_threshold
        self.name = "TriageAgent"

    def check_flagging_criteria(self, conf_stats: dict) -> tuple:
        """Determine if case requires human review"""
        very_high_pct = conf_stats['very_high_conf_pct']
        high_pct = conf_stats['high_conf_pct']
        uncertain_pct = conf_stats['uncertain_pct']

        # Check each criterion
        criterion_1 = very_high_pct < self.very_high_threshold
        criterion_2 = high_pct > self.high_threshold
        criterion_3 = uncertain_pct > self.uncertain_threshold

        # Flag if ALL criteria are met
        should_flag = criterion_1 and criterion_2 and criterion_3

        return should_flag, {
            'very_high_below_threshold': criterion_1,
            'high_above_threshold': criterion_2,
            'uncertain_above_threshold': criterion_3,
            'values': {
                'very_high_pct': very_high_pct,
                'high_pct': high_pct,
                'uncertain_pct': uncertain_pct
            }
        }

    def make_triage_decision(self, state: PolypAnalysisState) -> TriageDecision:
        """Make intelligent triage decision"""
        conf_stats = state['confidence_stats']
        quality_metrics = state['quality_metrics']

        # Check flagging criteria
        should_flag, flag_reasons = self.check_flagging_criteria(conf_stats)

        # Determine confidence level
        if conf_stats['very_high_conf_pct'] > 90:
            confidence_level = "high"
        elif conf_stats['very_high_conf_pct'] > 70:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Make decision
        if should_flag:
            decision = "flag_for_review"
            reasoning = f"""Case flagged for expert review due to:
  - Very High Confidence: {conf_stats['very_high_conf_pct']:.1f}% (threshold: <{self.very_high_threshold}%)
  - High Confidence: {conf_stats['high_conf_pct']:.1f}% (threshold: >{self.high_threshold}%)
  - Uncertain Confidence: {conf_stats['uncertain_pct']:.1f}% (threshold: >{self.uncertain_threshold}%)

This indicates potential out-of-distribution or ambiguous presentation requiring human expertise."""
            requires_human_review = True

        elif quality_metrics['dice'] < 0.3:
            decision = "reject"
            reasoning = f"Poor segmentation quality (Dice: {quality_metrics['dice']:.3f}). Prediction unreliable."
            requires_human_review = True

        else:
            decision = "auto_approve"
            reasoning = f"""Case automatically approved:
  - High confidence: {conf_stats['very_high_conf_pct']:.1f}%
  - Good quality: Dice={quality_metrics['dice']:.3f}
  - No flags triggered"""
            requires_human_review = False

        return TriageDecision(
            decision=decision,
            confidence_level=confidence_level,
            reasoning=reasoning,
            requires_human_review=requires_human_review,
            flag_criteria_met=flag_reasons
        )

    def __call__(self, state: PolypAnalysisState) -> PolypAnalysisState:
        """Perform intelligent triage"""
        print(f"\n{'=' * 80}")
        print(f"🚦 {self.name}: Making triage decision")
        print(f"{'=' * 80}")

        triage = self.make_triage_decision(state)

        # Determine status symbol
        if triage.decision == "auto_approve":
            status = "✓ AUTO-APPROVED"
        elif triage.decision == "flag_for_review":
            status = "⚠️ FLAGGED FOR REVIEW"
        else:
            status = "✗ REJECTED"

        message = f"""{status}
  - Decision: {triage.decision.upper()}
  - Confidence Level: {triage.confidence_level.upper()}
  - Requires Human Review: {triage.requires_human_review}

Reasoning:
{triage.reasoning}"""

        print(message)

        # Update state
        state['triage_decision'] = triage.decision
        state['triage_reasoning'] = triage.reasoning
        state['requires_human_review'] = triage.requires_human_review
        state['flag_reasons'] = triage.flag_criteria_met
        state['current_agent'] = self.name
        state['messages'] = state.get('messages', []) + [
            AIMessage(content=message, name=self.name)
        ]

        return state


# ============================================================================
# AGENT 5: DESCRIPTION AGENT
# ============================================================================

class DescriptionAgent:
    """Agent responsible for generating clinically meaningful interpretations"""

    def __init__(self, llm=None):
        self.name = "DescriptionAgent"
        self.llm = llm

    def generate_description(self, state: PolypAnalysisState) -> ClinicalDescription:
        """Generate clinical description and interpretation"""
        conf_stats = state['confidence_stats']
        quality_metrics = state['quality_metrics']
        triage_decision = state['triage_decision']

        # Basic description
        if conf_stats['total_polyp_pixels'] == 0:
            description = "No polyp detected in this image."
            interpretation = "The model did not identify any regions with sufficient confidence to be classified as polyp tissue."
            recommendations = ["Consider manual inspection if polyp is clinically suspected"]
            uncertainty = "N/A - No detection"
        else:
            # Describe polyp characteristics
            polyp_area_mm2 = conf_stats['total_polyp_pixels'] * 0.01  # Approximate

            description = f"""Polyp Detection Summary:
  - Detected region area: ~{polyp_area_mm2:.0f} pixels ({polyp_area_mm2 * 0.01:.1f} mm²)
  - Average model confidence: {conf_stats['mean_confidence']:.1%}
  - Confidence distribution:
    * Very high (>90%): {conf_stats['very_high_conf_pct']:.1f}%
    * High (70-90%): {conf_stats['high_conf_pct']:.1f}%
    * Uncertain (50-70%): {conf_stats['uncertain_pct']:.1f}%"""

            # Interpretation based on confidence
            if conf_stats['very_high_conf_pct'] > 85:
                interpretation = "High-confidence polyp detection. The model is highly certain about the polyp boundaries and characteristics."
            elif conf_stats['uncertain_pct'] > 20:
                interpretation = "Moderate-confidence detection with significant uncertain regions. This may indicate challenging morphology, poor visibility, or atypical presentation."
            else:
                interpretation = "Variable confidence detection. Some regions are well-defined while others show uncertainty."

            # Recommendations
            recommendations = []
            if triage_decision == "flag_for_review":
                recommendations.append("Expert review recommended due to confidence distribution")
            if quality_metrics['dice'] < 0.7:
                recommendations.append("Consider additional imaging or biopsy confirmation")
            if conf_stats['uncertain_pct'] > 15:
                recommendations.append("Close monitoring recommended due to uncertain boundaries")
            if not recommendations:
                recommendations.append("Standard polyp resection protocol applicable")

            # Uncertainty notes
            if conf_stats['std_confidence'] > 0.25:
                uncertainty = f"High variability in confidence (σ={conf_stats['std_confidence']:.3f}) suggests heterogeneous polyp characteristics or imaging artifacts."
            else:
                uncertainty = "Confidence levels are relatively consistent across the detected region."

        return ClinicalDescription(
            description=description,
            interpretation=interpretation,
            clinical_recommendations=recommendations,
            uncertainty_notes=uncertainty
        )

    def __call__(self, state: PolypAnalysisState) -> PolypAnalysisState:
        """Generate clinical description"""
        print(f"\n{'=' * 80}")
        print(f"📝 {self.name}: Generating clinical interpretation")
        print(f"{'=' * 80}")

        clinical = self.generate_description(state)

        message = f"""Clinical Interpretation:

{clinical.description}

Interpretation:
{clinical.interpretation}

Uncertainty Assessment:
{clinical.uncertainty_notes}

Recommendations:
{chr(10).join(['  - ' + rec for rec in clinical.clinical_recommendations])}"""

        print(message)

        # Update state
        state['clinical_description'] = clinical.description
        state['interpretation'] = clinical.interpretation
        state['current_agent'] = self.name
        state['messages'] = state.get('messages', []) + [
            AIMessage(content=message, name=self.name)
        ]

        return state


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

def create_polyp_analysis_workflow(model, device, opt):
    """Create the LangGraph workflow with all agents"""

    # Initialize agents
    segmentation_agent = SegmentationAgent(model, device)
    confidence_agent = ConfidenceAnalyzerAgent(threshold=opt.threshold)
    quality_agent = QualityAssuranceAgent(
        threshold=opt.threshold,
        min_polyp_size=opt.min_polyp_size
    )
    triage_agent = TriageAgent(
        very_high_threshold=opt.very_high_threshold,
        high_threshold=opt.high_threshold,
        uncertain_threshold=opt.uncertain_threshold
    )

    # Initialize OpenAI LLM (optional - for enhanced description generation)
    llm = None
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if openai_api_key:
        try:
            llm = ChatOpenAI(
                model=opt.openai_model,  # Use model from arguments
                temperature=0.3,
                openai_api_key=openai_api_key
            )
            print(f"✓ OpenAI LLM initialized with model: {opt.openai_model}")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize OpenAI LLM: {str(e)}")
            print("  Falling back to rule-based descriptions")
    else:
        print("ℹ OPENAI_API_KEY not found in .env file")
        print("  Using rule-based clinical descriptions")

    description_agent = DescriptionAgent(llm=llm)

    # Create workflow graph
    workflow = StateGraph(PolypAnalysisState)

    # Add nodes (agents)
    workflow.add_node("segmentation", segmentation_agent)
    workflow.add_node("confidence_analysis", confidence_agent)
    workflow.add_node("quality_assurance", quality_agent)
    workflow.add_node("triage", triage_agent)
    workflow.add_node("description", description_agent)

    # Define workflow edges (sequential pipeline)
    workflow.set_entry_point("segmentation")
    workflow.add_edge("segmentation", "confidence_analysis")
    workflow.add_edge("confidence_analysis", "quality_assurance")
    workflow.add_edge("quality_assurance", "triage")
    workflow.add_edge("triage", "description")
    workflow.add_edge("description", END)

    return workflow.compile()


def create_visualization(state: PolypAnalysisState) -> np.ndarray:
    """Create comprehensive visualization of the analysis results"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Prepare data
    original = state['original_image']
    pred_prob = state['pred_prob']
    pred_binary = state['pred_binary']
    gt = state['ground_truth']
    conf_stats = state['confidence_stats']
    quality_metrics = state['quality_metrics']

    # Resize images to match if needed
    if original is not None:
        h, w = original.shape[:2]
        if pred_prob.shape != (h, w):
            pred_prob_resized = cv2.resize(pred_prob, (w, h))
            pred_binary_resized = cv2.resize(pred_binary.astype(np.float32), (w, h))
            gt_resized = cv2.resize(gt.astype(np.float32), (w, h))
        else:
            pred_prob_resized = pred_prob
            pred_binary_resized = pred_binary
            gt_resized = gt
    else:
        pred_prob_resized = pred_prob
        pred_binary_resized = pred_binary
        gt_resized = gt

    # Create figure with subplots
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.3)

    # ========== Row 1: Image Analysis ==========
    # Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    if original is not None:
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(np.zeros_like(gt_resized), cmap='gray')
    ax1.set_title('Original Image', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_resized, cmap='gray')
    ax2.set_title('Ground Truth', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # Prediction Binary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_binary_resized, cmap='gray')
    ax3.set_title(f'Prediction (Binary)\nDice: {quality_metrics["dice"]:.3f}',
                  fontsize=13, fontweight='bold')
    ax3.axis('off')

    # Overlay on Original
    ax4 = fig.add_subplot(gs[0, 3])
    if original is not None:
        overlay = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
        # Create colored mask overlay
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[pred_binary_resized > 0.5] = [255, 0, 0]  # Red for prediction
        overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        ax4.imshow(overlay)
    else:
        ax4.imshow(pred_binary_resized, cmap='Reds', alpha=0.5)
    ax4.set_title('Prediction Overlay', fontsize=13, fontweight='bold')
    ax4.axis('off')

    # GT vs Prediction Comparison
    ax5 = fig.add_subplot(gs[0, 4])
    comparison = np.zeros((*gt_resized.shape, 3), dtype=np.uint8)
    comparison[gt_resized > 0.5] = [0, 255, 0]  # Green for GT
    comparison[pred_binary_resized > 0.5] = [255, 0, 0]  # Red for prediction
    comparison[(gt_resized > 0.5) & (pred_binary_resized > 0.5)] = [255, 255, 0]  # Yellow for overlap
    ax5.imshow(comparison)
    ax5.set_title('GT vs Prediction\n🟢GT 🔴Pred 🟡Overlap',
                  fontsize=13, fontweight='bold')
    ax5.axis('off')

    # ========== Row 2: Confidence & Attention Analysis ==========
    # Confidence Heatmap
    ax6 = fig.add_subplot(gs[1, 0])
    im1 = ax6.imshow(pred_prob_resized, cmap='jet', vmin=0, vmax=1)
    ax6.set_title('Confidence Heatmap', fontsize=13, fontweight='bold')
    ax6.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax6, fraction=0.046, pad=0.04)
    cbar1.set_label('Confidence', rotation=270, labelpad=15)

    # GradCAM-style Attention Map (using confidence as pseudo-gradcam)
    ax7 = fig.add_subplot(gs[1, 1])
    if original is not None:
        # Create attention overlay
        attention_map = cv2.applyColorMap((pred_prob_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        attention_map = cv2.cvtColor(attention_map, cv2.COLOR_BGR2RGB)

        # Blend with original
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(original_rgb, 0.5, attention_map, 0.5, 0)
        ax7.imshow(blended)
    else:
        ax7.imshow(pred_prob_resized, cmap='jet')
    ax7.set_title('Attention Map (GradCAM-style)', fontsize=13, fontweight='bold')
    ax7.axis('off')

    # Uncertainty Map (Standard Deviation visualization)
    ax8 = fig.add_subplot(gs[1, 2])
    # Calculate uncertainty as distance from decision boundary (0.5)
    uncertainty = 0.5 - np.abs(pred_prob_resized - 0.5)
    im2 = ax8.imshow(uncertainty, cmap='RdYlGn_r', vmin=0, vmax=0.5)
    ax8.set_title('Uncertainty Map\n(Distance from threshold)', fontsize=13, fontweight='bold')
    ax8.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax8, fraction=0.046, pad=0.04)
    cbar2.set_label('Uncertainty', rotation=270, labelpad=15)

    # Edge Detection Overlay
    ax9 = fig.add_subplot(gs[1, 3])
    if original is not None:
        # Detect edges in prediction
        pred_uint8 = (pred_binary_resized * 255).astype(np.uint8)
        edges = cv2.Canny(pred_uint8, 50, 150)

        # Overlay edges on original
        edge_overlay = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
        edge_overlay[edges > 0] = [255, 255, 0]  # Yellow edges
        ax9.imshow(edge_overlay)
    else:
        edges = cv2.Canny((pred_binary_resized * 255).astype(np.uint8), 50, 150)
        ax9.imshow(edges, cmap='gray')
    ax9.set_title('Boundary Detection', fontsize=13, fontweight='bold')
    ax9.axis('off')

    # Confidence Histogram
    ax10 = fig.add_subplot(gs[1, 4])
    if conf_stats['total_polyp_pixels'] > 0:
        polyp_mask = pred_prob_resized > 0.5
        polyp_confidences = pred_prob_resized[polyp_mask]

        counts, bins, patches = ax10.hist(polyp_confidences, bins=50,
                                          color='steelblue', edgecolor='black', alpha=0.7)

        # Color bars by confidence level
        for i, patch in enumerate(patches):
            if bins[i] < 0.5:
                patch.set_facecolor('red')
            elif bins[i] < 0.7:
                patch.set_facecolor('yellow')
            elif bins[i] < 0.9:
                patch.set_facecolor('lightgreen')
            else:
                patch.set_facecolor('darkgreen')

        ax10.axvline(conf_stats['mean_confidence'], color='black', linestyle='--',
                     linewidth=2, label=f'Mean: {conf_stats["mean_confidence"]:.3f}')
        ax10.axvline(0.5, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Threshold')
        ax10.set_xlabel('Confidence', fontsize=11)
        ax10.set_ylabel('Pixel Count', fontsize=11)
        ax10.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
        ax10.legend(fontsize=9)
        ax10.grid(alpha=0.3)
    else:
        ax10.text(0.5, 0.5, 'No polyp detected', ha='center', va='center',
                  fontsize=14, transform=ax10.transAxes)
        ax10.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
        ax10.axis('off')

    # ========== Row 3: Metrics & Clinical Assessment ==========
    # Confidence Level Breakdown
    ax11 = fig.add_subplot(gs[2, 0:2])
    if conf_stats['total_polyp_pixels'] > 0:
        categories = ['Very High\n(≥90%)', 'High\n(70-90%)', 'Uncertain\n(50-70%)',
                      'Low\n(30-50%)', 'Very Low\n(<30%)']
        values = [
            conf_stats['very_high_conf_pct'],
            conf_stats['high_conf_pct'],
            conf_stats['uncertain_pct'],
            conf_stats['low_conf_pct'],
            conf_stats['very_low_conf_pct']
        ]
        colors = ['#2E7D32', '#66BB6A', '#FDD835', '#FB8C00', '#E53935']
        bars = ax11.bar(categories, values, color=colors, edgecolor='black',
                        linewidth=1.5, alpha=0.8)
        ax11.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax11.set_title('Confidence Level Distribution', fontsize=13, fontweight='bold')
        ax11.grid(axis='y', alpha=0.3, linestyle='--')
        ax11.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 100)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width() / 2., height,
                      f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax11.text(0.5, 0.5, 'No polyp detected', ha='center', va='center',
                  fontsize=16, transform=ax11.transAxes, fontweight='bold')
        ax11.set_title('Confidence Level Distribution', fontsize=13, fontweight='bold')
        ax11.axis('off')

    # Performance Metrics Table
    ax12 = fig.add_subplot(gs[2, 2])
    ax12.axis('off')
    metrics_text = f"""QUALITY METRICS
━━━━━━━━━━━━━━━━━━━
Dice Score:    {quality_metrics['dice']:.4f}
IoU:           {quality_metrics['iou']:.4f}
Precision:     {quality_metrics['precision']:.4f}
Recall:        {quality_metrics['recall']:.4f}
F2 Score:      {quality_metrics['f2_score']:.4f}

DETECTIONS
━━━━━━━━━━━━━━━━━━━
Kept:          {quality_metrics['kept_detections']}
Removed:       {quality_metrics['removed_detections']}

STATISTICS
━━━━━━━━━━━━━━━━━━━
Polyp Pixels:  {conf_stats['total_polyp_pixels']:,}
Mean Conf:     {conf_stats['mean_confidence']:.3f}
Std Dev:       {conf_stats['std_confidence']:.3f}
"""

    # Color code by quality
    if quality_metrics['dice'] >= 0.8:
        box_color = '#C8E6C9'  # Light green
    elif quality_metrics['dice'] >= 0.5:
        box_color = '#FFF9C4'  # Light yellow
    else:
        box_color = '#FFCDD2'  # Light red

    ax12.text(0.05, 0.95, metrics_text, transform=ax12.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8,
                        edgecolor='black', linewidth=2))
    ax12.set_title('Performance Metrics', fontsize=13, fontweight='bold')

    # Triage Decision & Clinical Summary
    ax13 = fig.add_subplot(gs[2, 3:5])
    ax13.axis('off')

    # Determine triage status color
    if state['triage_decision'] == 'auto_approve':
        status_color = '#C8E6C9'  # Light green
        status_symbol = '✓'
        status_emoji = '✅'
    elif state['triage_decision'] == 'flag_for_review':
        status_color = '#FFE082'  # Light orange
        status_symbol = '⚠'
        status_emoji = '⚠️'
    else:
        status_color = '#FFCDD2'  # Light red
        status_symbol = '✗'
        status_emoji = '❌'

    decision_text = f"""{status_emoji} TRIAGE DECISION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: {state['triage_decision'].upper().replace('_', ' ')}

REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{state['triage_reasoning'][:350]}

CLINICAL INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{state['interpretation'][:280]}{'...' if len(state['interpretation']) > 280 else ''}
"""

    ax13.text(0.02, 0.98, decision_text, transform=ax13.transAxes,
              fontsize=9.5, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.9,
                        edgecolor='black', linewidth=2))
    ax13.set_title('Clinical Assessment', fontsize=13, fontweight='bold')

    # Add main title with status indicator
    title_color = 'green' if state['triage_decision'] == 'auto_approve' else 'orange' if state[
                                                                                             'triage_decision'] == 'flag_for_review' else 'red'
    fig.suptitle(f'{status_symbol} Polyp Analysis Report: {state["image_name"]}',
                 fontsize=18, fontweight='bold', y=0.98, color=title_color)

    # Add footer with timestamp
    import datetime
    footer_text = f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Framework: Agentic AI with OpenAI"
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=9, style='italic', alpha=0.7)

    # Convert plot to image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # Convert RGBA to RGB

    plt.close(fig)

    return img_array


def save_case_outputs(state: PolypAnalysisState, opt):
    """Save all outputs for a case"""
    image_name = state['image_name']
    output_name = os.path.splitext(image_name)[0] + '.png'

    # Determine save directory based on triage decision
    if state['triage_decision'] == "flag_for_review":
        base_dir = os.path.join(state['save_path'], 'flagged_cases')
    else:
        base_dir = os.path.join(state['save_path'], 'okay_cases')

    # Save binary prediction
    binary_path = os.path.join(base_dir, 'binary', output_name)
    imageio.imwrite(binary_path, (state['pred_binary'] * 255).astype(np.uint8))

    # Save probability map
    prob_path = os.path.join(base_dir, 'probability', output_name)
    imageio.imwrite(prob_path, (state['pred_prob'] * 255).astype(np.uint8))

    # Save ground truth
    gt_path = os.path.join(base_dir, 'ground_truth', output_name)
    imageio.imwrite(gt_path, (state['ground_truth'] * 255).astype(np.uint8))

    # Save original image if available
    if state['original_image'] is not None:
        orig_path = os.path.join(base_dir, 'original_images', output_name)
        cv2.imwrite(orig_path, state['original_image'])

    # Create and save visualization
    try:
        print(f"  → Creating visualization...")
        viz_image = create_visualization(state)
        viz_path = os.path.join(base_dir, 'visualization', output_name)

        # Ensure directory exists
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)

        # Save visualization
        success = cv2.imwrite(viz_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))

        if success:
            print(f"  ✓ Visualization saved: {viz_path}")
        else:
            print(f"  ✗ Failed to save visualization to: {viz_path}")

    except Exception as e:
        print(f"  ✗ ERROR creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

    # Save metadata
    metadata = {
        'image_name': image_name,
        'triage_decision': state['triage_decision'],
        'triage_reasoning': state['triage_reasoning'],
        'requires_human_review': state['requires_human_review'],
        'confidence_stats': state['confidence_stats'],
        'quality_metrics': state['quality_metrics'],
        'clinical_description': state['clinical_description'],
        'interpretation': state['interpretation']
    }

    metadata_path = os.path.join(base_dir, 'metadata',
                                 os.path.splitext(image_name)[0] + '.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./trained_models/CFFANet_OOD/polyp_ood_Kvasir_lr3_j.pth')
    parser.add_argument('--testsize', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min_polyp_size', type=int, default=500)

    # Flagging criteria
    parser.add_argument('--very_high_threshold', type=float, default=87.89)
    parser.add_argument('--high_threshold', type=float, default=6.68)
    parser.add_argument('--uncertain_threshold', type=float, default=4.96)

    parser.add_argument('--output_dir', type=str, default='./agentic_results/')
    parser.add_argument('--dataset_path', type=str, default='./dataset/test/')
    parser.add_argument('--test_datasets', nargs='+',
                        default=['Kvasir-SEG'])

    # OpenAI model selection (API key loaded from .env file)
    parser.add_argument('--openai_model', type=str, default='gpt-4',
                        choices=['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                        help='OpenAI model to use (API key from .env file)')

    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("🤖 AGENTIC AI FRAMEWORK FOR POLYP SEGMENTATION - OpenAI Version")
    print("=" * 80)
    print(f"\nArchitecture: 5 Specialized Autonomous Agents")
    print(f"  1. 🔬 Segmentation Agent      - Initial polyp detection")
    print(f"  2. 📊 Confidence Analyzer     - Prediction reliability assessment")
    print(f"  3. 🔍 Quality Assurance       - Output validation")
    print(f"  4. 🚦 Triage Agent           - Intelligent routing & human-in-the-loop")
    print(f"  5. 📝 Description Agent       - Clinical interpretation generation (OpenAI)")

    # Check for .env file
    if os.path.exists('.env'):
        print(f"\n✓ Found .env file")
    else:
        print(f"\n⚠ Warning: .env file not found")
        print(f"  Create a .env file with: OPENAI_API_KEY=your-api-key-here")

    # Load model
    print(f"\n{'=' * 80}")
    print(f"Loading model: {opt.model_path}")
    print(f"{'=' * 80}")
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

    # Create workflow
    workflow = create_polyp_analysis_workflow(model, device, opt)
    print("\n✓ Agentic workflow compiled")

    # Process datasets
    all_results = {}

    for dataset in opt.test_datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing Dataset: {dataset}")
        print(f"{'=' * 80}")

        img_root = os.path.join(opt.dataset_path, dataset, 'images/')
        gt_root = os.path.join(opt.dataset_path, dataset, 'masks_binary/')
        save_path = os.path.join(opt.output_dir, dataset)

        # Create output directories
        for case_type in ['flagged_cases', 'okay_cases']:
            for subdir in ['binary', 'probability', 'ground_truth',
                           'original_images', 'metadata', 'visualization']:
                os.makedirs(os.path.join(save_path, case_type, subdir), exist_ok=True)

        if not os.path.exists(img_root):
            print(f"✗ Dataset not found: {img_root}")
            continue

        # Load test data
        test_loader = test_dataset(img_root, gt_root, opt.testsize)
        print(f"Total images: {test_loader.size}")

        dataset_results = {
            'flagged_cases': [],
            'okay_cases': [],
            'metrics': {
                'dice_scores': [],
                'iou_scores': [],
                'precision_scores': [],
                'recall_scores': [],
                'f2_scores': []
            }
        }

        # Process each image through the agentic workflow
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()

            # Prepare ground truth
            gt_np = np.asarray(gt, np.float32)
            gt_np = gt_np / (gt_np.max() + 1e-8)

            # Load original image
            img_path = None
            base_name = os.path.splitext(name)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_path = os.path.join(img_root, base_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break

            original_img = None
            if img_path and os.path.exists(img_path):
                original_img = cv2.imread(img_path)

            # Initialize state
            initial_state = {
                'image_name': name,
                'image_tensor': image,
                'ground_truth': gt_np,
                'original_image': original_img,
                'pred_prob': None,
                'pred_binary': None,
                'confidence_stats': {},
                'quality_metrics': {},
                'post_processed': False,
                'triage_decision': '',
                'triage_reasoning': '',
                'flag_reasons': {},
                'clinical_description': '',
                'interpretation': '',
                'messages': [],
                'current_agent': '',
                'requires_human_review': False,
                'save_path': save_path
            }

            # Execute workflow
            print(f"\n{'#' * 80}")
            print(f"Processing Image {i + 1}/{test_loader.size}: {name}")
            print(f"{'#' * 80}")

            final_state = workflow.invoke(initial_state)

            # Save outputs
            print(f"\n{'→' * 40}")
            print(f"Saving outputs for {name}...")
            print(f"{'→' * 40}")
            save_case_outputs(final_state, opt)

            # Collect results
            case_result = {
                'image_name': name,
                'triage_decision': final_state['triage_decision'],
                'requires_human_review': final_state['requires_human_review'],
                'confidence_stats': final_state['confidence_stats'],
                'quality_metrics': final_state['quality_metrics'],
                'clinical_description': final_state['clinical_description'],
                'interpretation': final_state['interpretation']
            }

            if final_state['triage_decision'] == 'flag_for_review':
                dataset_results['flagged_cases'].append(case_result)
            else:
                dataset_results['okay_cases'].append(case_result)

            # Collect metrics
            metrics = final_state['quality_metrics']
            dataset_results['metrics']['dice_scores'].append(metrics['dice'])
            dataset_results['metrics']['iou_scores'].append(metrics['iou'])
            dataset_results['metrics']['precision_scores'].append(metrics['precision'])
            dataset_results['metrics']['recall_scores'].append(metrics['recall'])
            dataset_results['metrics']['f2_scores'].append(metrics['f2_score'])

        # Calculate aggregate metrics
        dataset_results['aggregate_metrics'] = {
            'mean_dice': float(np.mean(dataset_results['metrics']['dice_scores'])),
            'mean_iou': float(np.mean(dataset_results['metrics']['iou_scores'])),
            'mean_precision': float(np.mean(dataset_results['metrics']['precision_scores'])),
            'mean_recall': float(np.mean(dataset_results['metrics']['recall_scores'])),
            'mean_f2': float(np.mean(dataset_results['metrics']['f2_scores'])),
            'total_cases': test_loader.size,
            'flagged_cases': len(dataset_results['flagged_cases']),
            'okay_cases': len(dataset_results['okay_cases']),
            'flagging_rate': len(dataset_results['flagged_cases']) / test_loader.size * 100
        }

        all_results[dataset] = dataset_results

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"Dataset Summary: {dataset}")
        print(f"{'=' * 80}")
        agg = dataset_results['aggregate_metrics']
        print(f"  Mean Dice:      {agg['mean_dice']:.4f}")
        print(f"  Mean IoU:       {agg['mean_iou']:.4f}")
        print(f"  Mean Precision: {agg['mean_precision']:.4f}")
        print(f"  Mean Recall:    {agg['mean_recall']:.4f}")
        print(f"  Mean F2:        {agg['mean_f2']:.4f}")
        print(f"\n  Total Cases:    {agg['total_cases']}")
        print(f"  Flagged Cases:  {agg['flagged_cases']} ({agg['flagging_rate']:.1f}%)")
        print(f"  Okay Cases:     {agg['okay_cases']}")

        # Save dataset results
        results_file = os.path.join(save_path, 'agentic_results.json')
        with open(results_file, 'w') as f:
            json.dump(dataset_results, f, indent=4)
        print(f"\n✓ Results saved to: {results_file}")

    # Save overall results
    overall_file = os.path.join(opt.output_dir, 'overall_agentic_results.json')
    with open(overall_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n{'=' * 80}")
    print("✓ AGENTIC AI FRAMEWORK EXECUTION COMPLETED")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {opt.output_dir}")
    print(f"\nDirectory Structure:")
    print(f"  ├─ [dataset]/")
    print(f"  │   ├─ flagged_cases/")
    print(f"  │   │   ├─ binary/              - Segmentation masks (flagged)")
    print(f"  │   │   ├─ probability/         - Confidence maps (flagged)")
    print(f"  │   │   ├─ original_images/     - Source images (flagged)")
    print(f"  │   │   ├─ ground_truth/        - GT masks (flagged)")
    print(f"  │   │   └─ metadata/            - Agent outputs (flagged)")
    print(f"  │   ├─ okay_cases/")
    print(f"  │   │   ├─ binary/              - Segmentation masks (approved)")
    print(f"  │   │   ├─ probability/         - Confidence maps (approved)")
    print(f"  │   │   ├─ original_images/     - Source images (approved)")
    print(f"  │   │   ├─ ground_truth/        - GT masks (approved)")
    print(f"  │   │   └─ metadata/            - Agent outputs (approved)")
    print(f"  │   └─ agentic_results.json     - Complete agent outputs")
    print(f"  └─ overall_agentic_results.json - Cross-dataset summary")
    print()


if __name__ == '__main__':
    main()