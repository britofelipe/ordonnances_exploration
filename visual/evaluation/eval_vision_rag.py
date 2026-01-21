import os
import json
import argparse
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- METRIC HELPER: IOU ---

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two boxes [x1, y1, x2, y2].
    """
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    # Determine intersection rectangle
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    iou = intersection_area / float(area_a + area_b - intersection_area)
    return iou

# --- PARSING HELPERS ---

def parse_qwen_output(text: str, width: int, height: int) -> Dict[str, List[List[int]]]:
    """
    Parses Qwen2-VL output format: <box>(y1,x1),(y2,x2)</box>class_name
    Returns dict: {'class_name': [[x1, y1, x2, y2], ...]}
    Note: Qwen outputs coords normalize to 1000.
    """
    # Regex for standard detection output
    # Matches: <box>(200,100),(500,400)</box>header
    pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>(\w+)"
    
    matches = re.findall(pattern, text)
    results = {}
    
    for ny1, nx1, ny2, nx2, label in matches:
        # Convert normalized (0-1000) back to pixel coords
        y1 = int(int(ny1) / 1000 * height)
        x1 = int(int(nx1) / 1000 * width)
        y2 = int(int(ny2) / 1000 * height)
        x2 = int(int(nx2) / 1000 * width)
        
        if label not in results:
            results[label] = []
        results[label].append([x1, y1, x2, y2])
        
    return results

# --- MAIN EVALUATION ---

def evaluate_model(args):
    logger.info(f"Loading Base Model: {args.base_model}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    processor = Qwen2VLProcessor.from_pretrained(args.base_model)
    
    if args.adapter_path:
        logger.info(f"Loading Adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    
    model.eval()
    
    # Data Listing
    data_dir = Path(args.data_dir)
    image_dir = data_dir / "images" if (data_dir / "images").exists() else data_dir
    images = sorted(list(image_dir.glob("*.png")))
    
    # We evaluate on the last N images (simulated test set)
    test_images = images[-args.test_count:] if args.test_count > 0 else images
    logger.info(f"Evaluating on {len(test_images)} images")
    
    metrics = {"iou_header": [], "iou_med_block": [], "iou_footer": []}
    
    for img_path in tqdm(test_images):
        file_id = img_path.stem
        gt_path = data_dir / f"{file_id}.json"
        
        if not gt_path.exists(): 
            continue
            
        # 1. Inference
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Detect the layout regions (header, medication_block, footer)."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
            
        # 2. Parse Predictions
        preds = parse_qwen_output(output_text, W, H)
        
        # 3. Load Ground Truth
        with open(gt_path, "r") as f:
            gt = json.load(f) # format: {'header': [[x1, y1, x2, y2], ...]}

        # DEBUG: Print first few examples
        if len(metrics["iou_med_block"]) < 3:
            print(f"\n--- DEBUG SAMPLE {file_id} ---", flush=True)
            print(f"RAW OUTPUT: {repr(output_text[:300])}...", flush=True) # Print repr to see special tokens
            print(f"PARSED PREDS: {preds}", flush=True)
            print(f"GROUND TRUTH: {gt}", flush=True)
            print("------------------------------", flush=True)
            
        # 4. Compare (Simplify: Compare Box with Highest IOU for each class)
        for cls in ["header", "medication_block", "footer"]:
            gt_boxes = gt.get(cls, [])
            pred_boxes = preds.get(cls, [])
            
            if not gt_boxes: continue # Skip if no GT
            
            # Simple metric: Mean Best IoU
            # For each GT box, find best matching Pred box
            class_ious = []
            for gt_box in gt_boxes:
                # FIX: Ground Truth is in [x, y, w, h] format, need to convert to [x1, y1, x2, y2]
                gt_x1, gt_y1, gt_w, gt_h = gt_box
                gt_xyxy = [gt_x1, gt_y1, gt_x1 + gt_w, gt_y1 + gt_h]
                
                best_iou_for_this_gt = 0.0
                if pred_boxes:
                    # pred_boxes are already [x1, y1, x2, y2] from parse_qwen_output
                    ious = [calculate_iou(gt_xyxy, pb) for pb in pred_boxes]
                    best_iou_for_this_gt = max(ious)
                class_ious.append(best_iou_for_this_gt)
            
            avg_cls_iou = sum(class_ious) / len(class_ious)
            if cls == "header": metrics["iou_header"].append(avg_cls_iou)
            if cls == "medication_block": metrics["iou_med_block"].append(avg_cls_iou)
            if cls == "footer": metrics["iou_footer"].append(avg_cls_iou)

    # Report
    print("\n" + "="*40)
    print(" EVALUATION RESULTS ")
    print("="*40)
    for k, v in metrics.items():
        if v:
            print(f"Mean {k}: {np.mean(v):.4f} (N={len(v)})")
        else:
            print(f"Mean {k}: N/A")
            
    # Save to file
    out_file = Path(args.output_dir) / "metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        final_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--data_dir", default="../dataset_v1")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--test_count", type=int, default=50, help="Number of images to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args)
