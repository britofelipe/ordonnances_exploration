import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2VLProcessor, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2-VL-2B-Instruct")
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)

@dataclass
class DataArguments:
    data_dir: str = field(default="../dataset_v1", metadata={"help": "Path to dataset directory"})
    image_folder: str = field(default=".") # Relative to data_dir
    labels_folder: str = field(default=".") # Relative to data_dir

@dataclass
class TrainingArgs(TrainingArguments):
    output_dir: str = field(default="./checkpoints")
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-4)
    logging_steps: int = field(default=10)
    num_train_epochs: float = field(default=3.0)
    save_steps: int = field(default=100)
    bf16: bool = field(default=True)
    report_to: str = field(default="none")

class VisualPrescriptionDataset(Dataset):
    def __init__(self, data_dir: str, processor: Qwen2VLProcessor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Check if images are in a subdirectory
        if (self.data_dir / "images").exists():
            self.image_files = sorted(list((self.data_dir / "images").glob("*.png")))
        else:
            self.image_files = sorted(list(self.data_dir.glob("*.png")))

        if not self.image_files:
            raise ValueError(f"No images found in {data_dir} (checked subfolder 'images' too)")
        logger.info(f"Loaded {len(self.image_files)} images from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        file_id = img_path.stem # e.g. ordo_vis_000000
        json_path = self.data_dir / f"{file_id}.json"

        # Load Image
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # Load Labels
        if not json_path.exists():
            logger.warning(f"Label missing for {img_path}")
            return None # Handle in collate
        
        with open(json_path, "r") as f:
            boxes_dict = json.load(f)

        # Format Target Text
        # Qwen2-VL format: <box>(y1,x1),(y2,x2)</box>class_name
        # Note: Qwen expects coords normalized to 1000
        
        target_text = ""
        for cls_name, boxes in boxes_dict.items():
            for box in boxes:
                # Box is [x1, y1, x2, y2] unnormalized
                x1, y1, x2, y2 = box
                # Normalize to 0-1000
                nx1, ny1 = int((x1 / W) * 1000), int((y1 / H) * 1000)
                nx2, ny2 = int((x2 / W) * 1000), int((y2 / H) * 1000)
                
                # Qwen Format: (y1, x1), (y2, x2)
                # target_text += f"<box>({ny1},{nx1}),({ny2},{nx2})</box>{cls_name} "
                # Simplification: Let's train to output JSON-like structure or just the detection string
                # Qwen2-VL Prompt: "Detect the bounding box of text layout regions."
                
                # Using the standard Qwen detection format:
                target_text += f"<box>({ny1},{nx1}),({ny2},{nx2})</box>{cls_name}"

        # Prepare Conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Detect the layout regions (header, medication_block, footer)."}
                ]
            },
            {
                "role": "assistant",
                "content": target_text
            }
        ]

        # Process Inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Create inputs including image features
        image_inputs, video_inputs = transformers.models.qwen2_vl.image_processing_qwen2_vl.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Squeeze batch dim
        return {k: v.squeeze(0) for k, v in inputs.items()}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return {}
    
    # We need to pad input_ids and attention_mask
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['input_ids'] for item in batch] # Auto-regressive loss
    pixel_values = [item['pixel_values'] for item in batch]
    image_grid_thw = [item['image_grid_thw'] for item in batch]

    # Use torch.nn.utils.rnn.pad_sequence
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=151643) # <|endoftext|> ??? Check pad token
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Image tensors are concatenated
    pixel_values_cat = torch.cat(pixel_values, dim=0)
    image_grid_thw_cat = torch.cat(image_grid_thw, dim=0)

    # Attention mask
    attention_mask = (input_ids_padded != 151643).long()

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values_cat,
        "image_grid_thw": image_grid_thw_cat
    }

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Loading processor: {model_args.model_name_or_path}")
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path)

    logger.info("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # LoRA Configuration
    peft_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    train_dataset = VisualPrescriptionDataset(data_args.data_dir, processor)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
