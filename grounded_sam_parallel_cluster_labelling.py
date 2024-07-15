import argparse
import os
import sys
import numpy as np
import json
sys.path.append("..")
import torch
from PIL import Image
import time
import cv2
import matplotlib.pyplot as plt
from glob import glob
import yaml
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor
from utils.DatasetILSVRC2017 import ImageDetDataset, SAMImageBatchDataset, SAMBatchDataLoader, SAMImageDetDataset, SAMDataLoader, collate_fn
import re
from utils.generic import load_config, print_config
from tqdm import tqdm


from segment_anything.utils.transforms import ResizeLongestSide

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_image(image_path, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(device, checkpoint):
    model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_phrases_from_posmap_batch(logits_filt, tokenized, tokenizer, text_threshold):
    """
    Optimized version to get phrases from posmap in batch.
    """
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase)
    return pred_phrases

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    # Prepare the caption
    caption = caption.lower().strip() + "."

    # Move model and image to the specified device
    model = model.to(device)
    image = image.to(device)

    # Get model outputs without calculating gradients
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    # Extract logits and boxes
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    # Filter the outputs based on the threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # Tokenize the caption
    tokenized = model.tokenizer(caption)

    # Get phrases from the logits and tokenized caption
    pred_phrases = get_phrases_from_posmap_batch(logits_filt, tokenized, model.tokenizer, text_threshold)

    return boxes_filt, pred_phrases

def save_mask_mapping(mask_list, output_dir, base_name, mask_value):
    # Determine the shape of the mask image and initialize it
    mask_shape = mask_list.shape[-2:]
    mask_img = np.zeros(mask_shape)
    
    # Bring mask_list to CPU only once and convert to NumPy
    mask_np = mask_list.cpu().numpy()
    # Vectorized assignment using broadcasting
    for idx, mask in enumerate(mask_np):
        mask_img[mask[0] > 0] = int(mask_value)+1 # Add 1 to avoid 0 values (which are the background)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save the mask using plt.imsave with a colormap that can handle large integers
    #plt.imsave(os.path.join(output_dir, f'{base_name}_mask.png'), mask_img, cmap='gray')
    # Save the mask using imageio.imwrite
    output_path = os.path.join(output_dir, f'{base_name}_mask.png')
    #mask_img = mask_img.astype(np.uint16)
    cv2.imwrite(output_path, mask_img)


    
def get_object_identifier(filepath):
    # Split by '/' to isolate the filename
    filename = filepath.split('/')[-1]
    # Split by '_' to get the last part of the filename
    object_identifier = filename.split('_')[-1]
    # Remove the file extension
    object_identifier = object_identifier.split('.')[0]
    return object_identifier

def extract_image_number(filename):
  match = re.search(r'original_frame_(\d+)\.jpg', filename)
  if match:
      return int(match.group(1))
  return None  # In case there's a filename that doesn't match

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()


# Function to read a specific row from a text file
def read_specific_row(file_path, row_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if row_number < 1 or row_number > len(lines):
            raise IndexError("Row number out of range")
        return lines[row_number - 1].strip()  # Adjust for 1-based index and remove newline character

# Ensure that all other tensors used in the forward pass are also on the same device
def ensure_same_device(tensor, device):
    return tensor.to(device) if tensor.device != device else tensor

def check_tensors_on_gpu(batch_input):
    all_on_gpu = True
    for item in batch_input:
        print("Image: ", item['image'].is_cuda)
        print("Box: ", item['boxes'].is_cuda)
        print("Original boxes: ", item['original_boxes'].is_cuda)
        if not (item['image'].is_cuda and item['boxes'].is_cuda and item['original_boxes'].is_cuda):
            all_on_gpu = False
            break
    return all_on_gpu

def move_tensors_to_gpu(batch_input):
    for item in batch_input:
        if isinstance(item['image'], torch.Tensor):
            item['image'] = item['image'].to('cuda')
        if isinstance(item['boxes'], torch.Tensor):
            item['boxes'] = item['boxes'].to('cuda')
        if isinstance(item['original_boxes'], torch.Tensor):
            item['original_boxes'] = item['original_boxes'].to('cuda')
    return batch_input

def get_label_prefix(label):
    return label.split('_')[0]


def save_progress(progress_file, idx):
    with open(progress_file, 'w') as f:
        json.dump({"last_processed_idx": idx}, f)

def save_boxes_to_file(file_name_without_ext, boxes, file_path='boxes.txt'):
    """
    Save the boxes into a txt file in append mode with the specified format.

    Args:
    - file_name_without_ext (str): The name of the file without extension.
    - boxes (list): List of box coordinates.
    - file_path (str): The path of the file where boxes will be saved. Default is 'boxes.txt'.
    """
    with open(file_path, 'a') as file:
        for box in boxes:
            box_str = ' '.join([f'{coord:.6f}' for coord in box])
            file.write(f'{file_name_without_ext} {box_str}\n')


def run_pipeline(config_path, start_from=None):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the models 
    dino_model = load_model(device=device, checkpoint=config["dino_checkpoint"])
    sam = sam_model_registry[config["sam_model_type"]](checkpoint=config["sam_checkpoint"]).to(device)
    
    # Set the directories
    base_path = config["input_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataloader
    batch_size = config['batch_size']               # Batch size
    total_parts = config['total_parts']             # Total number of parts to split the data
    part = config['part']                           # The part to process          
    progress_file = f"progress_file{part}.json"     # File to store the progress
    list_file_path_new = f"file_list{part}.txt"    # Here we will store the new list file
    boxes_file_path = f"boxes{part}.txt"            # File to store the boxes

    print("Loading images and create label mapping: ")
    image_paths = []
    labels = []
    prompts = []
    list_file_path = config["list_file"]
    # Open and read the file
    with open(list_file_path, 'r') as file:
        for line in file:
            # Split the line by spaces
            parts = line.strip().split()
            
            # Extract the required elements
            image_path = parts[0]
            label = parts[1]
            prompt = parts[2]
            
            # Append the elements to the lists
            image_paths.append(image_path)          # Image path list
            labels.append(label)                    # Label list
            prompts.append(prompt)                  # Prompt list


    # Calculate split point
    split_point = len(image_paths) // total_parts
    
    part_start_idx = (part - 1) * split_point
    if part < total_parts:
        part_end_idx = part * split_point
    else:
        part_end_idx = len(image_paths)

    # Determine the actual starting index within the selected segment
    start_idx = start_from if start_from is not None else 0

    # Select the segment of the list to process and slice it starting from the given index
    image_paths = image_paths[part_start_idx:part_end_idx][start_idx*batch_size:]
    labels = labels[part_start_idx:part_end_idx][start_idx*batch_size:]
    prompts = prompts[part_start_idx:part_end_idx][start_idx*batch_size:]
    
    # Overwrite the file initially to clear its contents
    open(list_file_path_new, 'w').close()
    open(os.path.join(output_dir, boxes_file_path), 'w').close()
    # Create a new list file with the selected images
    with open(list_file_path_new, 'w') as new_file:
        for image_path, label, prompt in zip(image_paths, labels, prompts):
            new_file.write(f"{image_path} {label} {prompt}\n")
            
    print(f"Processing from index {part_start_idx + start_idx} to {part_end_idx} (local start index {start_idx})")
    print("Total images to process:", len(image_paths))
                
    box_thresh = config['box_threshold']
    text_thresh = config['text_threshold']

    # Set the dataloader
    sam_image_reader = SAMImageBatchDataset(base_path, list_file_path_new, image_paths, labels, prompts, sam, box_thresh, text_thresh, dino_model, device)
    image_loader = SAMBatchDataLoader(sam_image_reader, batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    print("Total batches: ", len(image_loader))
    total_batch = len(sam_image_reader) // batch_size
    torch.set_grad_enabled(False)

    # Buffer to collect data until the batch is full
    buffer = []

    for idx, batched_input in enumerate(tqdm(image_loader)):
        for item in batched_input:
            if item is not None:
                buffer.append(item)
            if len(buffer) == batch_size:
                # Process full batch
                batched_output = sam(buffer, multimask_output=False)
                for i, (output, input_data) in enumerate(zip(batched_output, buffer)):
                    mask_tensor = output['masks']
                    relative_image_path = input_data['relative_image_path']
                    file_name = os.path.basename(relative_image_path)  # Get the file name only
                    file_name_without_ext = os.path.splitext(file_name)[0]  # Remove the extension
                    label_prefix = get_label_prefix(file_name_without_ext)
                    save_dir = os.path.join(output_dir, label_prefix)
                    mask_value = input_data['mask_value']
                    box = input_data['original_boxes']
                    save_mask_mapping(mask_tensor, save_dir, file_name_without_ext, mask_value)  # Use label_prefix to match the category
                    save_boxes_to_file(file_name_without_ext, box, os.path.join(output_dir, boxes_file_path))
                buffer = []
                save_progress(progress_file, start_idx + idx)

    # Process any remaining items in the buffer
    if len(buffer) > 0:
        batched_output = sam(buffer, multimask_output=False)
        for i, (output, input_data) in enumerate(zip(batched_output, buffer)):
            mask_tensor = output['masks']
            relative_image_path = input_data['relative_image_path']
            file_name = os.path.basename(relative_image_path)  # Get the file name only
            file_name_without_ext = os.path.splitext(file_name)[0]  # Remove the extension
            label_prefix = get_label_prefix(file_name_without_ext)
            save_dir = os.path.join(output_dir, label_prefix)
            mask_value = input_data['mask_value']
            save_mask_mapping(mask_tensor, save_dir, file_name_without_ext, mask_value)  # Use label_prefix to match the category
            save_boxes_to_file(file_name_without_ext, box, os.path.join(output_dir, boxes_file_path))
        save_progress(progress_file, part_end_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument('--config', type=str, default='config/grounded_sam_cluster_labelling_config.yaml', help='Path to configuration file')
    parser.add_argument('--start_from', type=int, default=0, help='Starting batch index within the specified part')

    args = parser.parse_args()
    config = load_config(args.config)
    print_config(config)

    run_pipeline(args.config, args.start_from)