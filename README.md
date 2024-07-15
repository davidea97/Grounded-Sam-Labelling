# Grounded-Sam-Labelling
Annotation tool for labelling large scale dataset with text prompt based on Grounded Sam

## Set the environment
Clone the repository:
```bash
git clone https://github.com/davidea97/Grounded-Sam-Labelling.git
```

Before running this labelling tool it is recommended a python virtualenvironment with Python3.8:
```bash
virtualenv venv --python="/usr/bin/python3.8"
```

and install the required packages:
```bash
pip install -r requirements.txt
```

Then you can download all the DINO models
```bash
cd Grounded-Sam-Labelling

mkdir dino_models
cd dino_models

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
and the SAM models
```bash
cd ..
mkdir sam_models
cd sam_models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Run the labelling tool
To run the labelling tool you need to create or update the config/grounded_sam_cluster_labelling_config.yaml with the correct information

```bash
sam_checkpoint: sam_models/sam_vit_h_4b8939.pth # SAM models
dino_checkpoint: dino_models/groundingdino_swint_ogc.pth # DINO models

sam_model_type: vit_h
input_dir: <input_image_folder> # Folder with images to be processed

output_dir: <output_mask_folder> # Output folder where to save the masks
batch_size: 5
image_format: .JPEG
list_file: file_list.txt    # Txt file with the list of all images to be processed (required by the dataloader)

total_parts: 2 # Number of parts to split the dataset into
part: 2 # Part number to process

box_threshold: 0.5
text_threshold: 0.25
```

The txt file has to be created. It is required by the dataloader and it consists of N rows (one for each image to be processed) with this format:

```bash
<path/to/image> <mask_value> <text_prompt>
```

where the mask value is the number to assign to the mask.

Then you can run the labelling tool:
```bash
source venv/bin/activate
python grounded_sam_parallel_cluster_labelling.py --config <config_file>
```