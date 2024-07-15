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

Then you can download all the required models (Dino and SAM)
```bash
cd Grounded-Sam-Labelling

mkdir dino_models
cd dino_models

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd ../sam_models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
