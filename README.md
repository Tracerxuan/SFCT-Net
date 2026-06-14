# SFCT-Net

A Spatial–Frequency Collaborative Transformer for Robust Breast Ultrasound Tumor Segmentation


![Structure_of_model](figures/Structure_of_model.tif)

![Visuralization_of_segmentation](figures/Visuralization_of_segmentation.tif)

## Structure of project
```text
SFCT-Net/
├── datasets/            
│   └── breast.py        
├── models/              
│   ├── sfct_net.py      
│   └── wtconv.py        
├── utils/               
│   ├── dataloader_alone.py 
│   ├── evaluation.py    
│   ├── loss.py          
│   ├── metrics.py       
│   └── predict.py      
└── train.py       
```

## Environment

- Python 3.8+
- PyTorch >= 1.12
- torchvision
- opencv-python
- albumentations
- medpy
- scikit-learn
- tqdm
- GeodisTK
- mindspore


### Train model

  ```
  python train.py
  ```
