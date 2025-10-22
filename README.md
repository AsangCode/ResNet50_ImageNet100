# ResNet50 Training on TinyImageNet

This project implements training of ResNet50 from scratch on the TinyImageNet dataset, aiming to achieve 75% top-1 accuracy.

## Dataset
TinyImageNet is a subset of ImageNet with:
- 200 classes
- 500 training images per class
- 50 validation images per class
- Images are 64x64 pixels

## Project Structure
- `dataset.py`: Dataset loading and preprocessing
- `model.py`: ResNet50 model implementation
- `train.py`: Training loop and utilities
- `requirements.txt`: Project dependencies

## Local Setup and Training

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training locally:
```bash
python train.py
```

## AWS EC2 Training Pipeline

### Prerequisites
1. AWS CLI installed
2. EC2 instance with GPU support
3. S3 bucket created
4. Dataset in zip format (archive.zip)

### Setup AWS and EC2

1. Configure AWS CLI:
```bash
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., ap-south-1)
# - Default output format (json)
```

2. Update configuration in `run_training_pipeline.sh`:
```bash
EC2_INSTANCE="your-ec2-instance-dns"  # EC2 instance DNS/IP
EC2_KEY="path/to/your/key.pem"        # Path to your EC2 key file
S3_BUCKET="your-s3-bucket"            # Your S3 bucket name
DATASET_ZIP="archive.zip"             # Your dataset zip name
```

3. Make script executable:
```bash
chmod +x run_training_pipeline.sh
```

### Running the Training Pipeline

1. For testing (1 epoch):
```bash
bash run_training_pipeline_test.sh
```

2. For full training (100 epochs):
```bash
bash run_training_pipeline.sh
```

The pipeline will:
- Upload dataset to S3
- Set up EC2 environment
- Download dataset on EC2
- Install dependencies
- Run training
- Save models (.pth and .pt)
- Upload results to S3
- Download results locally

### Results and Outputs
- Checkpoints (.pth) in `results/checkpoints/`
- Model files (.pt) in `results/models/`
- Training logs in `results/logs/`
- TensorBoard logs in `results/tensorboard/`


## Training Details
- Batch size: 128
- Optimizer: SGD with momentum (0.9)
- Initial learning rate: 0.1
- Weight decay: 1e-4
- Learning rate schedule: Cosine Annealing
- Number of epochs: 100

## Data Augmentation
- Random resized crop
- Random horizontal flip
- Color jitter
- Normalization using ImageNet statistics

## Monitoring
Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir runs
```

## Model Checkpoints
Best model checkpoints are saved in the `checkpoints` directory.
