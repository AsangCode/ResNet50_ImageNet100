# ResNet50 Training on ImageNet100

This project implements training of ResNet50 from scratch on the ImageNet100 dataset using AWS EC2 for GPU acceleration.

## Dataset
ImageNet100 is a curated subset of ImageNet with:
- 100 classes
- Training and validation images for each class
- Full ImageNet resolution images

## Project Structure
- `train.py`: Training loop and utilities
- `model.py`: ResNet50 model implementation
- `dataset.py`: Dataset loading and preprocessing
- `lr_finder.py`: Learning rate finder utility
- `config.sh`: Configuration variables
- `config.sh.template`: Template for configuration
- `train_on_ec2.sh`: Main training pipeline script
- `requirements.txt`: Project dependencies

## Prerequisites
1. AWS CLI installed and configured locally
2. EC2 instance with GPU support
3. S3 bucket for dataset and results
4. Dataset in zip format (archive.zip)

## Setup and Configuration

1. Copy and configure `config.sh`:
```bash
cp config.sh.template config.sh
# Edit config.sh with your settings:
# - EC2_INSTANCE: Your EC2 instance DNS
# - EC2_KEY: Path to your EC2 key file
# - S3_BUCKET: Your S3 bucket name
# - Other configurations as needed
```

2. Set up AWS credentials:
```bash
# Create and edit set_aws_credentials.sh:
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="your_region"
```

3. Make scripts executable:
```bash
chmod +x train_on_ec2.sh
chmod +x set_aws_credentials.sh
```

## Running the Training Pipeline

1. Source AWS credentials:
```bash
source set_aws_credentials.sh
```

2. Run the training pipeline:
```bash
./train_on_ec2.sh
```

The pipeline will automatically:
1. Verify AWS connections and permissions
2. Check/upload dataset to S3 if needed
3. Find and use the largest available volume on EC2
4. Set up Python environment and dependencies
5. Download and prepare the dataset
6. Run training with proper configurations
7. Download results locally

## Training Configuration
- Batch size: 128 (configurable in config.sh)
- Number of workers: 2 (configurable in config.sh)
- Number of epochs: 100 (configurable in config.sh)
- Learning rate: Determined by lr_finder
- Optimizer: SGD with momentum
- Weight decay: 1e-4

## Results and Outputs
Results are automatically downloaded to `./results/` including:
- Checkpoints (.pth) in `results/checkpoints/`
- Model files (.pt) in `results/models/`
- Training logs in `results/logs/`
- TensorBoard logs in `results/runs/`

## Monitoring
Training progress can be monitored:
1. Through script output showing real-time progress
2. Using TensorBoard after downloading results:
```bash
tensorboard --logdir results/runs
```

## Storage Management
The script automatically:
- Uses the largest available volume on EC2
- Manages cache directories for pip, apt, and Python libraries
- Cleans up temporary files
- Verifies space requirements

## Error Handling
The script includes robust error handling for:
- AWS connectivity issues
- Space constraints
- Dataset integrity
- Training interruptions
