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

# EC2 Logs
asang@Asang_Durga MINGW64 /e/GitHub_Repo/ERA_Assignment/Colab_ResNet50
$ bash run_training_pipeline.sh
Debug: Starting script
Debug: EC2_KEY = E:/GitHub_Repo/ERA_Assignment/Colab_ResNet50/ResNet50_ImageNet100.pem
Debug: EC2_INSTANCE = ec2-13-203-221-230.ap-south-1.compute.amazonaws.com
Debug: S3_BUCKET = my-resnet50-training-bucket
[STATUS] Checking AWS CLI configuration...
[STATUS] Checking if dataset exists in S3...
[STATUS] Dataset already exists in S3, skipping upload...
[STATUS] Creating directories on EC2...
[STATUS] Copying project files to EC2...
train.py                                                                                 100% 7927   265.2KB/s   00:00    
model.py                                                                                 100% 4353   133.7KB/s   00:00    
dataset.py                                                                               100% 3157   110.8KB/s   00:00    
lr_finder.py                                                                             100% 4158   141.2KB/s   00:00    
requirements.txt                                                                         100%  136     4.5KB/s   00:00    
setup_ec2.sh                                                                             100% 2153    76.1KB/s   00:00    
[STATUS] Starting setup and training on EC2...
Filesystem                      Size  Used Avail Use% Mounted on
/dev/root                        96G   39G   58G  40% /
tmpfs                            16G     0   16G   0% /dev/shm
tmpfs                           6.2G  964K  6.2G   1% /run
tmpfs                           5.0M     0  5.0M   0% /run/lock
efivarfs                        128K  4.1K  119K   4% /sys/firmware/efi/efivars
/dev/nvme0n1p16                 881M  107M  713M  13% /boot
/dev/nvme0n1p15                 105M  6.2M   99M   6% /boot/efi
/dev/mapper/vg.01-lv_ephemeral  206G   51G  145G  26% /opt/dlami/nvme
tmpfs                           3.1G   12K  3.1G   1% /run/user/1000
Dataset already exists, skipping download and extraction...
Hit:1 http://ap-south-1.ec2.archive.ubuntu.com/ubuntu noble InRelease
Hit:2 http://ap-south-1.ec2.archive.ubuntu.com/ubuntu noble-updates InRelease
Hit:3 http://ap-south-1.ec2.archive.ubuntu.com/ubuntu noble-backports InRelease
Hit:4 https://nvidia.github.io/libnvidia-container/stable/deb/amd64  InRelease
Hit:5 https://download.docker.com/linux/ubuntu noble InRelease
Hit:6 https://apt.corretto.aws stable InRelease
Hit:7 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64  InRelease
Hit:8 http://security.ubuntu.com/ubuntu noble-security InRelease
Hit:9 https://fsx-lustre-client-repo.s3.amazonaws.com/ubuntu noble InRelease
Reading package lists...
Reading package lists...
Building dependency tree...
Reading state information...
python3-venv is already the newest version (3.12.3-0ubuntu2).
python3-pip is already the newest version (24.0+dfsg-1ubuntu1.3).
0 upgraded, 0 newly installed, 0 to remove and 13 not upgraded.
Requirement already satisfied: pip in ./venv/lib/python3.12/site-packages (25.2)
Requirement already satisfied: torch>=2.0.0 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 1)) (2.9.0)
Requirement already satisfied: torchvision>=0.15.0 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 2)) (0.24.0)
Requirement already satisfied: numpy>=1.21.0 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 3)) (2.3.4)
Requirement already satisfied: matplotlib>=3.4.3 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 4)) (3.10.7)
Requirement already satisfied: tqdm>=4.62.3 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 5)) (4.67.1)
Requirement already satisfied: requests>=2.26.0 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 6)) (2.32.5)
Requirement already satisfied: pillow>=8.3.1 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 7)) (12.0.0)
Requirement already satisfied: tensorboard>=2.7.0 in ./venv/lib/python3.12/site-packages (from -r requirements.txt (line 8)) (2.20.0)
Requirement already satisfied: filelock in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.20.0)
Requirement already satisfied: typing-extensions>=4.10.0 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (4.15.0)
Requirement already satisfied: setuptools in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (80.9.0)
Requirement already satisfied: sympy>=1.13.3 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.5)
Requirement already satisfied: jinja2 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (2025.9.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.3.20)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (1.13.1.3)
Requirement already satisfied: triton==3.5.0 in ./venv/lib/python3.12/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.5.0)
Requirement already satisfied: contourpy>=1.0.1 in ./venv/lib/python3.12/site-packages (from matplotlib>=3.4.3->-r requirements.txt (line 4)) (1.3.3)
Requirement already satisfied: cycler>=0.10 in ./venv/lib/python3.12/site-packages (from matplotlib>=3.4.3->-r requirements.txt (line 4)) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in ./venv/lib/python3.12/site-packages (from matplotlib>=3.4.3->-r requirements.txt (line 4)) (4.60.1)
Requirement already satisfied: kiwisolver>=1.3.1 in ./venv/lib/python3.12/site-packages (from matplotlib>=3.4.3->-r requirements.txt (line 4)) (1.4.9)
Requirement already satisfied: packaging>=20.0 in ./venv/lib/python3.12/site-packages (from matplotlib>=3.4.3->-r requirements.txt (line 4)) (25.0)
Requirement already satisfied: pyparsing>=3 in ./venv/lib/python3.12/site-packages (from matplotlib>=3.4.3->-r requirements.txt (line 4)) (3.2.5)
Requirement already satisfied: python-dateutil>=2.7 in ./venv/lib/python3.12/site-packages (from matplotlib>=3.4.3->-r requirements.txt (line 4)) (2.9.0.post0)
Requirement already satisfied: charset_normalizer<4,>=2 in ./venv/lib/python3.12/site-packages (from requests>=2.26.0->-r requirements.txt (line 6)) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in ./venv/lib/python3.12/site-packages (from requests>=2.26.0->-r requirements.txt (line 6)) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./venv/lib/python3.12/site-packages (from requests>=2.26.0->-r requirements.txt (line 6)) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in ./venv/lib/python3.12/site-packages (from requests>=2.26.0->-r requirements.txt (line 6)) (2025.10.5)
Requirement already satisfied: absl-py>=0.4 in ./venv/lib/python3.12/site-packages (from tensorboard>=2.7.0->-r requirements.txt (line 8)) (2.3.1)
Requirement already satisfied: grpcio>=1.48.2 in ./venv/lib/python3.12/site-packages (from tensorboard>=2.7.0->-r requirements.txt (line 8)) (1.76.0)
Requirement already satisfied: markdown>=2.6.8 in ./venv/lib/python3.12/site-packages (from tensorboard>=2.7.0->-r requirements.txt (line 8)) (3.9)
Requirement already satisfied: protobuf!=4.24.0,>=3.19.6 in ./venv/lib/python3.12/site-packages (from tensorboard>=2.7.0->-r requirements.txt (line 8)) (6.33.0)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in ./venv/lib/python3.12/site-packages (from tensorboard>=2.7.0->-r requirements.txt (line 8)) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in ./venv/lib/python3.12/site-packages (from tensorboard>=2.7.0->-r requirements.txt (line 8)) (3.1.3)
Requirement already satisfied: six>=1.5 in ./venv/lib/python3.12/site-packages (from python-dateutil>=2.7->matplotlib>=3.4.3->-r requirements.txt (line 4)) (1.17.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in ./venv/lib/python3.12/site-packages (from sympy>=1.13.3->torch>=2.0.0->-r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.1.1 in ./venv/lib/python3.12/site-packages (from werkzeug>=1.0.1->tensorboard>=2.7.0->-r requirements.txt (line 8)) (3.0.3)
Using device: cuda
Running learning rate finder...
Finding optimal learning rate:   0%|          | 0/200 [00:00<?, ?it/s]/opt/dlami/nvme/training/resnet50/venv/lib/python3.12/site-packages/torch/backends/cuda/__init__.py:131: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  return torch._C._get_cublas_allow_tf32()
W1021 19:16:07.636000 15565 venv/lib/python3.12/site-packages/torch/_inductor/utils.py:1558] [0/0] Not enough SMs to use ma32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  return torch._C._get_cublas_allow_tf32()
W1021 19:16:07.636000 15565 venv/lib/python3.12/site-packages/torch/_inductor/utils.py:1558] [0/0] Not enough SMs to use max_autotune_gemm mode
Finding optimal learning rate: 100%|██████████| 200/200 [05:41<00:00,  1.71s/it, loss=4.604, lr=9.661e-01]

LR Finder Results:
Minimum loss achieved at learning rate: 0.966051
Suggested learning rate (min_loss_lr/10): 0.096605
Epoch 1/15
----------
Training Epoch 1:   0%|          | 0/990 [00:00<?, ?it/s]/opt/dlami/nvme/training/resnet50/venv/lib/python3.12/site-packages/torch/optim/lr_scheduler.py:192: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
Training Epoch 1: 100%|██████████| 990/990 [08:30<00:00,  1.94it/s, loss=4.047, acc=4.04%]
Train Loss: 4.4409 Acc: 4.04%
Validation: 100%|██████████| 40/40 [00:18<00:00,  2.16it/s]
Val Loss: 4.0663 Acc: 8.62%
Epoch 2/15
----------
Training Epoch 2: 100%|██████████| 990/990 [07:25<00:00,  2.22it/s, loss=3.622, acc=12.43%]
Train Loss: 3.7723 Acc: 12.43%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.57it/s]
Val Loss: 3.3902 Acc: 17.48%
Epoch 3/15
----------
Training Epoch 3: 100%|██████████| 990/990 [07:12<00:00,  2.29it/s, loss=3.224, acc=19.52%]
Train Loss: 3.3563 Acc: 19.52%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.55it/s]
Val Loss: 3.1204 Acc: 23.62%
Epoch 4/15
----------
Training Epoch 4: 100%|██████████| 990/990 [07:12<00:00,  2.56it/s, loss=3.141, acc=25.49%]Train Loss: 3.0547 Acc: 25.49%
Training Epoch 4: 100%|██████████| 990/990 [07:12<00:00,  2.29it/s, loss=3.141, acc=25.49%]
Validation: 100%|██████████| 40/40 [00:16<00:00,  2.41it/s]
Val Loss: 2.9967 Acc: 27.90%
Epoch 5/15
----------
Training Epoch 5: 100%|██████████| 990/990 [07:17<00:00,  2.26it/s, loss=2.361, acc=30.01%]
Train Loss: 2.8255 Acc: 30.01%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.53it/s]
Val Loss: 2.7736 Acc: 32.28%
Epoch 6/15
----------
Training Epoch 6: 100%|██████████| 990/990 [07:11<00:00,  2.30it/s, loss=2.355, acc=34.57%]
Train Loss: 2.6243 Acc: 34.57%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.54it/s]
Val Loss: 2.4351 Acc: 37.14%
Epoch 7/15
----------
Training Epoch 7: 100%|██████████| 990/990 [07:07<00:00,  2.32it/s, loss=2.257, acc=38.56%]
Train Loss: 2.4401 Acc: 38.56%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.54it/s]
Val Loss: 2.0853 Acc: 44.62%
Epoch 8/15
----------
Training Epoch 8: 100%|██████████| 990/990 [07:09<00:00,  2.57it/s, loss=2.009, acc=41.88%]Train Loss: 2.2889 Acc: 41.88%
Training Epoch 8: 100%|██████████| 990/990 [07:09<00:00,  2.31it/s, loss=2.009, acc=41.88%]
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.54it/s]
Val Loss: 2.2410 Acc: 43.58%
Epoch 9/15
----------
Training Epoch 9: 100%|██████████| 990/990 [07:09<00:00,  2.31it/s, loss=2.461, acc=44.89%]
Train Loss: 2.1590 Acc: 44.89%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.53it/s]
Val Loss: 1.9679 Acc: 48.90%
Epoch 10/15
----------
Training Epoch 10: 100%|██████████| 990/990 [07:09<00:00,  2.31it/s, loss=2.228, acc=47.82%]
Train Loss: 2.0388 Acc: 47.82%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.55it/s]
Val Loss: 2.0474 Acc: 49.02%
Epoch 11/15
----------
Training Epoch 11: 100%|██████████| 990/990 [07:08<00:00,  2.31it/s, loss=2.200, acc=49.81%]
Train Loss: 1.9464 Acc: 49.81%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.53it/s]
Val Loss: 1.8256 Acc: 52.64%
Epoch 12/15
----------
Training Epoch 12: 100%|██████████| 990/990 [07:08<00:00,  2.31it/s, loss=1.535, acc=51.82%]
Train Loss: 1.8620 Acc: 51.82%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.51it/s]
Val Loss: 1.6620 Acc: 55.20%
Epoch 13/15
----------
Training Epoch 13: 100%|██████████| 990/990 [07:05<00:00,  2.33it/s, loss=1.727, acc=53.77%]
Train Loss: 1.7814 Acc: 53.77%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.60it/s]
Val Loss: 1.5798 Acc: 56.60%
Epoch 14/15
----------
Training Epoch 14: 100%|██████████| 990/990 [06:59<00:00,  2.36it/s, loss=1.640, acc=55.21%]
Train Loss: 1.7227 Acc: 55.21%
Validation: 100%|██████████| 40/40 [00:15<00:00,  2.61it/s]
Val Loss: 1.6558 Acc: 56.44%
Epoch 15/15
----------
Training Epoch 15: 100%|██████████| 990/990 [06:58<00:00,  2.36it/s, loss=1.425, acc=56.64%]Train Loss: 1.6666 Acc: 56.64%

Validation: 100%|██████████| 40/40 [00:15<00:00,  2.63it/s]
Val Loss: 1.5490 Acc: 58.26%
