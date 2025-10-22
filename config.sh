#!/bin/bash

# EC2 Configuration
export EC2_INSTANCE="ec2-13-203-66-90.ap-south-1.compute.amazonaws.com"
export EC2_KEY="E:/GitHub_Repo/ERA_Assignment/Colab_ResNet50/ResNet50_ImageNet100.pem"

# AWS Configuration
export AWS_DEFAULT_REGION="ap-south-1"
export S3_BUCKET="my-resnet50-training-bucket"

# Dataset Configuration
export DATASET_ZIP="archive.zip"
export DATASET_NAME="ImageNet100"  # Directory name in the zip file
export DATASET_TRAIN_DIR="train"   # Training data subdirectory
export DATASET_VAL_DIR="val"       # Validation data subdirectory

# Directory Structure
export BASE_DIRS=(
    "code"      # For Python scripts
    "data"      # For dataset storage
    "venv"      # For Python virtual environment
    "cache"     # For pip and other caches
    "tmp"       # For temporary files
    "results"   # For training outputs
)

# Python Environment
export PYTHON_VERSION="python3"
export VENV_NAME="venv"
export PIP_NO_CACHE_DIR="true"

# Training Configuration
export BATCH_SIZE=128
export NUM_WORKERS=2
export NUM_EPOCHS=100

# Required Project Files
export PROJECT_FILES=(
    "train.py"
    "model.py"
    "dataset.py"
    "requirements.txt"
    "lr_finder.py"
)

# Result Directories to Save
export RESULT_DIRS=(
    "checkpoints"
    "logs"
    "runs"
    "models"
)