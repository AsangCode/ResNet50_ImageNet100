#!/bin/bash
set -e

# Load configuration
source config.sh

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
log() { echo -e "${GREEN}[STATUS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

###########################################
# 1. AWS Connection Check
###########################################
log "Checking AWS connections..."

# Check AWS CLI and credentials
log "Checking AWS credentials..."
aws sts get-caller-identity > /dev/null 2>&1 || error "AWS CLI not configured"

# Check S3 bucket access and dataset
log "Checking S3 bucket access..."
aws s3 ls "s3://$S3_BUCKET" --region $AWS_DEFAULT_REGION > /dev/null 2>&1 || error "Cannot access S3 bucket $S3_BUCKET"

# Verify dataset path in S3
DATASET_PATH="s3://$S3_BUCKET/data/$DATASET_ZIP"
log "Checking dataset at: $DATASET_PATH"
aws s3 ls "$DATASET_PATH" --region $AWS_DEFAULT_REGION > /dev/null 2>&1 || error "Dataset not found at $DATASET_PATH"

# Check EC2 connection
log "Checking EC2 connection..."
ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    echo 'Testing EC2 connection...'
    # Check if AWS CLI is installed on EC2
    if ! command -v aws &> /dev/null; then
        echo 'Installing AWS CLI...'
        sudo apt-get update
        sudo apt-get install -y unzip
        curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
    aws --version
" || error "Cannot connect to EC2"

log "All connections verified successfully!"

###########################################
# 2. Dataset Management in S3
###########################################
log "Checking dataset in S3..."
if aws s3 ls "s3://$S3_BUCKET/data/$DATASET_ZIP" > /dev/null 2>&1; then
    log "Dataset already exists in S3"
else
    if [ -f "$DATASET_ZIP" ]; then
        log "Uploading dataset to S3..."
        aws s3 cp "$DATASET_ZIP" "s3://$S3_BUCKET/data/" || error "Failed to upload dataset"
    else
        error "Dataset file not found locally or in S3"
    fi
fi

###########################################
# 3. Find and Use Largest Volume on EC2
###########################################
log "Finding and setting up largest volume..."

# Find and setup largest volume
log "Finding and setting up largest volume..."
ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    # Find largest volume
    echo 'Checking available volumes...'
    df -h
    
    # Find largest volume by available space
    LARGEST_VOL=\$(df -h | grep -v tmpfs | grep -v loop | sort -hr -k4 | head -n1)
    MOUNT_POINT=\$(echo \$LARGEST_VOL | awk '{print \$6}')
    AVAIL_SPACE=\$(echo \$LARGEST_VOL | awk '{print \$4}')
    echo \"Found largest volume at \$MOUNT_POINT with \$AVAIL_SPACE available\"
    
    # Set up work directory
    WORK_DIR=\"\$MOUNT_POINT/training_data\"
    echo \"Setting up work directory at \$WORK_DIR\"
    
    # Create and set up tmp directory first
    sudo mkdir -p \$MOUNT_POINT/tmp
    sudo chmod 1777 \$MOUNT_POINT/tmp
    export TMPDIR=\$MOUNT_POINT/tmp
    echo \"Using TMPDIR=\$TMPDIR\"
    
    # Create all directories from config
    echo \"Creating directories from config...\"
    for dir in \"\${BASE_DIRS[@]}\"; do
        sudo mkdir -p \"\$WORK_DIR/\$dir\"
        sudo chown ubuntu:ubuntu \"\$WORK_DIR/\$dir\"
        sudo chmod 755 \"\$WORK_DIR/\$dir\"
        echo \"Created \$WORK_DIR/\$dir\"
    done
    
    # Show directory structure
    echo \"Directory structure:\"
    ls -la \$WORK_DIR
    
    # Show available space
    echo \"Available space:\"
    df -h \$WORK_DIR
    
    # Save and output work directory
    echo \$WORK_DIR > \$MOUNT_POINT/tmp/work_dir.txt
    cat \$MOUNT_POINT/tmp/work_dir.txt"

# Get work directory path
scp -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE:$MOUNT_POINT/tmp/work_dir.txt" .
WORK_DIR=$(cat work_dir.txt)
rm work_dir.txt

log "Storage setup completed. Using work directory: $WORK_DIR"

###########################################
# 4. Dataset and Environment Setup
###########################################
log "Setting up environment and dataset..."
ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    # First, find the largest volume
    echo 'Finding largest volume...'
    df -h
    LARGEST_VOL=\$(df -h | grep -v tmpfs | grep -v loop | sort -hr -k4 | head -n1)
    MOUNT_POINT=\$(echo \$LARGEST_VOL | awk '{print \$6}')
    AVAIL_SPACE=\$(echo \$LARGEST_VOL | awk '{print \$4}')
    
    # Set TMPDIR to use largest volume immediately
    export TMPDIR=\"\$MOUNT_POINT/tmp\"
    sudo mkdir -p \$TMPDIR
    sudo chmod 1777 \$TMPDIR  # Same permissions as /tmp
    sudo chown ubuntu:ubuntu \$TMPDIR
    echo \"Found largest volume at \$MOUNT_POINT with \$AVAIL_SPACE available\"

    # Set up work directory in the largest volume
    WORK_DIR=\"\$MOUNT_POINT/training_data\"
    echo \"Setting up work directory at \$WORK_DIR\"
    
    # Create all necessary directories with proper permissions
    for dir in code data venv cache; do
        sudo mkdir -p \$WORK_DIR/\$dir
        sudo chown ubuntu:ubuntu \$WORK_DIR/\$dir
        sudo chmod 755 \$WORK_DIR/\$dir
        echo \"Created \$WORK_DIR/\$dir\"
    done
    
    # Verify directories
    ls -la \$WORK_DIR
    
    # Check if dataset is already properly set up
    cd \$WORK_DIR/data
    echo \"Checking if dataset is already set up...\"
    
    if [ -d \"$DATASET_NAME\" ] && \
       [ -d \"$DATASET_NAME/$DATASET_TRAIN_DIR\" ] && \
       [ -d \"$DATASET_NAME/$DATASET_VAL_DIR\" ]; then
        
        # Count classes to verify dataset integrity
        TRAIN_CLASSES=\$(ls \"$DATASET_NAME/$DATASET_TRAIN_DIR\" | wc -l)
        VAL_CLASSES=\$(ls \"$DATASET_NAME/$DATASET_VAL_DIR\" | wc -l)
        
        if [ \"\$TRAIN_CLASSES\" -eq 100 ] && [ \"\$VAL_CLASSES\" -eq 100 ]; then
            echo \"Dataset already exists and looks valid:\"
            echo \"- Training classes: \$TRAIN_CLASSES\"
            echo \"- Validation classes: \$VAL_CLASSES\"
            echo \"- Location: \$PWD/$DATASET_NAME\"
            echo \"Skipping dataset download and extraction\"
            return 0
        fi
    fi
    
    echo \"Dataset not found or incomplete, proceeding with download...\"
    rm -f $DATASET_ZIP  # Clean up any previous downloads

    # Set up environment variables
    export AWS_ACCESS_KEY_ID='$AWS_ACCESS_KEY_ID'
    export AWS_SECRET_ACCESS_KEY='$AWS_SECRET_ACCESS_KEY'
    export AWS_DEFAULT_REGION='$AWS_DEFAULT_REGION'
    
    # Ensure TMPDIR is set and exists
    export TMPDIR=\$MOUNT_POINT/tmp
    echo \"Setting TMPDIR to \$TMPDIR\"
    sudo mkdir -p \$TMPDIR
    sudo chmod 1777 \$TMPDIR
    sudo chown ubuntu:ubuntu \$TMPDIR
    
    echo 'Downloading dataset from S3...'
    echo 'Using temp directory: '\$TMPDIR
    echo 'Downloading to: '\$PWD
    
    # Verify space again before download
    AVAIL_GB=\$(df -BG . | awk 'NR==2 {print \$4}' | sed 's/G//')
    echo \"Available space before download: \${AVAIL_GB}GB\"
    
    # Get file size from S3 for progress calculation
    SIZE_BYTES=\$(aws s3api head-object --bucket $S3_BUCKET --key data/$DATASET_ZIP --query 'ContentLength' --output text)
    SIZE_MB=\$((\$SIZE_BYTES / 1024 / 1024))
    echo \"Downloading dataset (\$SIZE_MB MB)...\"
    
    # Download the dataset
    echo \"Starting download to \$PWD...\"
    aws s3 cp s3://$S3_BUCKET/data/$DATASET_ZIP . \
        --region $AWS_DEFAULT_REGION
    
    if [ ! -f $DATASET_ZIP ]; then
        echo 'Failed to download dataset'
        exit 1
    fi
    
    echo 'Extracting dataset...'
    echo "Extracting files..."
    unzip -o $DATASET_ZIP
    
    # Verify extraction
    if [ ! -d \"$DATASET_NAME\" ]; then
        echo \"Dataset directory '$DATASET_NAME' not found after extraction\"
        ls -la  # Show what files were extracted
        exit 1
    fi
    
    # Show dataset structure
    echo \"Dataset extracted successfully. Contents:\"
    ls -la $DATASET_NAME/
    echo \"Number of classes:\"
    ls $DATASET_NAME/$DATASET_TRAIN_DIR/ | wc -l
    echo \"Sample of classes:\"
    ls $DATASET_NAME/$DATASET_TRAIN_DIR/ | head -n 5
    
    # Cleanup
    rm -f $DATASET_ZIP
    echo 'Dataset setup completed'" || error "Failed to setup dataset"

###########################################
# 5. Upload Code Files
###########################################
log "Uploading code files..."
TOTAL_FILES=${#PROJECT_FILES[@]}
CURRENT=0

# First ensure the code directory exists and has correct permissions
ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    # Find largest volume again to ensure we have the right path
    LARGEST_VOL=\$(df -h | grep -v tmpfs | grep -v loop | sort -hr -k4 | head -n1)
    MOUNT_POINT=\$(echo \$LARGEST_VOL | awk '{print \$6}')
    AVAIL_SPACE=\$(echo \$LARGEST_VOL | awk '{print \$4}')
    WORK_DIR=\"\$MOUNT_POINT/training_data\"
    
    echo \"Using work directory: \$WORK_DIR\"
    echo \"Available space: \$AVAIL_SPACE\"
    
    # Create and set up code directory with full path
    sudo mkdir -p \$WORK_DIR/code
    sudo chown ubuntu:ubuntu \$WORK_DIR/code
    sudo chmod 755 \$WORK_DIR/code
    echo \"Prepared code directory at \$WORK_DIR/code\"
    
    # Verify directory exists and is writable
    if [ ! -d \"\$WORK_DIR/code\" ] || [ ! -w \"\$WORK_DIR/code\" ]; then
        echo \"Error: Code directory not created or not writable\"
        ls -la \$WORK_DIR
        df -h \$WORK_DIR
        exit 1
    fi
    
    echo \"Directory is ready and writable\"
    ls -la \$WORK_DIR/code
"

# Get work directory path again to ensure consistency
WORK_DIR=$(ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    LARGEST_VOL=\$(df -h | grep -v tmpfs | grep -v loop | sort -hr -k4 | head -n1)
    MOUNT_POINT=\$(echo \$LARGEST_VOL | awk '{print \$6}')
    echo \"\$MOUNT_POINT/training_data\"
")

# Upload files
for file in "${PROJECT_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    SIZE=$(ls -lh "$file" | awk '{print $5}')
    log "[$CURRENT/$TOTAL_FILES] Uploading $file ($SIZE)..."
    
    # Use full path for target
    TARGET_PATH="$WORK_DIR/code/$file"
    log "Uploading to: $TARGET_PATH"
    
    scp -i "$EC2_KEY" "$file" "ubuntu@$EC2_INSTANCE:$TARGET_PATH" || error "Failed to copy $file"
    
    # Verify file was copied
    ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
        if [ ! -f \"$TARGET_PATH\" ]; then
            echo \"Error: File $file not found after copy\"
            ls -la \"$WORK_DIR/code\"
            exit 1
        fi
        echo \"Verified $file was copied successfully\"
    "
done
log "All files uploaded successfully"

###########################################
# 6. Install Requirements
###########################################
log "Installing Python requirements..."

# Get work directory path again to ensure consistency
WORK_DIR=$(ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    LARGEST_VOL=\$(df -h | grep -v tmpfs | grep -v loop | sort -hr -k4 | head -n1)
    MOUNT_POINT=\$(echo \$LARGEST_VOL | awk '{print \$6}')
    echo \"\$MOUNT_POINT/training_data\"
")

ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    cd $WORK_DIR
    echo \"Working in: \$PWD\"
    
    # Set up environment variables for pip
    export TMPDIR=\"\$PWD/tmp\"
    export PIP_CACHE_DIR=\"\$PWD/cache/pip\"
    
    # Create directories with proper permissions
    sudo mkdir -p \"\$TMPDIR\" \"\$PIP_CACHE_DIR\"
    sudo chown ubuntu:ubuntu \"\$TMPDIR\" \"\$PIP_CACHE_DIR\"
    sudo chmod 755 \"\$TMPDIR\" \"\$PIP_CACHE_DIR\"
    
    echo \"Created directories:\"
    ls -la \"\$TMPDIR\" \"\$PIP_CACHE_DIR\"
    
    # Set up apt to use our large volume
    echo \"Setting up apt to use large volume...\"
    sudo rm -rf /var/lib/apt/lists/*
    sudo mkdir -p \"$WORK_DIR/cache/apt/lists\"
    sudo mkdir -p \"$WORK_DIR/cache/apt/archives\"
    
    # Create apt configuration to use our directories
    echo 'Dir::Cache \"$WORK_DIR/cache/apt/\";
Dir::Cache::archives \"archives/\";
Dir::Cache::srcpkgcache \"srcpkgcache.bin\";
Dir::Cache::pkgcache \"pkgcache.bin\";
Dir::State::lists \"$WORK_DIR/cache/apt/lists/\";' | sudo tee /etc/apt/apt.conf.d/99custom-dirs
    
    # Show apt configuration
    echo \"APT will use these directories:\"
    cat /etc/apt/apt.conf.d/99custom-dirs
    
    # Ensure python3-venv is installed
    echo \"Installing python3-venv...\"
    sudo apt-get clean
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip
    
    # Create and activate virtual environment
    echo 'Setting up Python virtual environment...'
    python3 -m venv venv
    source venv/bin/activate
    
    # Verify venv activation
    which python3
    which pip
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install packages with progress
    echo \"Installing Python packages...\"
    if [ ! -f \"code/requirements.txt\" ]; then
        echo \"Error: requirements.txt not found in \$PWD/code/\"
        ls -la code/
        exit 1
    fi
    
    TOTAL_REQS=\$(wc -l < code/requirements.txt)
    echo \"Found \$TOTAL_REQS packages to install\"
    
    # Install with progress bar
    pip install --no-cache-dir -r code/requirements.txt \
        --progress-bar on \
        --disable-pip-version-check" || error "Failed to install requirements"

###########################################
# 7. Run Training
###########################################
log "Starting training..."
ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    cd $WORK_DIR/code
    source ../venv/bin/activate
    
    # Set up environment variables for various library configs
    export MPLCONFIGDIR=\"$WORK_DIR/cache/matplotlib\"
    export XDG_CACHE_HOME=\"$WORK_DIR/cache\"
    export XDG_CONFIG_HOME=\"$WORK_DIR/cache/config\"
    export TORCH_HOME=\"$WORK_DIR/cache/torch\"
    export CUDA_CACHE_PATH=\"$WORK_DIR/cache/cuda\"
    
    # Install CUDA if not present
    if ! command -v nvcc &> /dev/null; then
        echo 'Installing CUDA toolkit...'
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit
    fi
    
    # Verify CUDA installation
    echo 'CUDA version:'
    nvcc --version
    
    # Check GPU availability
    echo 'Checking GPU availability...'
    nvidia-smi || error \"No GPU found! Please ensure you're using a GPU instance\"
    
    # Set CUDA environment variables
    export CUDA_VISIBLE_DEVICES=0
    export TORCH_CUDA_ARCH_LIST=\"7.0;7.5;8.0;8.6\"  # Common architectures
    
    # Create cache directories
    mkdir -p \"\$MPLCONFIGDIR\" \"\$XDG_CACHE_HOME\" \"\$XDG_CONFIG_HOME\" \"\$TORCH_HOME\" \"\$CUDA_CACHE_PATH\"
    
    # Show environment setup
    echo \"Using config directories:\"
    echo \"- Matplotlib: \$MPLCONFIGDIR\"
    echo \"- XDG Cache: \$XDG_CACHE_HOME\"
    echo \"- XDG Config: \$XDG_CONFIG_HOME\"
    echo \"- PyTorch: \$TORCH_HOME\"
    echo \"- CUDA: \$CUDA_CACHE_PATH\"
    
    # Run training
    python train.py" || error "Training failed"

###########################################
# 8. Download Results
###########################################
log "Downloading results..."
mkdir -p results

# Get sizes first
ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "
    cd $WORK_DIR/code
    echo 'Checking result sizes...'
    for dir in checkpoints logs runs; do
        if [ -d \$dir ]; then
            size=\$(du -sh \$dir | cut -f1)
            echo \"\$dir \$size\"
        fi
    done
"

# Download each result directory with progress
for dir in "${RESULT_DIRS[@]}"; do
    if ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "[ -d '$WORK_DIR/code/$dir' ]"; then
        size=$(ssh -i "$EC2_KEY" "ubuntu@$EC2_INSTANCE" "du -sh $WORK_DIR/code/$dir | cut -f1")
        log "Downloading $dir ($size)..."
        rsync -ah --progress -e "ssh -i $EC2_KEY" "ubuntu@$EC2_INSTANCE:$WORK_DIR/code/$dir" results/
    else
        log "No $dir directory found"
    fi
done

log "Training pipeline completed successfully!"
log "Results downloaded to ./results/"
