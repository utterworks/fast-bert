#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
IMAGE="fluent-fast-bert"

TAG="$1"

# parameters
FASTAI_VERSION="1.0"
PY_VERSION="py36"

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

chmod +x bert_batch/train

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${IMAGE}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${IMAGE}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)
# aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 579360261297.dkr.ecr.eu-west-1.amazonaws.com/fluent-fast-bert

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 520713654638 --region ${region} --no-include-email)

# loop for each architecture (cpu & gpu)
for arch in gpu
do  
    echo "Building image with arch=${arch}, region=${region}"
    
    FULLNAME="${account}.dkr.ecr.${region}.amazonaws.com/${IMAGE}:${TAG}-batch"
    docker build -t ${IMAGE}:${TAG}-batch --build-arg ARCH="$arch" -f "batch.Dockerfile" .
    docker tag ${IMAGE}:${TAG}-batch ${FULLNAME}
    docker push ${FULLNAME}
done
