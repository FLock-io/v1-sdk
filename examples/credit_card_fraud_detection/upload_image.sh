#!/bin/bash
set -e

IMAGE_TAG="flock_model"
OUTPUT_FILE=$(mktemp)
echo "Building the model image."
docker build -t $IMAGE_TAG .

time (tar -czf $OUTPUT_FILE .)

# Use the pinata_api.py script to pin the file to IPFS
echo "Uploading the compressed image to IPFS.."
response=$(python pinata_api.py "$OUTPUT_FILE")

# Extract the IpfsHash from the response using Python
echo "Extracting IpfsHash.."
ipfs_hash=$(python -c "import json; data = $response; print(data.get('IpfsHash', ''))")
echo "Model definition IPFS hash: $ipfs_hash"

# Clean up the temporary output file
rm $OUTPUT_FILE
