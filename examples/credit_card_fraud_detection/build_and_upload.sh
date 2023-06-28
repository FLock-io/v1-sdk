#!/bin/bash
IMAGE_TAG="flock_model"
OUTPUT_FILE=`mktemp`
echo "Building the model image."
docker build -t $IMAGE_TAG .

echo "Saving the docker image to a file and compressing it. It may take a while.."
#time (docker save $IMAGE_TAG | xz -T 0 > $OUTPUT_FILE)
docker save $IMAGE_TAG -o $OUTPUT_FILE

echo "Uploading the compressed image to IPFS.."
json=`curl -F "file=@$OUTPUT_FILE" 127.0.0.1:5001/api/v0/add`
hash=`echo $json | grep -o '"Hash":"[^"]*' | grep -o '[^"]*$'`
rm $OUTPUT_FILE
echo "Model definition IPFS hash: $hash"
