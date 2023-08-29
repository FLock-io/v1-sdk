#!/bin/bash
set -e

OUTPUT_FILE=`mktemp`

time (tar -czf $OUTPUT_FILE .)

echo "Uploading the compressed archive to IPFS.."
# json=`curl -F "file=@$OUTPUT_FILE" ipfs.flock.io/api/v0/add`

# Uncomment if you'd like to upload to your local IPFS
json=`curl -F "file=@$OUTPUT_FILE" 127.0.0.1:5001/api/v0/add`

hash=`echo $json | grep -o '"Hash":"[^"]*' | grep -o '[^"]*$'`
rm $OUTPUT_FILE
echo "Model definition IPFS hash: $hash"
