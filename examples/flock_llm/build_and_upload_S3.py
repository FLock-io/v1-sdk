import hashlib
import io
import os
import subprocess
import tempfile

import boto3
from loguru import logger

class S3StorageManager:
    def __init__(self, images_bucket_name: str) -> None:
        self.client = boto3.client("s3")
        self.images_bucket_name = images_bucket_name

    def hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def generate_file_name_in_hash(self, filepath: str) -> str:
        with open(filepath, "rb") as f:
            return self.hash_bytes(f.read())

    def upload_file(self, file_path: str, _hash):
        try:
            with open(file_path, 'rb') as file:
                self.client.upload_fileobj(file, self.images_bucket_name, _hash)
            logger.success(f"{file_path} is file uploaded successfully")
        except Exception as e:
            logger.error(f"Upload failed to S3, error: {e}")
            raise

def compress_directory(output_file_path: str):
    subprocess.run(["tar", "-czf", output_file_path, "."], check=True)

if __name__ == "__main__":

    S3_MODEL_IMAGES_BUCKET = "flock-fl-image"

    s3_storage_manager = S3StorageManager(images_bucket_name=S3_MODEL_IMAGES_BUCKET)

    # Create a temp file name
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        target_file_path = None

    try:
        logger.info("Compressing files..")
        compress_directory(temp_file_path)

        logger.info("Generating file name based on hash..")
        upload_file_name_in_hash = s3_storage_manager.generate_file_name_in_hash(temp_file_path)

        # Generate the target file path
        target_file_path = os.path.join(os.path.dirname(temp_file_path), upload_file_name_in_hash)

        if os.path.exists(target_file_path):
            os.remove(target_file_path)

        # Rename the temp file to the target file path
        os.rename(temp_file_path, target_file_path)
        logger.info(f"File {temp_file_path} renamed to: {target_file_path}")

        logger.info("Uploading the file to S3..")
        s3_storage_manager.upload_file(target_file_path, upload_file_name_in_hash)

        logger.warning(f"Please keep this hash code for the FLock client to download the model, hash: {upload_file_name_in_hash}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if target_file_path is not None and os.path.exists(target_file_path) and temp_file_path != target_file_path:
            os.remove(target_file_path)


