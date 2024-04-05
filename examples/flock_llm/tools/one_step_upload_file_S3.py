import hashlib
import sys
import os
import subprocess
import tempfile
import threading

import boto3
from boto3.s3.transfer import TransferConfig
from loguru import logger
from tqdm import tqdm

def install_pigz():
    try:
        subprocess.check_call(['sudo', 'apt-get', 'update'])
        subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'pigz'])
        logger.success("pigz installed successfully.")
    except subprocess.CalledProcessError:
        logger.error("Failed to install pigz.")
        sys.exit(1)

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._progress_bar = tqdm(total=self._size, unit='B', unit_scale=True, desc=f"Uploading {filename}")

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            self._progress_bar.update(bytes_amount)

    def __del__(self):
        self._progress_bar.close()

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
            config = TransferConfig(use_threads=True)
            with open(file_path, 'rb') as file:
                self.client.upload_fileobj(
                    file,
                    self.images_bucket_name,
                    _hash,
                    Callback=ProgressPercentage(file_path),
                    Config=config
                )
            logger.success(f"{file_path} is file uploaded successfully")
        except Exception as e:
            logger.error(f"Upload failed to S3, error: {e}")
            raise

# def compress_directory(output_file_path: str):
#     subprocess.run(["tar", "-czf", output_file_path, "."], check=True)

# Efficient compression using pigz
def compress_directory(output_file_path: str, target_directory="."):
    subprocess.run(f"tar -cf - {target_directory} | pigz > {output_file_path}", stdout=open(output_file_path, "wb"), shell=True, check=True)

def get_file_size(file_path):
    '''
        Calculate the file size and return it in a human-readable format.
    '''
    file_size_bytes = os.path.getsize(file_path)

    file_size_kb = file_size_bytes / 1024

    if file_size_kb < 1024:
        return f"{file_size_kb:.2f} KB"
    elif file_size_kb < 1024 * 1024:
        return f"{file_size_kb / 1024:.2f} MB"
    else:
        return f"{file_size_kb / (1024 * 1024):.2f} GB"

if __name__ == "__main__":
    # Install pigz
    install_pigz()

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

        # Print the size of the compressed file
        file_size_str = get_file_size(target_file_path)
        logger.info(f"The size of the compressed file is: {file_size_str}")

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


