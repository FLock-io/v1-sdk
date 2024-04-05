import hashlib
import os
import threading
import typing
import boto3
from boto3.s3.transfer import TransferConfig
from loguru import logger
from tqdm import tqdm

S3_MODEL_IMAGES_BUCKET = "flock-fl-image"


class ProgressPercentage(object):
    def __init__(self, filename, size: float, action: str):
        self._filename = filename
        self._size = size
        self._seen_so_far = 0
        self._lock = threading.Lock()
        action_desc = "Uploading" if action == "upload" else "Downloading"
        self._progress_bar = tqdm(total=self._size, unit='B', unit_scale=True, desc=f"{action_desc} {filename}")

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
                    Callback=ProgressPercentage(file_path, os.path.getsize(file_path), "upload"),
                    Config=config
                )
            logger.success(f"{file_path} file uploaded successfully")
        except Exception as e:
            logger.error(f"Upload failed to S3, error: {e}")
            raise

    def download_file(self, _hash: str, directory_path: str) -> None:
        file_path = os.path.join(directory_path, _hash)

        try:
            response = self.client.head_object(Bucket=self.images_bucket_name, Key=_hash)
            size = response['ContentLength']

            with open(file_path, 'wb') as file:
                self.client.download_fileobj(
                    self.images_bucket_name,
                    _hash,
                    file,
                    Callback=ProgressPercentage(file_path, size, "download")
                )
            logger.success(f"{file_path} file downloaded successfully")
        except Exception as e:
            logger.error(f"Download failed from S3, error: {e}")
            raise
