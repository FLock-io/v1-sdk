import os
import tempfile

from loguru import logger
from s3_storage_manager import S3StorageManager, S3_MODEL_IMAGES_BUCKET
from utils.file_operations import compress_directory, get_file_size

if __name__ == "__main__":

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
