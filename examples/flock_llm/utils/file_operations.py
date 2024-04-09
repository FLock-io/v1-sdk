import subprocess
import os
import sys
import mimetypes
import zipfile
import tarfile
try:
    import rarfile
except ImportError:
    rarfile = None
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

# def compress_directory(output_file_path: str):
#     subprocess.run(["tar", "-czf", output_file_path, "."], check=True)

# Efficient compression using pigz
def compress_directory(output_file_path: str, target_directory="."):
    # Install pigz
    install_pigz()

    subprocess.run(f"tar -cf - {target_directory} | pigz > {output_file_path}", stdout=open(output_file_path, "wb"), shell=True, check=True)

def get_member_size(members):
    return sum(member.size for member in members)

def extract_with_progress(archive, members, extract_to, total_size, progress_prefix="Extracting"):
    with tqdm(total=total_size, desc=progress_prefix, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        for member in members:
            archive.extract(member, path=extract_to)
            pbar.update(member.size)

def extract_with_progress_in_files(archive, members, extract_to, progress_prefix="Extracting"):
    total_files = len(members)
    with tqdm(total=total_files, desc=progress_prefix, unit='file') as pbar:
        for member in members:
            archive.extract(member, path=extract_to)
            pbar.update(1)

def extract_file(file_path, extract_to, final_directory_path):
    current_working_directory = os.getcwd()
    target_directory_abs_path = os.path.join(current_working_directory, extract_to)
    final_directory_abs_path = os.path.join(extract_to, final_directory_path)

    if os.path.exists(final_directory_abs_path):
        logger.warning(f"Target directory in absolute path '{final_directory_abs_path}' already exists. Skipping extraction.")
        return

    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type in ['application/x-tar', 'application/gzip', 'application/x-gzip']:
        with tarfile.open(file_path, 'r:*') as tar_ref:
            members = tar_ref.getmembers()
            total_size = get_member_size(members)
            extract_with_progress(tar_ref, members, extract_to, total_size, "Extracting TAR/GZ")
            # extract_with_progress_in_files(tar_ref, members, extract_to, "Extracting TAR/GZ")
    elif mime_type in ['application/zip']:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            total_size = get_member_size(members)
            extract_with_progress(zip_ref, members, extract_to, total_size, "Extracting ZIP")
            # extract_with_progress_in_files(zip_ref, members, extract_to, "Extracting ZIP")
    elif mime_type in ['application/x-rar'] and rarfile:
        with rarfile.RarFile(file_path) as rar_ref:
            members = rar_ref.infolist()
            total_size = get_member_size(members)
            extract_with_progress(rar_ref, members, extract_to, total_size, "Extracting RAR")
            # extract_with_progress_in_files(rar_ref, members, extract_to, "Extracting RAR")
    else:
        logger.warning(
            f"Detected type in unsupported file format {mime_type}, using default. Or required library (rarfile) not installed.")
        with tarfile.open(file_path, 'r:*') as tar_ref:
            members = tar_ref.getmembers()
            total_size = get_member_size(members)
            extract_with_progress(tar_ref, members, extract_to, total_size, "Extracting TAR/GZ")
            # extract_with_progress_in_files(tar_ref, members, extract_to, "Extracting TAR/GZ")

    logger.info(f"Extracted {file_path} to absolute path: {target_directory_abs_path}")

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
