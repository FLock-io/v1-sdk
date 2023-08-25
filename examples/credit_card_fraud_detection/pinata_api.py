import os
from dotenv import load_dotenv
from pinatapy import PinataPy

load_dotenv()

PINATA_API_KEY = os.getenv("PINATA_API_KEY")
PINATA_SECRET_API_KEY = os.getenv("PINATA_SECRET_API_KEY")

def pin_file_to_ipfs(path_to_file):
    pinata = PinataPy(PINATA_API_KEY, PINATA_SECRET_API_KEY)
    response = pinata.pin_file_to_ipfs(path_to_file, "/", False)
    return response

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pinata_api.py <path_to_file>")
        sys.exit(1)

    path_to_file = sys.argv[1]
    response = pin_file_to_ipfs(path_to_file)
    print(response)
