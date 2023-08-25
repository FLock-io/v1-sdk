# import os
# from dotenv import load_dotenv
# from pinatapy import PinataPy

# load_dotenv()

# PINATA_API_KEY = os.getenv("PINATA_API_KEY")
# PINATA_SECRET_API_KEY = os.getenv("PINATA_SECRET_API_KEY")

# pinata = PinataPy(PINATA_API_KEY, PINATA_SECRET_API_KEY)

# response = pinata.pin_file_to_ipfs(
#     path_to_file="/var/folders/5m/6sgc559j5d19f_nm41wg9t900000gn/T/tmp.V5eZlimQ")

# print(response)

import os
from dotenv import load_dotenv
from pinatapy import PinataPy

load_dotenv()

PINATA_API_KEY = os.getenv("PINATA_API_KEY")
PINATA_SECRET_API_KEY = os.getenv("PINATA_SECRET_API_KEY")

def pin_file_to_ipfs(path_to_file):
    pinata = PinataPy(PINATA_API_KEY, PINATA_SECRET_API_KEY)
    response = pinata.pin_file_to_ipfs(path_to_file)
    return response

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pinata_api.py <path_to_file>")
        sys.exit(1)

    path_to_file = sys.argv[1]
    response = pin_file_to_ipfs(path_to_file)
    print(response)
