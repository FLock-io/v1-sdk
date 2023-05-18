# Flock SDK
An SDK for building applications on top of FLock V1

## Creating a new model

Prerequisites:
You must have `docker` and `docker compose` [installed](https://docs.docker.com/engine/install/) and set up to [run as non-root user](https://docs.docker.com/engine/security/rootless/).


### Installing Flock SDK as a package
This step is only required if you want to test invoking the model without the client running it in a docker container.
Installing the SDK as a python package is pretty simple, all you need to do is run `pip install flock-sdk`.


### If you are running a local flock chain
1. Modify the example file and implement the evaluate, train and aggregate functions as required.
2. Make sure that IPFS is started by running `docker-compose up` **in the client directory**
2. Run `./build_and_upload.sh` to build and upload the model definition to IPFS
3. Modify the `MODEL_DEFINITION_HASH` in the `contracts/deploy.sh` file **in the client directory** to match the returned hash
4. Rebuild the contracts using `docker-compose build` and restart the network using `docker-compose restart`
5. Launch the clients using instructions in the client directory's README.

### If you are running on the official flock chain
1. Modify the example file and implement the evaluate, train and aggregate functions as required.
2. Run `./build_and_upload.sh` to build and upload the model definition to IPFS
3. Create a new FlockTask with the returned IPFS hash