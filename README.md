# Flock SDK
<p align="center">
<a href=""><img src="assets/flock_logo.png" alt="logo" width="350px"></a>
</p>
<p align="center">
<img src="https://img.shields.io/badge/python-3.11-blue?style=round-square&logo=Python&color=3776AB" alt="Python" >
<img src="https://img.shields.io/badge/pytorch-latest-orange?style=round-square&logo=PyTorch&color=EE4C2C" alt="Pytorch" >
<a href="https://timothyshen1.gitbook.io/flock.io/"><img src="https://img.shields.io/badge/document-English-blue.svg" alt="EN doc"></a>
<a href="https://discord.gg/ay8MnJCg2W"><img src="https://dcbadge.vercel.app/api/server/ay8MnJCg2W?style=flat&theme=discord-inverted" alt="Discord Follow"></a>
<a href="https://twitter.com/flock_io"><img src="https://img.shields.io/twitter/follow/flock_io?style=social" alt="Twitter Follow"></a>
</p>

Welcome to the Flock SDK repository! 🚀 Flock SDK is a powerful software development kit that leverages Federated Learning and Blockchain to enable data and computation resources owners to collaboratively train machine learning models using any source data. With Flock SDK, you can define and start ML training tasks on the Flock platform efficiently. 💪


## Key Features

1. 🔗 **Federated Learning**: Flock SDK harnesses the power of Federated Learning, a distributed approach that allows data owners to train machine learning models collaboratively while keeping their data locally.  

2. 🤝 **Collaborative Training**: By sharing source data, Flock enables multiple participants to contribute to the training process, resulting in improved model performance and accuracy.  
3. 💰 **Rewards and Smart Contracts**: Flock's ML training participants are incentivized with rewards and penalties specified by pre-defined smart contracts, ensuring fair and transparent compensation.
4. 🔒 **Secure and Privacy-preserving**: Flock SDK prioritizes data privacy and security, allowing data owners to retain control over their sensitive information throughout the training process.
5. 🧩 **Flexible Integration**: The SDK is designed to be easily integrated into your existing workflows and systems, making it a perfect fit for a wide range of applications.

## Example Usages
1. 🌐🤖 **Flock Large Language Model**: Visit the [`flock_llm`](examples/flock_llm) directory for an example usage of Flock SDK with the Flock Large Language Model(LLM). This demonstrates how to finetune a **Vicuna-7B** using the instruction sets provided by different contributors and train a [`LoRA`](https://arxiv.org/abs/2106.09685) adapeter using federated leaning on chain. 

2. 🛡️💳 **Credit Card Fraud Detection**: Check out the [`credit_card_fraud_detection`](examples/credit_card_fraud_detection) directory for an example usage of Flock SDK to train a fraud detection model using a federated learning approach.

3. 📸🔎 **MobileNet Example**: Explore the [`mobilenet_example`](examples/mobilenet_example) directory for an example implementation of Flock SDK to train a MobileNet model for image classification, leveraging the power of federated learning. 

## Quick Start
### Creating a new model

Prerequisites:
You must have `docker` and `docker compose` [installed](https://docs.docker.com/engine/install/) and set up to [run as non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).


### Installing Flock SDK as a package
This step is only required if you want to test invoking the model without the client running it in a docker container.
Installing the SDK as a python package is pretty simple, all you need to do is run `pip install flock-sdk`.


### If you are running a local flock chain
1. Choose the appropriate example to build from inside the `examples` folder and implement the evaluate, train and aggregate functions as required.
2. Make sure that IPFS is started by running `docker-compose up` **in the client directory**
2. Run `./upload_image.sh` inside the chosen example folder to build and upload the model definition to IPFS
3. Modify the `MODEL_DEFINITION_HASH` in the `Makefile` **inside the client repo** to match the returned hash
4. Start the network by running `make chain` **inside the client repo**
5. Launch the clients using instructions **in the client directory's** README.

### If you are running on the official flock chain
1. Choose the appropriate example to build from inside the `examples` folder and implement the evaluate, train and aggregate functions as required.
2. Run `./upload_image.sh` inside the chosen example folder to build and upload the model definition to IPFS
3. Create a new FlockTask with the returned IPFS hash in the Flock frontend