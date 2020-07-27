# HoleForest
## Introduction
HoleForest is a terminal-based tool for performing bulk and specific classification tasks on LIGO glitch metadata using a random forest model. It comes pre-packaged with a default model trained on the o3a and o3b LIGO datasets but has the capacity to train and use a new model given a user-provided glitch dataset in csv format.

## Installation
There are two ways to install HoleForest on your machine:
1. Manual Installation
2. Docker Image

### Manual Installation
Begin by opening the directory where you would like to install HoleForest in your terminal. Then clone the most recent version of this repository using the following command:
```bash
git clone https://github.com/huesca12/HoleForest.git
```
Install the required dependencies into your working environment by running the following command in the same directory as the requirements.txt file:
```bash
pip install -r requirements.txt
```
You should now be able to run HoleForest by using the command in the same directory as ```main.py```: 
```bash
python main.py
```
### Docker Image
Ensure that Docker is installed in your working enviroment using:
```bash
docker info
```
Get the latest image of HoleForest from Docker Hub:
```bash
docker pull huesca12/holeforest:latest
```
## Features
