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
**Recommended**: The model file generated on one machine may not be loaded properly on a different machine. So, the user may train a model using a supplifed data set with the following command:
```bash
python main.py train model/train.csv trained_model.joblib
```
Keep in mind, a model **has** been provided at `model/model.joblib`, but it is highly recommended and sometimes necessary to train on a new machine.

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

`main.py`

&nbsp;&nbsp;&nbsp;&nbsp;`-v/--verbose` | Flag<br>
&nbsp;&nbsp;&nbsp;&nbsp;If enabled, print extra/debug messages

&nbsp;&nbsp;&nbsp;&nbsp;`predict` | Command<br>
&nbsp;&nbsp;&nbsp;&nbsp;Classify glitches command (more info below)

&nbsp;&nbsp;&nbsp;&nbsp;`train` | Command<br>
&nbsp;&nbsp;&nbsp;&nbsp;Train a new ML model from an input data set (more info below)

### Predict

The `main.py` file has a command `predict` which utilizes a machine learning model (Random Forest) to classify glitches based on 7 characteristics.

`main.py predict`

&nbsp;&nbsp;&nbsp;&nbsp;`-m/--model` | Option (str)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Path to model file (defulat: model/model.joblib)

&nbsp;&nbsp;&nbsp;&nbsp;`-c/--count` | Option (int)<br>
&nbsp;&nbsp;&nbsp;&nbsp;How many glitch (and probability) guesses to make

&nbsp;&nbsp;&nbsp;&nbsp;`-p/--printout` | Flag<br>
&nbsp;&nbsp;&nbsp;&nbsp;If enabled, print the output DataFrame

&nbsp;&nbsp;&nbsp;&nbsp;`csv` | Command<br>
&nbsp;&nbsp;&nbsp;&nbsp;Classify a CSV of glitches

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`FILE` | Argument (str)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CSV of glitches to classify

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`-o/--output` | Option (str)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Path to file for saving output DataFrame

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`-d/--delete-extras` | Flag <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Do not preserve extraneous columns in output

&nbsp;&nbsp;&nbsp;&nbsp;`glitch` | Command<br>
&nbsp;&nbsp;&nbsp;&nbsp;Classify one glitch by inputting 7 values

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`-o/--output` | Option (str)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Path to file for saving output DataFrame

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For inputting parameter values, the user may use options.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If options are not supplied, the user is prompted.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python main.py predict glitch --help` for more.<br>

### Train

The `main.py` file has a command `train` which trains a new model based on an input data set.

`main.py train`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`FILE` | Argument (str)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CSV of glitches on which to train

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`OUTPUT` | Argument (str)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Path to output .joblib model file

**NOTE**: The input CSV for both the `predict` and `train` commands **must** have the following columns:
```
"peakFreq", "snr", "amplitude", "centralFreq", "duration", "bandwidth", "Q-value"
```
These columns are necessary and hardcoded into the script as they represent the 7 glitch parameters. If these columns are not present, errors will be raised.
