import click
from click import argument as arg, command, group, option as opt
import colorama
from colorama import Fore
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

s = time()
MODEL_PATH = os.path.abspath(os.path.dirname(__file__)) + "/model/model.joblib"
PARAMS = [
    "peakFreq",
    "snr",
    "amplitude",
    "centralFreq",
    "duration",
    "bandwidth",
    "Q-value"
]
# Globals
model = opcount = opprintout = opverbose = ...


class HoleForestException(Exception):
    """Base exception class"""
    def __str__(self):
        return Fore.RED + super().__str__() + Fore.RESET


class InvalidPath(HoleForestException, FileNotFoundError):
    """Raised when a path to a file does not exist"""


class MissingColumns(HoleForestException):
    """Raised when an input CSV does not have the necessary columns"""


class CountTooHigh(HoleForestException):
    """Raised when the count option is greater than the number of labels"""


def info(*args, **kwargs):
    if opverbose:
        print("[~]", *args, **kwargs)


def warn(*args, **kwargs):
    if opverbose:
        print(Fore.YELLOW + "[!]", *args, Fore.RESET, **kwargs)


def success(*args, **kwargs):
    if opverbose:
        print(Fore.GREEN + "[+]", *args, Fore.RESET, **kwargs)


def load_df(file):
    df = pd.read_csv(file)
    info(f"Verifying DataFrame columns...")
    if not all(param in df for param in PARAMS):
        warn("DataFrame failed verification.")
        raise MissingColumns("DataFrame", file, "missing necessary column(s).")
    success("Verified DataFrame structure!")
    return df


def validate_path(path):
    if not os.path.exists(path):
        warn("Path to file", path, "not found.")
        raise InvalidPath(path, "does not exist.")


def validate_extension(path, ext):
    if not path.endswith(ext):
        warn("File", path, "does not have the", ext, "extension.")
        return False
    return True


def run_model(df, output):
    guesses = model.predict_proba(df[PARAMS])

    info("Running model predictions...")
    predictions = model.classes_[np.argsort(guesses)[:, :-opcount - 1:-1]]
    predictions = [
        [preds[i] for preds in predictions] for i in range(opcount)
    ]
    success("Extracted predictions.")

    info("Running model confidence...")
    probabilities = [
        sorted(probas, reverse=True)[:opcount] for probas in guesses
    ]
    probabilities = [
        [probas[i] for probas in probabilities] for i in range(opcount)
    ]
    success("Extracted confidence.")

    info("Initializing output...")
    output_df = df

    info("Writing output predictions...")
    for i, prediction in enumerate(predictions, start=1):
        df[f"prediction{i}"] = prediction
    info("Writing output confidence...")
    for i, proba in enumerate(probabilities, start=1):
        df[f"p{i} confidence"] = proba
    success("Finalized output DataFrame.")

    if opprintout:
        print(output_df)
    elif output is None:
        warn("No output path specified.")
        if click.confirm("Would you like to print the output DataFrame?"):
            print(output_df)

    if output:
        if not validate_extension(output, ".csv"):
            output = f"{output}.csv"
            warn(f"Added CSV extension (new output: {output}).")
        if os.path.exists(output):
            click.confirm(
                f"Output path {output} already exists. "
                "Do you want to overwrite?",
                abort=True
            )
        output_df.to_csv(output, index=False)
        success(f"Saved output to {output}")


@group()
@opt("-v", "--verbose", is_flag=True, help="Print verbose/debug messages")
def main(verbose):
    global opverbose
    opverbose = verbose if verbose else False
    colorama.init()


@main.command()
@arg("file")
@arg("output")
def train(file, output):
    """Train an ML model from CSV data set"""
    np.random.seed(12)

    info(f"Loading input CSV {file}...")
    validate_path(file)
    validate_extension(file, ".csv")
    df = load_df(file)
    success(f"Loaded {file}!")
    if not validate_extension(output, ".joblib"):
        output = f"{output}.joblib"
        warn(f"Added .joblib extension (new output: {output}).")

    info("Training model...")
    df = df.drop(
        columns=["chisq", "chisqDof", "GPStime", "ifo", "imgUrl", "id", "confidence"]
    )
    y = df["label"]
    x = df.drop(columns=["label"])
    x_train, _, y_train, _ = train_test_split(x, y, random_state=0)
    forest = RandomForestClassifier()
    forest.fit(x_train, y_train)
    joblib.dump(forest, output)
    success("Trained new model successfully!")


@main.group()
@opt("--model-path", "-m", help="Path to ML model",
     default=MODEL_PATH, show_default=True)
@opt("--count", "-c", help="How many predictions to extract",
     default=3, show_default=True)
@opt("--printout", "-p", is_flag=True, help="Print output DataFrame")
def predict(model_path, count, printout):
    """Predict glitch(es) type(s)"""
    global model, opcount, opprintout
    opcount = count
    opprintout = printout

    info(f"Loading model from {model_path}...")
    validate_path(model_path)
    validate_extension(model_path, ".joblib")
    model = joblib.load(model_path)
    if count > len(model.classes_):
        raise CountTooHigh(
            f"Count ({count}) is larger than "
            f"# of labels ({len(model.classes_)})."
        )
    success(f"Successfully loaded model.")


@predict.command()
@arg("file")
@opt("--output", "-o", help="CSV output file path")
@opt("--delete-extras", "-d", is_flag=True,
     help="Remove extra columns from input to output")
def csv(file, output, delete_extras):
    """Load CSV file of glitches"""
    info(f"Loading {file} into DataFrame...")
    validate_path(file)
    validate_extension(file, ".csv")
    df = load_df(file)
    df = pd.read_csv(file)[PARAMS] if delete_extras else df
    success(f"{file} successfully loaded.")

    info(f"Starting model...")
    run_model(df, output)


@predict.command()
@opt("--peak-freq", "-p", type=float)
@opt("--snr", "-s", type=float)
@opt("--amplitude", "-a", type=float)
@opt("--central-freq", "-c", type=float)
@opt("--duration", "-d", type=float)
@opt("--bandwidth", "-b", type=float)
@opt("--q-value", "-q", type=float)
@opt("--output", "-o", help="CSV output file path")
def glitch(
        peak_freq,
        snr,
        amplitude,
        central_freq,
        duration,
        bandwidth,
        q_value,
        output
):
    """Input parameters of one glitch"""
    if peak_freq is None:
        peak_freq = click.prompt("Peak Frequency", type=float)
    if snr is None:
        snr = click.prompt("Signal-to-Noise Ratio", type=float)
    if amplitude is None:
        amplitude = click.prompt("Amplitude", type=float)
    if central_freq is None:
        central_freq = click.prompt("Central Frequency", type=float)
    if duration is None:
        duration = click.prompt("Duration", type=float)
    if bandwidth is None:
        bandwidth = click.prompt("Bandwidth", type=float)
    if q_value is None:
        q_value = click.prompt("Q-Value", type=float)
    params = [
        peak_freq,
        snr,
        amplitude,
        central_freq,
        duration,
        bandwidth,
        q_value
    ]

    info(f"Loading parameters into DataFrame...")
    df = pd.DataFrame(
        {param: val for (param, val) in zip(PARAMS, params)}, index=[0]
    )
    success(f"Parameters successfully loaded.")

    info(f"Starting model...")
    run_model(df, output)


@main.resultcallback()
def process_result(_, **__):
    colorama.deinit()
    success("Finished |", time() - s, "seconds.")


if __name__ == "__main__":
    main()
