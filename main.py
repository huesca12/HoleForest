import click
import colorama
from colorama import Fore
import joblib
import numpy as np
import os
import pandas as pd
import sklearn

DEFAULT_MODEL_PATH = f"{os.path.abspath(os.path.dirname(__file__))}" + \
                     "/model/model.joblib"
PARAM_LIST = ["peakFreq", "snr", "amplitude", "centralFreq",
              "duration", "bandwidth", "Q-value"]
model = count_ = printout_ = info = warn = success = ...


class HoleForestException(Exception):
    """Base exception class"""
    def __str__(self):
        return {Fore.RED}, \
               super(HoleForestException, self).__str__(), \
               {Fore.RESET}


class InvalidPath(HoleForestException, FileNotFoundError):
    """Raised when a path to a file does not exist"""


class MissingColumns(HoleForestException):
    """Raised when an input CSV does not have the necessary columns"""


class CountTooHigh(HoleForestException):
    """Raised when the count option is greater than the number of labels"""


def _info(msg, *args, **kwargs):
    print(f"[~] {msg}", *args, **kwargs)


def _warn(msg, *args, **kwargs):
    print(f"{Fore.YELLOW}[!] {msg}{Fore.RESET}", *args, **kwargs)


def _success(msg, *args, **kwargs):
    print(f"{Fore.GREEN}[+] {msg}{Fore.RESET}", *args, **kwargs)


def validate_path(path):
    if not os.path.exists(path):
        warn(f"Path to file ({path}) not found.")
        raise InvalidPath(f"{path} does not exist")


def validate_extension(path, ext):
    if not path.endswith(ext):
        warn(f"File ({path}) does not have the {ext} extension.")
        return False
    return True


def validate_dataframe(dataframe, file):
    info(f"Verifying DataFrame columns...")
    if not all(param in dataframe for param in PARAM_LIST):
        warn("DataFrame failed verification.")
        raise MissingColumns(
            f"DataFrame ({file}) missing necessary column(s)."
        )
    success("Verified DataFrame structure!")


def run_model(df, output):
    guesses = model.predict_proba(df[PARAM_LIST])

    info("Running model predictions...")
    predictions = model.classes_[np.argsort(guesses)[:, :-count_ - 1:-1]]
    predictions = [
        [preds[i] for preds in predictions] for i in range(count_)
    ]
    success("Extracted predictions.")

    info("Running model confidence...")
    probabilities = [
        sorted(probas, reverse=True)[:count_] for probas in guesses
    ]
    probabilities = [
        [probas[i] for probas in probabilities] for i in range(count_)
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

    if printout_:
        print(output_df)
    elif output is None:
        warn("No output path specified.")
        if click.confirm("Would you like to print the output DataFrame?"):
            print(output_df)
    else:
        if not validate_extension(output, ".csv"):
            output = f"{output}.csv"
            warn(f"Added CSV extension (new output: {output}).")
        if os.path.exists(output):
            click.confirm(
                f"Output path {output} already exists."
                "Do you want to overwrite?",
                abort=True
            )
        output_df.to_csv(output, index=False)
        success(f"Saved output to {output}")


@click.group()
@click.option("-v", "--verbose",
              is_flag=True, help="Print verbose/debug messages")
def main(verbose):
    global info, warn, success
    info = _info if verbose else lambda *args, **kwargs: None
    warn = _warn if verbose else lambda *args, **kwargs: None
    success = _success if verbose else lambda *args, **kwargs: None
    colorama.init()


@main.command(help="Train an ML model from CSV data set")
@click.argument("file")
@click.argument("output")
def train(file, output):
    np.random.seed(12)
    info(f"Loading input CSV {file}...")
    validate_path(file)
    validate_extension(file, ".csv")
    df = pd.read_csv(file)
    validate_dataframe(df, file)
    success(f"Loaded {file}!")
    if not validate_extension(output, ".joblib"):
        output = f"{output}.joblib"
        warn(f"Added .joblib extension (new output: {output}).")
    info("Training model...")
    success("Trained new model successfully!")


@main.group(help="Predict glitch(es) type(s)")
@click.option("--model-path", "-m", help="Path to ML model",
              default=DEFAULT_MODEL_PATH, metavar="", show_default=True)
@click.option("--printout", "-p", is_flag=True, help="Print output DataFrame")
@click.option("--count", "-c", help="How many predictions to extract",
              default=3, show_default=True)
def predict(model_path, printout, count):
    global model, count_, printout_
    count_ = count
    printout_ = printout

    info(f"Loading model from {model_path}...")
    validate_path(model_path)
    validate_extension(model_path, ".joblib")
    model = joblib.load(model_path)
    if count > len(model.classes_):
        raise CountTooHigh(
            f"Count ({count}) is larger than "
            f"# of labels ({len(model.classes_)})"
        )
    success(f"Successfully loaded model.")


@predict.command(help="Load CSV file of glitches")
@click.argument("file")
@click.option("--output", "-o", help="CSV output file path", metavar="")
@click.option("--delete-extras", "-d", is_flag=True,
              help="Remove extra columns from input to output")
def csv(file, output, delete_extras):
    info(f"Loading {file} into DataFrame...")
    validate_path(file)
    validate_extension(file, ".csv")
    df = pd.read_csv(file)
    validate_dataframe(df, file)
    df = pd.read_csv(file)[PARAM_LIST] if delete_extras else df
    success(f"{file} successfully loaded.")

    info(f"Starting model...")
    run_model(df, output)


@predict.command(help="Input parameters of one glitch")
@click.option("--peak-freq", "-p", type=float, metavar="")
@click.option("--snr", "-s", type=float, metavar="")
@click.option("--amplitude", "-a", type=float, metavar="")
@click.option("--central-freq", "-c", type=float, metavar="")
@click.option("--duration", "-d", type=float, metavar="")
@click.option("--bandwidth", "-b", type=float, metavar="")
@click.option("--q-value", "-q", type=float, metavar="")
@click.option("--output", "-o", help="CSV output file path", metavar="")
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
    params = [peak_freq, snr, amplitude, central_freq,
              duration, bandwidth, q_value]

    info(f"Loading parameters into DataFrame...")
    df = pd.DataFrame(
        {param: val for (param, val) in zip(PARAM_LIST, params)}, index=[0]
    )
    success(f"Parameters successfully loaded.")

    info(f"Starting model...")
    run_model(df, output)


@main.resultcallback()
def process_result(_, **__):
    colorama.deinit()
    success(f"Finished")


if __name__ == "__main__":
    main()
