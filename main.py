import click
import colorama
from colorama import Fore
import joblib
import numpy
import os
import pandas
import sklearn

DEFAULT_MODEL_PATH = f"{os.path.abspath(os.path.dirname(__file__))}/model/model.joblib"
PARAM_LIST = ["peakFreq", "snr", "amplitude", "centralFreq", "duration", "bandwidth", "Q-value"]
model = printout_ = info = warn = success = ...  # All will be modified before use


class HoleForestException(Exception):
    """Base exception class"""
    def __str__(self):
        return f"{Fore.RED}{super(HoleForestException, self).__str__()}{Fore.RESET}"


class InvalidPath(HoleForestException, FileNotFoundError):
    """Raised when a path to a file does not exist"""


class MissingColumns(HoleForestException):
    """Raised when an input CSV does not have the necessary columns"""


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
        raise MissingColumns(f"DataFrame ({file}) missing necessary column(s).")
    success("Verified DataFrame structure!")


def run_model(df, output, num_preds=3):
    params_only = df[PARAM_LIST]
    info("Running model predictions...")
    predictions = model.classes_[numpy.argsort(model.predict_proba(params_only))[:, :-num_preds - 1:-1]]
    prediction1 = [preds[0] for preds in predictions]
    prediction2 = [preds[1] for preds in predictions]
    prediction3 = [preds[2] for preds in predictions]
    success("Extracted predictions.")
    info("Running model confidence...")
    probas = model.predict_proba(params_only)[:, 1]
    proba1 = ...
    proba2 = ...
    proba3 = ...
    success("Extracted confidence.")
    info("Initializing output...")
    output_df = df
    info("Writing output predictions...")
    output_df["prediction1"] = prediction1
    output_df["prediction2"] = prediction2
    output_df["prediction3"] = prediction3
    info("Writing output confidence...")
    output_df["prediction1 prob"] = proba1
    output_df["prediction2 prob"] = proba2
    output_df["prediction3 prob"] = proba3
    success("Finalized output DataFrame.")
    if printout_:
        print(output_df)
    if output is None:
        warn("No output path specified.")
        if not printout_:
            if click.confirm("Would you like to print the output DataFrame?"):
                print(output_df)
    else:
        if not validate_extension(output, ".csv"):
            output = f"{output}.csv"
            warn(f"Added CSV extension (new output: {output}).")
        if os.path.exists(output):
            click.confirm(f"Output path {output} already exists. Do you want to overwrite?", abort=True)
        output_df.to_csv(output, index=False)
        success(f"Saved output to {output}")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Print verbose/debug messages")
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
    info(f"Loading input CSV {file}...")
    validate_path(file)
    validate_extension(file, ".csv")
    df = pandas.read_csv(file)
    validate_dataframe(df, file)
    success(f"Loaded {file}!")
    if not validate_extension(output, ".joblib"):
        output = f"{output}.joblib"
        warn(f"Added .joblib extension (new output: {output}).")
    info("Training model...")
    success("Trained new model successfully!")


@main.group(help="Predict glitch(es) type(s)")
@click.option("--model-path", "-m", help="Path to ML model", default=DEFAULT_MODEL_PATH, metavar="", show_default=True)
@click.option("--printout", "-p", is_flag=True, help="Print output DataFrame")
def predict(model_path, printout):
    global model, printout_
    printout_ = printout
    info(f"Loading model from {model_path}...")
    validate_path(model_path)
    validate_extension(model_path, ".joblib")
    model = joblib.load(model_path)
    success(f"Successfully loaded model.")


@predict.command(help="Load CSV file of glitches")
@click.argument("file")
@click.option("--output", "-o", help="CSV output file path", metavar="")
@click.option("--delete-extras", "-d", is_flag=True, help="Remove extra columns from input to output")
def csv(file, output, delete_extras):
    info(f"Loading {file} into DataFrame...")
    validate_path(file)
    validate_extension(file, ".csv")
    df = pandas.read_csv(file)
    validate_dataframe(df, file)
    df = pandas.read_csv(file)[PARAM_LIST] if delete_extras else df
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
def glitch(peak_freq, snr, amplitude, central_freq, duration, bandwidth, q_value, output):
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
    params = [peak_freq, snr, amplitude, central_freq, duration, bandwidth, q_value]

    info(f"Loading parameters into DataFrame...")
    df = pandas.DataFrame({param: val for (param, val) in zip(PARAM_LIST, params)}, index=[0])
    success(f"Parameters successfully loaded.")
    info(f"Starting model...")
    run_model(df, output)


@main.resultcallback()
def process_result(_, **__):
    colorama.deinit()
    success(f"Finished")


if __name__ == "__main__":
    main()
