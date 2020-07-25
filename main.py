import click
import colorama
from colorama import Fore
import joblib
import numpy
import os
import pandas
import sklearn

MODEL_PATH = f"{os.path.abspath(os.path.dirname(__file__))}/model/model.joblib"
PARAM_LIST = ["peakFreq", "snr", "amplitude", "centralFreq", "duration", "bandwidth", "Q-value"]
model_ = printout_ = info = warn = success = ...  # All will be modified before use


def _info(msg, *args, **kwargs):
    print(f"[~] {msg}", *args, **kwargs)


def _warn(msg, *args, **kwargs):
    print(f"{Fore.YELLOW}[!] {msg}{Fore.RESET}", *args, **kwargs)


def _success(msg, *args, **kwargs):
    print(f"{Fore.GREEN}[+] {msg}{Fore.RESET}", *args, **kwargs)


def run_model(df, output):
    info("Running model predictions...")
    predictions = model_.predict(df)
    success("Extracted predictions.")
    info("Running model confidence...")
    probas = ...
    success("Extracted confidence.")
    info("Initializing output...")
    output_df = df
    info("Writing output predictions...")
    output_df["prediction"] = predictions
    info("Writing output confidence...")
    output_df["confidence"] = probas
    success("Finalized output DataFrame.")
    if printout_:
        print(output_df)
    if output is None:
        warn("No output path specified.")
        if not printout_:
            if click.confirm("Would you like to print the output DataFrame?"):
                print(output_df)
    else:
        if not output.endswith(".csv"):
            warn("Output file does not have the CSV extension.")
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
@click.argument("file", help="CSV of glitches for training")
@click.argument("output", help="Output .joblib trained model file")
def train(file, output):
    info(f"Loading input CSV {file}...")
    if not os.path.exists(file):
        warn(f"Path to input file ({file}) not found.")
    if not file.endswith(".csv"):
        warn("Input file does not have the CSV extension.")
    success(f"Loaded {file}!")
    if not output.endswith(".joblib"):
        warn(f"Output path does not have the .joblib extension.")
        output = f"{output}.joblib"
        warn(f"Added .joblib extension (new output: {output}).")
    info("Training model...")
    ...
    success("Trained new model successfully!")


@main.group(help="Predict glitch(es) type(s)")
@click.option("--model", "-m", help="Path to ML model", default="model/model.joblib", metavar="", show_default=True)
@click.option("--printout", "-p", is_flag=True, help="Print output DataFrame")
def predict(model, printout):
    global printout_, model_, MODEL_PATH
    printout_ = printout
    MODEL_PATH = model
    info(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        warn(f"Path to model ({MODEL_PATH}) not found.")
    if not MODEL_PATH.endswith(".joblib"):
        warn(f"Model file does not have the .joblib extension.")
    model_ = joblib.load(MODEL_PATH)
    success(f"Successfully loaded model.")


@predict.command(help="Load CSV file of glitches")
@click.argument("file", help="CSV of glitches")
@click.option("--output", "-o", help="CSV output file path", metavar="")
def csv(file, output):
    info(f"Loading {file} into DataFrame...")
    if not os.path.exists(file):
        warn(f"Path to input file ({file}) not found.")
    if not file.endswith(".csv"):
        warn("Input file does not have the CSV extension.")
    df = pandas.read_csv(file)[PARAM_LIST]
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
    success(f"Finished")


if __name__ == "__main__":
    main()
