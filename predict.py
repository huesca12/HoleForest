import click
import colorama
from colorama import Fore
import joblib
import os
import pandas

MODEL_PATH = f"{os.path.abspath(os.path.dirname(__file__))}/model/model.joblib"
PARAM_LIST = ["peakFreq", "snr", "amplitude", "centralFreq", "duration", "bandwidth", "Q-value"]
model = info = warn = success = ...  # All will be modified before use


def _info(msg):
    print(f"[~] {msg}")


def _warn(msg):
    print(f"{Fore.YELLOW}[!] {msg}{Fore.RESET}")


def _success(msg):
    print(f"{Fore.GREEN}[+] {msg}{Fore.RESET}")


def run_model(df, output):
    info("Running model predictions...")
    predictions = model.predict(df)
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
    if output is None:
        warn("No output path specified.")
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
def predict(verbose):
    global model, info, warn, success
    info = _info if verbose else lambda *args, **kwargs: None
    warn = _warn if verbose else lambda *args, **kwargs: None
    success = _success if verbose else lambda *args, **kwargs: None
    colorama.init()
    info(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    success(f"Successfully loaded model.")


@predict.command(help="Load CSV file of glitches")
@click.argument("file")
@click.option("--output", "-o", help="CSV output file path", metavar="")
def csv(file, output):
    info(f"Loading {file} into DataFrame...")
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


@predict.resultcallback()
def process_result(_, **__):
    success(f"Finished")


if __name__ == "__main__":
    predict()
