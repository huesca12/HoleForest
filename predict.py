import click
import joblib
import os
import pandas

PARAM_LIST = ["peakFreq", "snr", "amplitude", "centralFreq", "duration", "bandwidth", "Q-value"]
PATH = os.path.abspath(os.path.dirname(__file__))


def run_model(df, output):
    predictions = model.predict(df)
    probas = ...

    output_df = df
    output_df["prediction"] = predictions
    output_df["confidence"] = probas
    if output is not None:
        output = output if output.endswith(".csv") else f"{output}.csv"
        output_df.to_csv(output, index=False)


@click.group()
def predict():
    pass


@predict.command()
@click.argument("file")
@click.option("--output", "-o")
def csv(file, output):
    df = pandas.read_csv(file)[PARAM_LIST]
    run_model(df, output)


@predict.command()
@click.option("--peak-freq", "-p", type=float)
@click.option("--snr", "-s", type=float)
@click.option("--amplitude", "-a", type=float)
@click.option("--central-freq", "-c", type=float)
@click.option("--duration", "-d", type=float)
@click.option("--bandwidth", "-b", type=float)
@click.option("--q-value", "-q", type=float)
@click.option("--output", "-o")
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

    df = pandas.DataFrame({param: val for (param, val) in zip(PARAM_LIST, params)}, index=[0])
    run_model(df, output)


if __name__ == "__main__":
    model = joblib.load(f"{PATH}/model/model.joblib")
    predict()
