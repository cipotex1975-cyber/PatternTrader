"""Descargar datos de Yahoo Finance y convertir al formato del backtest."""
from pathlib import Path

import yfinance as yf
import pandas as pd


def download_yfinance(
    symbol: str = "USDJPY=X",
    period: str = "730d",
    interval: str = "1h",
    output_dir: str = "app/datos_test",
) -> str:
    print(f"Descargando {symbol} ({interval}, {period})...")
    data = yf.download(symbol, period=period, interval=interval)

    if data.empty:
        print("Error: no se obtuvieron datos")
        return ""

    data.columns = data.columns.get_level_values(0)

    df = pd.DataFrame({
        "DateTime": data.index.strftime("%Y.%m.%d"),
        "time": data.index.strftime("%H:%M:%S"),
        "Open": data["Open"].round(5),
        "High": data["High"].round(5),
        "Low": data["Low"].round(5),
        "Close": data["Close"].round(5),
        "Tickvol": data.get("Volume", 0).astype(int),
        "Volume": 0,
        "Spread": 10,
    })

    out_path = Path(__file__).parent / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol.replace('=', '').replace('.', '_')}_{interval}_{period}.txt"
    filepath = out_path / filename

    df.to_csv(filepath, sep="\t", index=False)
    print(f"Guardado: {filepath}")
    print(f"Filas: {len(df)}, Rango: {df['DateTime'].iloc[0]} a {df['DateTime'].iloc[-1]}")

    return str(filepath)


if __name__ == "__main__":
    download_yfinance()
