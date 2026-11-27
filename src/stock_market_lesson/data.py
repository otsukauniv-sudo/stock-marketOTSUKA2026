"""第13回で使う三銘柄の価格を、指定したデータ源から読み込む。"""

from pathlib import Path

import numpy as np
import pandas as pd


TICKERS = {"Nintendo": "7974.T", "Sony": "6758.T", "SoftBankG": "9984.T"}


def _demo_prices() -> pd.DataFrame:
    """練習用の架空価格。平日を使うが、取引所の休場日は再現しない。"""
    dates = pd.bdate_range("2025-01-06", periods=120, name="Date")
    rng = np.random.default_rng(123)
    market = rng.normal(0.0002, 0.008, size=(len(dates) - 1, 1))
    individual = rng.normal(0, [0.012, 0.010, 0.022], size=(len(dates) - 1, 3))
    cumulative = np.vstack([np.zeros(3), (market + individual).cumsum(axis=0)])
    return pd.DataFrame(100 * np.exp(cumulative), index=dates, columns=list(TICKERS))


def _yahoo_prices() -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError("Yahoo Finance から取得するには yfinance が必要です。") from error

    # yfinance の end は範囲外なので、最終観測日（7月1日）の翌日を指定する。
    raw = yf.download(
        list(TICKERS.values()), start="2025-01-06", end="2026-07-02",
        interval="1d", auto_adjust=True, group_by="column", progress=False,
    )
    if raw is None or raw.empty or "Close" not in raw:
        raise ValueError("Yahoo Finance から調整済み終値を取得できませんでした。")
    return raw["Close"].rename(columns={code: name for name, code in TICKERS.items()})


def load_prices(mode: str = "demo", csv_path: Path | None = None) -> pd.DataFrame:
    """demo は架空データ、csv は教材と同形式のCSV、yahoo は日次調整済み終値。"""
    if mode != "csv" and csv_path is not None:
        raise ValueError("CSVファイルの指定には mode='csv' を使ってください。")
    if mode == "demo":
        prices = _demo_prices()
    elif mode == "csv":
        if csv_path is None:
            raise ValueError("mode='csv' では csv_path を指定してください。")
        prices = pd.read_csv(csv_path, index_col="Date", encoding="utf-8-sig")
    elif mode == "yahoo":
        prices = _yahoo_prices()
    else:
        raise ValueError("mode は 'demo', 'csv', 'yahoo' のいずれかです。")

    if not set(TICKERS).issubset(prices.columns):
        raise ValueError("価格表には Nintendo, Sony, SoftBankG の3列が必要です。")
    prices = prices.loc[:, list(TICKERS)].astype(float)
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index), name="Date").normalize()
    if prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("日付に欠損または重複があります。1日1行の価格表を指定してください。")
    if len(prices) < 3 or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("3日分以上の、欠損のない正の価格が必要です。")
    prices = prices.sort_index()
    if mode == "yahoo" and (prices.index[0].date().isoformat(), prices.index[-1].date().isoformat()) != (
        "2025-01-06", "2026-07-01",
    ):
        raise ValueError("Yahoo Finance の取得結果に観測期間の初日または最終日がありません。")
    prices.attrs["source"] = mode
    return prices
