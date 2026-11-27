"""株価と人工市場の図表を作成する。python -m stock_market_lesson --help を参照。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from .data import load_prices
from .model import (
    AGENT_JP,
    MC_N_SIM_ABLATION,
    MC_N_SIM_MAIN,
    STOCK_JP,
    STOCKS,
    analyze_agent_ablation,
    order_probabilities,
    run_abm,
    run_monte_carlo,
)


COLORS = ("#28629a", "#bb6a2b", "#167367")
ENGLISH_AGENT_LABELS = {
    "nintendo_fan": "Nintendo fan",
    "sony_fan": "Sony fan",
    "ai_investor": "AI enthusiast",
    "trend_follower": "Trend follower",
    "contrarian": "Contrarian",
    "news_reactor": "News reactor",
    "zero_intelligence": "Random trader",
}


def configure_figures() -> bool:
    """日本語のフォントがある場合は図を日本語で出力する。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = ("Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic")
    font = next((name for name in fonts if name in available), None)
    plt.rcParams.update({
        "font.family": font or "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return font is not None


def plot_indices(indices, destination, *, jp, model=False, demo=False):
    figure, axes = plt.subplots(figsize=(9, 3.6), layout="constrained")
    for stock, color, line_style in zip(STOCKS, COLORS, ("-", "--", "-.")):
        label = STOCK_JP[stock] if jp else stock
        axes.plot(indices.index, indices[stock], label=label, color=color, ls=line_style)
    axes.set_ylabel("株価指数（初日＝100）" if jp else "Price index (first day = 100)")
    if model:
        axes.set_xlabel("モデル内の日数（初日＝0）" if jp else "Simulation day (first day = 0)")
    if demo:
        axes.set_title("練習用の架空株価" if jp else "Synthetic prices for practice")
    axes.legend(loc="upper left", ncol=3, frameon=False)
    axes.grid(axis="y", color="0.88")
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_returns(returns, destination, *, jp, demo):
    figure, axes = plt.subplots(figsize=(8, 3.6), layout="constrained")
    boxes = axes.boxplot([returns[stock] for stock in STOCKS], patch_artist=True)
    for box, color in zip(boxes["boxes"], COLORS):
        box.set(facecolor=color, alpha=0.35)
    axes.set_xticks(range(1, 4), [STOCK_JP[stock] if jp else stock for stock in STOCKS])
    axes.set_ylabel("日次収益率（％）" if jp else "Daily return (%)")
    if demo:
        axes.set_title("練習用の架空株価" if jp else "Synthetic prices for practice")
    axes.grid(axis="y", color="0.88")
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_ablation(changes, trials, destination, *, jp):
    figure, axes = plt.subplots(figsize=(8.5, 4.3), layout="constrained")
    names = AGENT_JP if jp else ENGLISH_AGENT_LABELS
    positions = range(len(changes))
    axes.barh(positions, changes, color=["#bb6a2b" if value < 0 else "#28629a" for value in changes])
    axes.set_yticks(positions, [names[agent] for agent in changes.index])
    axes.axvline(0, color="0.23", lw=0.85)
    axes.set_xlabel("基準 − 除去条件の標準偏差（パーセントポイント）" if jp
                    else "Baseline minus exclusion (percentage points)")
    axes.set_title(f"各条件 {trials} 回・三銘柄平均" if jp else f"{trials} runs per condition, mean over three stocks")
    axes.grid(axis="x", color="0.88")
    axes.set_axisbelow(True)
    offset = max(abs(changes)) * 0.035
    for position, value in zip(positions, changes):
        label = f"{value:+.2f}" if abs(value) >= 0.005 else "0.00"
        axes.text(value + (offset if value >= 0 else -offset), position, label,
                  ha="left" if value >= 0 else "right", va="center", fontsize=9)
    axes.margins(x=0.16)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def order_table() -> pd.DataFrame:
    """表3と同じ三類型の注文確率をモデルから読み出す。"""
    rows = []
    for agent in ("trend_follower", "contrarian"):
        for previous_return, condition in ((0.01, "前日収益率 > 0.1%"), (-0.01, "前日収益率 < −0.1%")):
            buy, hold, sell = order_probabilities(agent, "Nintendo", previous_return, 0.0)
            rows.append((AGENT_JP[agent], condition, buy * 100, hold * 100, sell * 100))
    for signal, condition in ((0.01, "ニュース指標 > 0.002"), (-0.01, "ニュース指標 < −0.002"),
                              (0.0, "−0.002 ≤ ニュース指標 ≤ 0.002")):
        buy, hold, sell = order_probabilities("news_reactor", "Nintendo", 0.0, signal)
        rows.append((AGENT_JP["news_reactor"], condition, buy * 100, hold * 100, sell * 100))
    return pd.DataFrame(rows, columns=["類型", "条件", "買い（％）", "見送り（％）", "売り（％）"])


def run_lesson(prices, destination: Path, *, mode: str, full: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    jp = configure_figures()
    demo = mode == "demo"

    # 第13回：初日の値を100とした指数と、日次収益率をそれぞれ読む。
    index_prices = 100 * prices / prices.iloc[0]
    returns = prices.pct_change().dropna() * 100
    plot_indices(index_prices, destination / "01_stock_index.png", jp=jp, demo=demo)
    plot_returns(returns, destination / "02_daily_returns.png", jp=jp, demo=demo)
    returns.std(ddof=1).rename("std_pct").to_csv(destination / "stock_volatility.csv")

    # 第14回：単一の価格経路を見てから、類型を1つずつ除いた条件を比べる。
    single = run_abm()
    plot_indices(single["prices"], destination / "03_market_path.png", jp=jp, model=True)
    trials = MC_N_SIM_ABLATION if full else 5
    print(f"人工市場を各条件 {trials} 回計算します。", flush=True)
    results = analyze_agent_ablation(n_sims=trials)
    results.to_csv(destination / "ablation_by_stock.csv", index=False)
    changes = (results.loc[results["scenario"] != "baseline"]
               .groupby("remove_type")["delta_baseline_minus_scenario"].mean().sort_values())
    changes.rename("difference_pp").to_csv(destination / "ablation_average.csv")
    plot_ablation(changes, trials, destination / "04_agent_comparison.png", jp=jp)

    conditions = ["baseline", "no_trend_follower", "no_contrarian", "no_news_reactor"]
    table4 = results.pivot(index="scenario", columns="stock", values="mean_volatility").loc[conditions, STOCKS]
    table4.index = ["基準条件", "ミーハー型を除く", "ひねくれ型を除く", "ニュース反応型を除く"]
    table4.rename(columns=STOCK_JP).to_csv(destination / "table4_volatility.csv", encoding="utf-8-sig")
    order_table().to_csv(destination / "table3_order_probabilities.csv", index=False, encoding="utf-8-sig", float_format="%.2f")
    if full:
        print(f"補足の基準条件を {MC_N_SIM_MAIN} 回計算します。", flush=True)
        supplemental = run_monte_carlo(MC_N_SIM_MAIN)
        supplemental.vol_df.mean().rename("mean_std_pct").to_csv(destination / "baseline_80_trials.csv")

    source = {"demo": "練習用の架空株価", "csv": "指定された株価CSV", "yahoo": "Yahoo Financeの調整済み終値"}[mode]
    info = (f"株価データ  {source}\n"
            f"観測期間  {prices.index[0]:%Y-%m-%d} ～ {prices.index[-1]:%Y-%m-%d}\n"
            f"観測日数  {len(prices)}\n人工市場の条件ごとの試行数  {trials}\n")
    (destination / "run_info.txt").write_text(info, encoding="utf-8")
    print(info, end="")
    print(f"図と表の保存先  {destination.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="株価データと人工市場の図表を作ります。")
    prices = parser.add_mutually_exclusive_group()
    prices.add_argument("--yahoo", action="store_true", help="Yahoo Financeで授業と同じ銘柄・期間を取得します。")
    prices.add_argument("--prices-csv", type=Path, help="Date,Nintendo,Sony,SoftBankG列の株価CSVを読み込みます。")
    parser.add_argument("--full", action="store_true", help="各条件30回と補足80回を計算します。省略時は各条件5回です。")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="出力先のフォルダ。既定は outputs。")
    args = parser.parse_args()
    mode = "yahoo" if args.yahoo else "csv" if args.prices_csv else "demo"
    try:
        series = load_prices(mode, args.prices_csv)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(2, f"株価データを読み込めませんでした。{error}\n")
    run_lesson(series, args.output, mode=mode, full=args.full)


if __name__ == "__main__":
    main()
