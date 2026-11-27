"""株価の値動きを読み解く授業用の人工市場モデル。

投資家は類型ごとのルールに従って注文を選ぶ。類型ごとに集計した
注文の偏り、市場全体の動き、ニュースなどから翌日の価格を決める。
価格は初日を100とする。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


STOCKS = ["Nintendo", "Sony", "SoftBankG"]
STOCK_JP = {
    "Nintendo": "任天堂",
    "Sony": "ソニーグループ",
    "SoftBankG": "ソフトバンクグループ",
}
N_AGENTS = 250
N_DAYS = 160
BASE_SEED = 42
MC_N_SIM_MAIN = 80
MC_N_SIM_ABLATION = 30

AGENT_TYPES = [
    "nintendo_fan",
    "sony_fan",
    "ai_investor",
    "trend_follower",
    "contrarian",
    "news_reactor",
    "zero_intelligence",
]

AGENT_JP = {
    "nintendo_fan": "任天堂ファン型",
    "sony_fan": "ソニーファン型",
    "ai_investor": "AI期待型",
    "trend_follower": "ミーハー型",
    "contrarian": "ひねくれ型",
    "news_reactor": "ニュース反応型",
    "zero_intelligence": "適当型",
}

# 適当型を除いた六類型の基礎比率
BASE_TYPE_PROBS_NON_ZI = {
    "nintendo_fan": 0.17,
    "sony_fan": 0.17,
    "ai_investor": 0.20,
    "trend_follower": 0.22,
    "contrarian": 0.14,
    "news_reactor": 0.10,
}

# 銘柄ごとの基礎ドリフト
MU = {
    "Nintendo": 0.00015,
    "Sony": 0.00012,
    "SoftBankG": 0.00020,
}

# 市場ムード C_t への感応度
BETA_MARKET = {
    "Nintendo": 0.35,
    "Sony": 0.45,
    "SoftBankG": 0.60,
}

# 各類型の純注文が価格に反映される強さ
ALPHA_TYPE = {
    "Nintendo": {
        "nintendo_fan": 0.016,
        "sony_fan": 0.003,
        "ai_investor": 0.002,
        "trend_follower": 0.026,
        "contrarian": 0.018,
        "news_reactor": 0.030,
        "zero_intelligence": 0.008,
    },
    "Sony": {
        "nintendo_fan": 0.003,
        "sony_fan": 0.014,
        "ai_investor": 0.004,
        "trend_follower": 0.023,
        "contrarian": 0.017,
        "news_reactor": 0.028,
        "zero_intelligence": 0.008,
    },
    "SoftBankG": {
        "nintendo_fan": 0.002,
        "sony_fan": 0.003,
        "ai_investor": 0.036,
        "trend_follower": 0.036,
        "contrarian": 0.020,
        "news_reactor": 0.040,
        "zero_intelligence": 0.012,
    },
}

# ニュース持続効果 H_{i,t+1} = phi_i H_{i,t} + g_i(e_t)
PHI_NEWS = {
    "Nintendo": 0.45,
    "Sony": 0.45,
    "SoftBankG": 0.50,
}

# 銘柄ごとの乱数による変動の基準
SIGMA0 = {
    "Nintendo": 0.007,
    "Sony": 0.008,
    "SoftBankG": 0.014,
}

# 注文が偏るほどボラティリティが上がる
LAMBDA_VOL = {
    "Nintendo": 0.80,
    "Sony": 0.90,
    "SoftBankG": 1.30,
}

# 市場ムード過程 C_t = rho C_{t-1} + eta_t
MARKET_MOOD_RHO = 0.35
MARKET_MOOD_SD = 0.006

# ニュースイベント
NEWS_EVENTS = [
    {
        "name": "新型ゲーム機が話題",
        "weight": 0.10,
        "impact": {"Nintendo": 0.018, "Sony": 0.000, "SoftBankG": 0.000},
    },
    {
        "name": "人気ゲームが大ヒット",
        "weight": 0.08,
        "impact": {"Nintendo": 0.022, "Sony": 0.004, "SoftBankG": 0.000},
    },
    {
        "name": "PlayStation関連ニュース",
        "weight": 0.10,
        "impact": {"Nintendo": 0.000, "Sony": 0.018, "SoftBankG": 0.000},
    },
    {
        "name": "映画・音楽事業が好調",
        "weight": 0.08,
        "impact": {"Nintendo": 0.000, "Sony": 0.014, "SoftBankG": 0.000},
    },
    {
        "name": "AI投資ブーム",
        "weight": 0.10,
        "impact": {"Nintendo": 0.000, "Sony": 0.004, "SoftBankG": 0.030},
    },
    {
        "name": "テック株に買いが集まる",
        "weight": 0.10,
        "impact": {"Nintendo": 0.003, "Sony": 0.006, "SoftBankG": 0.024},
    },
    {
        "name": "円高進行",
        "weight": 0.10,
        "impact": {"Nintendo": -0.012, "Sony": -0.012, "SoftBankG": -0.010},
    },
    {
        "name": "世界的に株安",
        "weight": 0.11,
        "impact": {"Nintendo": -0.020, "Sony": -0.022, "SoftBankG": -0.028},
    },
    {
        "name": "特にニュースなし",
        "weight": 0.23,
        "impact": {"Nintendo": 0.000, "Sony": 0.000, "SoftBankG": 0.000},
    },
]

NEWS_WEIGHTS = np.array([e["weight"] for e in NEWS_EVENTS], dtype=float)
NEWS_WEIGHTS = NEWS_WEIGHTS / NEWS_WEIGHTS.sum()


def make_type_probabilities(
    zero_share: float = 0.10, remove_type: str | None = None
) -> dict[str, float]:
    """各類型に割り当てる確率を求め、除去する類型の分を残りに再配分する。"""
    if not 0 <= zero_share <= 1:
        raise ValueError("適当型の割合は0以上1以下で指定してください。")

    total_non_random = sum(BASE_TYPE_PROBS_NON_ZI.values())
    probabilities = {
        name: (1 - zero_share) * share / total_non_random
        for name, share in BASE_TYPE_PROBS_NON_ZI.items()
    }
    probabilities["zero_intelligence"] = zero_share

    if remove_type is None:
        return probabilities
    if remove_type not in probabilities:
        raise ValueError(f"投資家の類型が見つかりません: {remove_type}")

    probabilities[remove_type] = 0.0
    remaining = sum(probabilities.values())
    if remaining <= 0:
        raise ValueError("割当確率が正の投資家類型を少なくとも一つ残してください。")
    return {name: share / remaining for name, share in probabilities.items()}


def order_probabilities(
    agent_type: str, stock: str, last_return: float, signal: float
) -> tuple[float, float, float]:
    """
    投資家の類型とその日の情報から、買い・見送り・売りの確率を返す。

    signal は当日と過去のニュースの影響をまとめた指標。
    """
    pos = signal > 0.002
    neg = signal < -0.002

    # 任天堂ファン型
    if agent_type == "nintendo_fan":
        if stock == "Nintendo":
            if pos:
                return (0.72, 0.20, 0.08)
            if neg:
                return (0.35, 0.35, 0.30)
            return (0.58, 0.32, 0.10)
        return (0.25, 0.55, 0.20)

    # ソニーファン型
    if agent_type == "sony_fan":
        if stock == "Sony":
            if pos:
                return (0.68, 0.24, 0.08)
            if neg:
                return (0.32, 0.38, 0.30)
            return (0.55, 0.35, 0.10)
        return (0.25, 0.55, 0.20)

    # AI期待型
    if agent_type == "ai_investor":
        if stock == "SoftBankG":
            if pos:
                return (0.78, 0.16, 0.06)
            if neg:
                return (0.25, 0.35, 0.40)
            return (0.62, 0.25, 0.13)
        return (0.25, 0.55, 0.20)

    # ミーハー型
    if agent_type == "trend_follower":
        if last_return > 0.001:
            return (0.64, 0.26, 0.10)
        if last_return < -0.001:
            return (0.12, 0.33, 0.55)
        return (1 / 3, 1 / 3, 1 / 3)

    # ひねくれ型
    if agent_type == "contrarian":
        if last_return < -0.001:
            return (0.64, 0.26, 0.10)
        if last_return > 0.001:
            return (0.12, 0.33, 0.55)
        return (1 / 3, 1 / 3, 1 / 3)

    # ニュース反応型
    if agent_type == "news_reactor":
        if pos:
            return (0.70, 0.22, 0.08)
        if neg:
            return (0.08, 0.22, 0.70)
        return (0.25, 0.50, 0.25)

    # 適当型
    if agent_type == "zero_intelligence":
        return (1 / 3, 1 / 3, 1 / 3)

    return (1 / 3, 1 / 3, 1 / 3)


def _draw_orders(
    rng: np.random.Generator,
    type_indices: np.ndarray,
    stock: str,
    last_return: float,
    news_state: float,
) -> np.ndarray:
    """買いを1、見送りを0、売りを−1として、全員分の注文を選ぶ。"""
    probabilities = [
        order_probabilities(name, stock, last_return, news_state)
        for name in AGENT_TYPES
    ]
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative /= cumulative[:, -1:]
    thresholds = cumulative[type_indices]

    # 教材の元の実装と同じ順序で、投資家一人につき一つの乱数を引く。
    draws = rng.random(size=len(type_indices))
    choices = (draws >= thresholds[:, 0]).astype(int)
    choices += draws >= thresholds[:, 1]
    return 1 - choices


def run_abm(
    n_agents: int = N_AGENTS,
    n_days: int = N_DAYS,
    zero_share: float = 0.10,
    remove_type: str | None = None,
    seed: int = BASE_SEED,
) -> dict:
    """人工市場を一回動かし、株価、日次収益率（％）、注文などの記録を返す。

    n_days は価格の初期値を含む日数。remove_type を指定すると、その類型
    に割り当てる確率をゼロにし、投資家の総数を保って比較する。
    """
    if n_agents < 1 or n_days < 1:
        raise ValueError("投資家の人数と日数は1以上で指定してください。")

    rng = np.random.default_rng(seed)
    probabilities = make_type_probabilities(zero_share, remove_type)
    weights = np.array([probabilities[name] for name in AGENT_TYPES])
    weights /= weights.sum()
    type_indices = rng.choice(len(AGENT_TYPES), size=n_agents, p=weights)
    agent_types = np.asarray(AGENT_TYPES)[type_indices]
    counts = np.bincount(type_indices, minlength=len(AGENT_TYPES))

    prices = {stock: [100.0] for stock in STOCKS}
    news_state = dict.fromkeys(STOCKS, 0.0)
    market_mood = 0.0
    detail_records = []

    for day in range(n_days - 1):
        market_mood = MARKET_MOOD_RHO * market_mood + rng.normal(0, MARKET_MOOD_SD)
        event = NEWS_EVENTS[rng.choice(len(NEWS_EVENTS), p=NEWS_WEIGHTS)]
        daily = {"t": day, "news": event["name"], "market_mood": market_mood}

        for stock in STOCKS:
            series = prices[stock]
            last_return = (series[-1] - series[-2]) / series[-2] if day else 0.0
            news_state[stock] = PHI_NEWS[stock] * news_state[stock] + event["impact"][stock]
            orders = _draw_orders(rng, type_indices, stock, last_return, news_state[stock])

            order_totals = np.bincount(
                type_indices, weights=orders, minlength=len(AGENT_TYPES)
            )
            # 類型ごとに一人当たりの純注文（買いの数−売りの数）を求める。
            order_balance = np.divide(
                order_totals, counts, out=np.zeros(len(AGENT_TYPES)), where=counts > 0
            )
            contributions = [
                ALPHA_TYPE[stock][name] * balance
                for name, balance in zip(AGENT_TYPES, order_balance)
            ]
            total_balance = orders.sum() / n_agents
            noise_scale = SIGMA0[stock] * (1 + LAMBDA_VOL[stock] * abs(total_balance))
            type_contribution = sum(contributions)
            simulated_return = (
                MU[stock]
                + BETA_MARKET[stock] * market_mood
                + type_contribution
                + news_state[stock]
                + noise_scale * rng.normal()
            )
            new_price = max(1.0, series[-1] * (1 + simulated_return))
            series.append(new_price)

            daily.update({
                f"{stock}_return": simulated_return,
                f"{stock}_price": new_price,
                f"{stock}_M_total": total_balance,
                f"{stock}_buy": int(np.sum(orders == 1)),
                f"{stock}_hold": int(np.sum(orders == 0)),
                f"{stock}_sell": int(np.sum(orders == -1)),
                f"{stock}_news_state": news_state[stock],
                f"{stock}_sigma_t": noise_scale,
                f"{stock}_type_contribution": type_contribution,
            })
            for name, balance, contribution in zip(AGENT_TYPES, order_balance, contributions):
                daily[f"{stock}_M_{name}"] = balance
                daily[f"{stock}_contrib_{name}"] = contribution

        detail_records.append(daily)

    sim_prices = pd.DataFrame(prices)
    return {
        "prices": sim_prices,
        "returns": sim_prices.pct_change().dropna() * 100,
        "detail": pd.DataFrame(detail_records),
        "agent_types": agent_types,
        "type_counts": pd.Series(counts, index=AGENT_TYPES, name="count"),
        "zero_share": zero_share,
        "remove_type": remove_type,
        "seed": seed,
    }


@dataclass
class MCResult:
    """各試行の価格・収益率と、試行別の集計をまとめた結果。"""

    scenario: str
    price_cube: np.ndarray
    return_cube: np.ndarray
    vol_df: pd.DataFrame
    final_price_df: pd.DataFrame
    detail_list: list[pd.DataFrame]


def run_monte_carlo(
    n_sims: int,
    scenario: str = "baseline",
    zero_share: float = 0.10,
    remove_type: str | None = None,
    base_seed: int = BASE_SEED,
    collect_details: bool = False,
    *,
    n_agents: int = N_AGENTS,
    n_days: int = N_DAYS,
) -> MCResult:
    """同じ条件を乱数を変えて繰り返し、試行別の標準偏差と最終価格を返す。"""
    if n_sims < 1:
        raise ValueError("試行回数は1以上で指定してください。")

    all_prices = []
    all_returns = []
    volatilities = []
    final_prices = []
    details = []

    for trial in range(n_sims):
        result = run_abm(
            n_agents=n_agents,
            n_days=n_days,
            zero_share=zero_share,
            remove_type=remove_type,
            seed=base_seed + 1000 * trial + 7,
        )
        prices = result["prices"]
        returns = result["returns"]
        all_prices.append(prices.to_numpy())
        all_returns.append(returns.to_numpy())
        volatilities.append(returns.std())
        final_prices.append(prices.iloc[-1])
        if collect_details:
            details.append(result["detail"])

    return MCResult(
        scenario=scenario,
        price_cube=np.stack(all_prices),
        return_cube=np.stack(all_returns),
        vol_df=pd.DataFrame(volatilities, columns=STOCKS),
        final_price_df=pd.DataFrame(final_prices, columns=STOCKS),
        detail_list=details,
    )


def analyze_agent_ablation(
    n_sims: int = MC_N_SIM_ABLATION,
    *,
    n_agents: int = N_AGENTS,
    n_days: int = N_DAYS,
    base_seed: int = BASE_SEED + 50_000,
) -> pd.DataFrame:
    """基準条件と、一つの投資家類型を除いた各条件の標準偏差を比較する。

    差は「基準条件の平均 − 比較条件の平均」の順に計算するため、
    正ならその類型を除いた条件で日々の値動きのばらつきが小さい。
    """
    scenarios = {"baseline": None}
    scenarios.update({f"no_{name}": name for name in AGENT_TYPES})

    baseline = run_monte_carlo(
        n_sims, n_agents=n_agents, n_days=n_days, base_seed=base_seed
    ).vol_df.mean()
    rows = []
    for scenario, remove_type in scenarios.items():
        if remove_type is None:
            average = baseline
        else:
            trials = run_monte_carlo(
                n_sims,
                scenario=scenario,
                remove_type=remove_type,
                base_seed=base_seed,
                n_agents=n_agents,
                n_days=n_days,
            )
            average = trials.vol_df.mean()
        for stock in STOCKS:
            rows.append({
                "scenario": scenario,
                "remove_type": str(remove_type),
                "stock": stock,
                "mean_volatility": average[stock],
                "delta_baseline_minus_scenario": baseline[stock] - average[stock],
            })
    return pd.DataFrame(rows)
