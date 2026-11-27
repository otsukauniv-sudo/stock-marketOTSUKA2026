# 株価データと人工市場の授業教材

青森中央学院大学「データサイエンス・AI基礎」第13・14回の教材です。第13回は株価の変化と日々の値動きについて。
第14回は、七種類の投資家を置いた人工市場から一種類ずつ外し、値動きのばらつきがどう変わるかについてです。

## set up 

Python 3.10以上を用意します。GitHubの **Code → Download ZIP** から取得して解凍し、`README.md` があるフォルダで端末を開いて次を実行します。Windowsで `python` が見つからない場合は `py` に置き換えてください。

```sh
python -m pip install .
python -m stock_market_lesson
```

初回のインストールにはインターネット接続が必要です。実行時に指定を省くと、練習用の架空株価を使い、人工市場を各条件5回計算します。図とCSVは `outputs` に保存されます。まず `01_stock_index.png` と `04_agent_comparison.png` を開いてください。

## 報告書と同じ設定で計算のやり方

### Yahoo Financeから株価を取得する場合

```sh
python -m pip install ".[yahoo]"
python -m stock_market_lesson --yahoo --full --output outputs/yahoo
```

任天堂（7974.T）、ソニーグループ（6758.T）、ソフトバンクグループ（9984.T）の2025年1月6日から2026年7月1日までの調整済み終値を取得します。配当や株式分割の反映時期などにより、再取得時の価格は教材に保存した値と異なる場合があります。取得できなかった場合はエラーで停止します。

### 授業時の株価を保存したCSVを使う場合

```sh
python -m stock_market_lesson --prices-csv real_prices.csv --full --output outputs/csv
```

CSVの先頭行は `Date,Nintendo,Sony,SoftBankG` です。1取引日につき1行で、日付は `YYYY-MM-DD`、各銘柄の価格は調整済み終値です。株価データはYahoo Financeから取得するか、手元のCSVを指定してください。

実株価を取得せず、人工市場だけ報告書と同じ回数・乱数設定で計算するには、`python -m stock_market_lesson --full` を使えます。このとき図1・2の株価は練習用です。

## どの出力を見るか

- `01_stock_index.png`（図1）　最初の取引日の価格を100とした株価指数。
- `02_daily_returns.png`（図2）　前の取引日からの価格変化率の箱ひげ図。標準偏差は `stock_volatility.csv`。
- `03_market_path.png`（図3）　250人・160日で計算した人工市場の一例。乱数の初期値は42。
- `04_agent_comparison.png`（図4）　「基準条件の標準偏差 − ある類型を除いた条件の標準偏差」を三銘柄で平均した値。プラスなら、その類型を除いた条件の方がばらつきが小さい。
- `table3_order_probabilities.csv`、`table4_volatility.csv`　売買ルールと銘柄別の標準偏差（表3・4）。全条件の値は `ablation_by_stock.csv`。
- `baseline_80_trials.csv`　基準条件の補足集計。`--full` を付けたときに作成。
- `run_info.txt`　株価の出典、観測期間、試行数。

`--full` の図4では、ニュース反応型は **＋1.65**、ミーハー型は **＋0.51**、ひねくれ型は **−0.48** パーセントポイントです。これはモデル内で、除いた類型の分を残りの類型に配分し直して比べた結果です。

## コードを読む順番

1. [`data.py`](src/stock_market_lesson/data.py) で、株価をどこから読むかを確認する。
2. [`model.py`](src/stock_market_lesson/model.py) の `order_probabilities` で、どういうときに買い・見送り・売りが出やすいかを読む。`run_abm` は人工市場を一回動かし、`analyze_agent_ablation` は投資家を一類型ずつ除いた比較を行う。
3. [`__main__.py`](src/stock_market_lesson/__main__.py) の `run_lesson` で、図と表を作る順番を確認する。
