# E-Commerce Product Recommendation System

A beginner-friendly Streamlit dashboard for exploring synthetic online-store
purchases, grouping customers with K-Means, and suggesting products that a
customer has not bought yet.

The bundled `data/purchases.csv` is **synthetic demo data** generated locally
with a fixed random seed. It does not contain real customer information. The
application has no external API, database, or login requirement.

## Project objective

This project demonstrates a simple end-to-end recommendation workflow:
generate or upload purchase records, clean and summarize them, segment customers,
and explain product suggestions using purchases by similar customers and overall
popularity.

## Project files

- `app.py` — Streamlit dashboard and navigation.
- `generate_data.py` — reproducible product catalog and purchase data generator.
- `src/data_processing.py` — CSV schema validation, cleaning, and customer
  feature engineering.
- `src/clustering.py` — feature scaling and K-Means clustering.
- `src/recommendations.py` — same-cluster recommendations and global fallback.
- `src/visualizations.py` — Matplotlib sales and clustering charts.
- `data/purchases.csv` — generated synthetic demo data (created on first run if
  missing).

## Dataset

The purchase CSV has one row per transaction and these required columns:

| Column | Description |
| --- | --- |
| `TransactionID` | Unique transaction identifier; duplicate IDs are removed. |
| `CustomerID` | Customer identifier. |
| `ProductID` | Stable identifier from the generated product catalog. |
| `ProductName` | Product name. |
| `Category` | Electronics, Fashion, Home, Books, or Beauty in demo data. |
| `Price` | Positive unit price in Indian rupees. |
| `Quantity` | Positive whole-number units purchased. |
| `PurchaseDate` | Purchase date. |

The generator creates 160 customers and 40 catalog products. Customers have
different category preferences, and the random seed makes generated data
reproducible. The app also accepts an uploaded CSV with the same required
columns. Rows with missing identifiers, invalid dates, non-positive prices,
non-positive or fractional quantities, or repeated transaction IDs are removed.
Missing product names and categories are labeled rather than silently treated
as valid catalog data.

## Run on Windows or in VS Code

1. Open the project folder in VS Code (or another terminal).
2. In the integrated terminal, install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

The first run creates `data/purchases.csv` if the file is not already present.
To generate it separately, run `python generate_data.py`.

## How it works

### Customer features and scaling

For each customer, the app calculates total spending, transaction count, total
quantity, average transaction value, and the proportion of spending in each
category. `StandardScaler` centers and scales these numeric features so that
currency values do not overwhelm smaller-scale counts or proportions.

### Segmentation

K-Means assigns customers to a selectable number of clusters (default 3) using
`random_state=42` and `n_init=10`. Cluster IDs are arbitrary labels: inspect
cluster counts and feature averages instead of assuming a cluster is
high-value or low-value. The spending-versus-transaction scatter plot shows
only two of the features used by the model.

### Recommendations

For the selected customer, products already purchased are excluded. The app
first ranks products bought by other customers in the same cluster by distinct
customer count and then total units purchased. If that group does not provide
enough unseen products, globally popular unseen products fill the remaining
places. Each item states its recommendation source and the purchase counts
behind it. Popularity counts are not prediction probabilities, and the project
does not claim measured recommendation accuracy.

## Dashboard guide

- **Overview** — KPI cards for customers, products, transactions, and revenue;
  category revenue and top-products charts; category/date filters for sales
  exploration.
- **Customer Segmentation** — cluster counts, customer spending/activity
  scatter plot, cluster feature averages, and a cluster CSV download.
- **Recommendations** — choose a customer, review spending and purchase history,
  select 3–10 suggestions, and download the recommendation CSV.
- **Dataset** — upload a purchase CSV, see cleaning counts and active records,
  and download the cleaned active dataset.

The model always uses the full active dataset. Category and date filters affect
only the sales charts.

## Troubleshooting

- **`python` is not recognized:** install Python and enable its PATH option, or
  use the Python launcher `py -m pip install -r requirements.txt` and
  `py -m streamlit run app.py`.
- **Package installation errors:** update pip with
  `python -m pip install --upgrade pip`, then retry installing
  `requirements.txt`. Use a Python version supported by the listed packages.
- **`ModuleNotFoundError`:** run the install command from the project folder
  using the same Python interpreter used to launch Streamlit.
- **CSV upload validation message:** ensure all eight required columns are
  present and check that IDs are populated, dates can be parsed, prices are
  positive, and quantities are positive whole numbers.
- **No valid data or too few customers:** check the Dataset page cleaning
  counts. At least two customers and enough distinct customer features are
  required for segmentation.
- **Port is in use:** start on another port with
  `python -m streamlit run app.py --server.port 8502`.

## Limitations and future improvements

- Demo purchases are synthetic and do not reflect real customer behavior.
- Recommendation accuracy has not been evaluated; sparse clusters use global
  popularity as a transparent fallback.
- K-Means requires a chosen cluster count and can change results when the
  active dataset changes.
- Future work could evaluate recommendations against held-out purchases,
  compare ranking strategies, tune clustering, and add richer product or
  customer context using appropriately governed data.
