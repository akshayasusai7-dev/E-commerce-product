"""Interactive e-commerce customer segmentation and recommendations dashboard."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from generate_data import ensure_dataset
from src.clustering import cluster_customers, feature_columns
from src.data_processing import build_customer_features, clean_transactions, validate_columns
from src.recommendations import recommend_products
from src.visualizations import (
    category_revenue_chart,
    cluster_count_chart,
    customer_scatter_chart,
    top_products_chart,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "purchases.csv"
MONEY_FORMAT = "₹{:,.0f}"

st.set_page_config(
    page_title="E-Commerce Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #0b0a10; color: #f4f0ff; }
    [data-testid="stSidebar"] { background: #111018; border-right: 1px solid #30293a; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #17131f, #211831);
        border: 1px solid #3a2b50; border-radius: 14px; padding: 18px 20px;
    }
    [data-testid="stMetricLabel"] { color: #b8abc9; }
    [data-testid="stMetricValue"] { color: #d7c4ff; }
    .hero {
        background: linear-gradient(115deg, #211432, #17131f 68%);
        border: 1px solid #493365; border-radius: 18px; padding: 26px 30px;
        margin: 8px 0 22px;
    }
    .hero h1 { margin: 0 0 8px; color: #f4f0ff; }
    .hero p { margin: 0; color: #c7b9dc; }
    .stButton button, .stDownloadButton button {
        border: 1px solid #7652a6; border-radius: 9px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_and_clean(source_bytes: bytes) -> tuple[pd.DataFrame, dict[str, int]]:
    """Cache parsed input by its contents, so uploading another file invalidates it."""
    raw = pd.read_csv(BytesIO(source_bytes))
    validate_columns(raw)
    return clean_transactions(raw)


@st.cache_data(show_spinner=False)
def prepare_customers(
    transactions: pd.DataFrame, cluster_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_customer_features(transactions)
    return cluster_customers(features, cluster_count)


def money(value: float) -> str:
    return MONEY_FORMAT.format(value)


def render_overview(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    summary: pd.DataFrame,
    original_rows: int,
    removed_rows: int,
) -> None:
    st.markdown(
        '<div class="hero"><h1>E-Commerce Product Recommendation System</h1>'
        "<p>Explore customer segments and discover transparent, popularity-based product suggestions. "
        "<strong>Synthetic demo data</strong> is used unless you upload your own CSV.</p></div>",
        unsafe_allow_html=True,
    )
    revenue = (transactions["Price"] * transactions["Quantity"]).sum()
    products = transactions["ProductID"].nunique()
    metrics = st.columns(4)
    metrics[0].metric("Customers", f"{transactions['CustomerID'].nunique():,}")
    metrics[1].metric("Products", f"{products:,}")
    metrics[2].metric("Transactions", f"{transactions['TransactionID'].nunique():,}")
    metrics[3].metric("Revenue", money(revenue))

    st.caption(
        "Recommendation and segmentation use the full active dataset. Category and date filters "
        "below affect sales charts only."
    )
    left, right = st.columns(2)
    categories = sorted(transactions["Category"].dropna().unique().tolist())
    selected_categories = left.multiselect(
        "Sales chart categories", categories, default=categories
    )
    dates = transactions["PurchaseDate"]
    date_range = right.date_input(
        "Sales chart date range",
        value=(dates.min().date(), dates.max().date()),
        min_value=dates.min().date(),
        max_value=dates.max().date(),
    )
    chart_data = transactions.loc[transactions["Category"].isin(selected_categories)]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
        chart_data = chart_data.loc[
            chart_data["PurchaseDate"].dt.date.between(start, end)
        ]

    if chart_data.empty:
        st.info("No sales match these chart filters. Choose another category or date range.")
    else:
        left, right = st.columns(2)
        left.pyplot(category_revenue_chart(chart_data), use_container_width=True)
        right.pyplot(top_products_chart(chart_data), use_container_width=True)
    st.caption(
        f"Active data: {original_rows:,} input rows; {removed_rows:,} invalid or duplicate "
        "rows removed during cleaning."
    )


def render_segmentation(customers: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.header("Customer Segmentation")
    st.write(
        "K-Means groups customers after StandardScaler puts spending, activity, and category "
        "spending shares on comparable scales. Cluster numbers are arbitrary group labels, "
        "not fixed value rankings."
    )
    left, right = st.columns(2)
    left.pyplot(cluster_count_chart(customers), use_container_width=True)
    right.pyplot(customer_scatter_chart(customers), use_container_width=True)
    st.caption(
        "The scatter plot displays only two of the model's features: total spending and "
        "transaction count. Clustering uses all listed numeric features."
    )
    st.subheader("Cluster feature averages")
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "TotalSpending": st.column_config.NumberColumn(
                "Average spending", format="₹%.0f"
            ),
            "AverageTransactionValue": st.column_config.NumberColumn(
                "Average transaction value", format="₹%.0f"
            ),
        },
    )
    st.download_button(
        "Download customer clusters",
        data=customers.to_csv(index=False).encode("utf-8"),
        file_name="customer_clusters.csv",
        mime="text/csv",
    )


def render_recommendations(
    transactions: pd.DataFrame, customers: pd.DataFrame
) -> None:
    st.header("Product Recommendations")
    customer_ids = sorted(customers["CustomerID"].astype(str).unique())
    selected = st.selectbox("Choose a CustomerID", customer_ids)
    count = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    customer_transactions = transactions.loc[
        transactions["CustomerID"].astype(str).eq(selected)
    ].copy()
    customer_features = customers.loc[
        customers["CustomerID"].astype(str).eq(selected)
    ].iloc[0]
    category_spend = (
        customer_transactions.assign(
            Spend=customer_transactions["Price"] * customer_transactions["Quantity"]
        )
        .groupby("Category")["Spend"]
        .sum()
    )
    favourite = category_spend.idxmax() if not category_spend.empty else "Not available"

    metrics = st.columns(3)
    metrics[0].metric("Total spending", money(float(customer_features["TotalSpending"])))
    metrics[1].metric("Favourite category", favourite)
    metrics[2].metric("Cluster", f"Cluster {int(customer_features['Cluster'])}")
    st.subheader("Purchase history")
    st.dataframe(
        customer_transactions.sort_values("PurchaseDate", ascending=False)[
            ["PurchaseDate", "ProductName", "Category", "Quantity", "Price"]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="₹%.0f"),
        },
    )

    suggestions = recommend_products(
        transactions, customers, selected, recommendation_count=count
    )
    st.subheader("Suggested products")
    st.caption(
        "Suggestions rank by distinct customers, then units purchased. Popularity counts are "
        "not prediction probabilities, and recommendation accuracy has not been evaluated."
    )
    if suggestions.empty:
        st.info("This customer has purchased every available product; no unseen products remain.")
        return
    st.dataframe(
        suggestions[
            [
                "ProductName",
                "Category",
                "Price",
                "RecommendationSource",
                "DistinctCustomers",
                "UnitsPurchased",
                "Reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="₹%.0f"),
            "DistinctCustomers": st.column_config.NumberColumn(
                "Distinct customers", help="Popularity count, not a probability."
            ),
        },
    )
    st.download_button(
        "Download recommendations",
        data=suggestions.to_csv(index=False).encode("utf-8"),
        file_name=f"recommendations_{selected}.csv",
        mime="text/csv",
    )


def render_dataset(transactions: pd.DataFrame, stats: dict[str, int]) -> None:
    st.header("Dataset")
    st.info(
        "The bundled purchases.csv is reproducible synthetic demo data, not real customer "
        "information. Upload a CSV with the same required columns to use your own data."
    )
    st.write(
        "Required columns: TransactionID, CustomerID, ProductID, ProductName, Category, "
        "Price, Quantity, PurchaseDate."
    )
    st.caption(
        f"Rows loaded: {stats['input_rows']:,} · Valid transactions: {stats['valid_rows']:,} "
        f"· Removed invalid/duplicate rows: {stats['removed_rows']:,}"
    )
    st.dataframe(
        transactions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="₹%.0f"),
        },
    )
    st.download_button(
        "Download active dataset",
        data=transactions.to_csv(index=False).encode("utf-8"),
        file_name="active_purchases.csv",
        mime="text/csv",
    )


def main() -> None:
    ensure_dataset(DATA_PATH)
    st.sidebar.title("ShopScope")
    st.sidebar.caption("Customer insights · Synthetic demo data")
    uploaded_file = st.sidebar.file_uploader("Upload purchase CSV", type=["csv"])
    try:
        if uploaded_file is None:
            source_bytes = DATA_PATH.read_bytes()
            data_label = "Bundled synthetic demo dataset"
        else:
            source_bytes = uploaded_file.getvalue()
            data_label = f"Uploaded: {uploaded_file.name}"
        transactions, stats = load_and_clean(source_bytes)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not load this dataset: {error}")
        st.info("Check that the CSV has the required columns and valid, readable data.")
        return
    except OSError as error:
        st.error(f"Could not read the dataset file: {error}")
        return

    st.sidebar.caption(data_label)
    if transactions.empty:
        st.warning(
            "No valid transactions remain after cleaning. Check that IDs are present, "
            "prices and integer quantities are positive, and dates are valid."
        )
        return
    customers = build_customer_features(transactions)
    if len(customers) < 2:
        st.warning("At least two customers with valid transactions are needed for clustering.")
        return

    numeric_features = feature_columns(customers)
    distinct_count = len(customers[numeric_features].fillna(0).drop_duplicates())
    maximum_clusters = min(len(customers), distinct_count)
    if maximum_clusters < 1:
        st.warning("Customer features do not contain enough distinct rows to cluster.")
        return
    requested_clusters = st.sidebar.slider(
        "Number of clusters",
        min_value=1,
        max_value=maximum_clusters,
        value=min(3, maximum_clusters),
    )
    try:
        clustered_customers, cluster_summary = prepare_customers(
            transactions, requested_clusters
        )
    except ValueError as error:
        st.error(f"Segmentation could not be calculated: {error}")
        return

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Customer Segmentation", "Recommendations", "Dataset"],
    )
    if page == "Overview":
        render_overview(
            transactions,
            clustered_customers,
            cluster_summary,
            stats["input_rows"],
            stats["removed_rows"],
        )
    elif page == "Customer Segmentation":
        render_segmentation(clustered_customers, cluster_summary)
    elif page == "Recommendations":
        render_recommendations(transactions, clustered_customers)
    else:
        render_dataset(transactions, stats)


if __name__ == "__main__":
    main()
