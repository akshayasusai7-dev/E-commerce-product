"""Customer feature scaling and K-Means segmentation."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


METRIC_COLUMNS = [
    "TotalSpending",
    "TransactionCount",
    "TotalQuantity",
    "AverageTransactionValue",
]


def feature_columns(customer_features: pd.DataFrame) -> list[str]:
    """Return numeric clustering features, excluding the customer identifier."""
    return [
        column
        for column in customer_features.columns
        if column != "CustomerID" and pd.api.types.is_numeric_dtype(customer_features[column])
    ]


def cluster_customers(
    customer_features: pd.DataFrame, cluster_count: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale customer features and assign K-Means cluster labels."""
    if customer_features.empty:
        raise ValueError("There are no customers available to cluster.")
    if cluster_count < 1 or cluster_count > len(customer_features):
        raise ValueError(
            f"Choose between 1 and {len(customer_features)} clusters for this dataset."
        )

    columns = feature_columns(customer_features)
    if not columns:
        raise ValueError("No numeric customer features are available for clustering.")

    values = customer_features[columns].fillna(0)
    distinct_count = len(values.drop_duplicates())
    if distinct_count < cluster_count:
        raise ValueError(
            f"Only {distinct_count} distinct customer feature rows are available; "
            f"choose at most {distinct_count} clusters."
        )

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    result = customer_features.copy()
    result["Cluster"] = model.fit_predict(scaled_values)

    summary_columns = [
        column
        for column in METRIC_COLUMNS
        if column in result.columns
    ]
    summary_columns.extend(
        column for column in columns if column.startswith("CategoryShare_")
    )
    summary = (
        result.groupby("Cluster")
        .agg(CustomerCount=("CustomerID", "nunique"), **{
            column: (column, "mean") for column in summary_columns
        })
        .reset_index()
    )
    return result, summary
