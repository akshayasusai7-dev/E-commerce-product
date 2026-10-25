"""Dataset validation, cleaning, and customer feature engineering."""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "TransactionID",
    "CustomerID",
    "ProductID",
    "ProductName",
    "Category",
    "Price",
    "Quantity",
    "PurchaseDate",
]


def _normalize_text(value: object) -> str:
    """Convert an identifier or label to a trimmed string without exposing NaN."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_columns(data: pd.DataFrame) -> None:
    """Raise a helpful error when a dataset does not have the required schema."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing)
            + ". Please include all eight documented columns."
        )


def clean_transactions(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize and validate transactions, reporting how many rows were removed."""
    validate_columns(data)
    cleaned = data[REQUIRED_COLUMNS].copy()
    original_count = len(cleaned)

    for column in ("TransactionID", "CustomerID", "ProductID"):
        cleaned[column] = cleaned[column].map(_normalize_text)
    for column in ("ProductName", "Category"):
        cleaned[column] = cleaned[column].map(_normalize_text)

    cleaned["Price"] = pd.to_numeric(cleaned["Price"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["PurchaseDate"] = pd.to_datetime(
        cleaned["PurchaseDate"], errors="coerce"
    )

    required_text = ["TransactionID", "CustomerID", "ProductID"]
    valid = cleaned[required_text].apply(lambda column: column.ne("")).all(axis=1)
    valid &= cleaned["Price"].notna() & np.isfinite(cleaned["Price"])
    valid &= cleaned["Price"].gt(0)
    valid &= cleaned["Quantity"].notna() & np.isfinite(cleaned["Quantity"])
    valid &= cleaned["Quantity"].gt(0)
    valid &= np.isclose(cleaned["Quantity"], np.floor(cleaned["Quantity"]))
    valid &= cleaned["PurchaseDate"].notna()

    cleaned = cleaned.loc[valid].copy()
    cleaned["Quantity"] = cleaned["Quantity"].astype(int)
    cleaned["Price"] = cleaned["Price"].astype(float)
    cleaned["ProductName"] = cleaned["ProductName"].fillna("Unknown product")
    cleaned["Category"] = cleaned["Category"].fillna("Uncategorized")
    cleaned.loc[cleaned["ProductName"].eq(""), "ProductName"] = "Unknown product"
    cleaned.loc[cleaned["Category"].eq(""), "Category"] = "Uncategorized"
    cleaned = cleaned.drop_duplicates(subset=["TransactionID"], keep="first")
    cleaned = cleaned.sort_values(["PurchaseDate", "TransactionID"]).reset_index(
        drop=True
    )

    stats = {
        "input_rows": original_count,
        "valid_rows": len(cleaned),
        "removed_rows": original_count - len(cleaned),
    }
    return cleaned, stats


def build_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction measures and category-spending proportions."""
    if transactions.empty:
        return pd.DataFrame(
            columns=[
                "CustomerID",
                "TotalSpending",
                "TransactionCount",
                "TotalQuantity",
                "AverageTransactionValue",
            ]
        )

    data = transactions.copy()
    data["LineTotal"] = data["Price"] * data["Quantity"]
    customer = data.groupby("CustomerID", as_index=False).agg(
        TotalSpending=("LineTotal", "sum"),
        TransactionCount=("TransactionID", "nunique"),
        TotalQuantity=("Quantity", "sum"),
        AverageTransactionValue=("LineTotal", "sum"),
    )
    customer["AverageTransactionValue"] /= customer["TransactionCount"]

    category_spending = data.pivot_table(
        index="CustomerID",
        columns="Category",
        values="LineTotal",
        aggfunc="sum",
        fill_value=0,
    )
    category_proportions = category_spending.div(
        category_spending.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)
    category_proportions.columns = [
        f"CategoryShare_{category}" for category in category_proportions.columns
    ]

    return customer.merge(
        category_proportions.reset_index(), on="CustomerID", how="left"
    ).fillna(0)
