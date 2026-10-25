"""Transparent same-cluster and popularity-fallback recommendations."""

import pandas as pd


def recommend_products(
    transactions: pd.DataFrame,
    clustered_customers: pd.DataFrame,
    customer_id: str,
    recommendation_count: int = 5,
) -> pd.DataFrame:
    """Rank unseen products by distinct buyers, then units purchased."""
    if recommendation_count < 1:
        raise ValueError("recommendation_count must be at least 1.")
    customer_rows = clustered_customers.loc[
        clustered_customers["CustomerID"].astype(str).eq(str(customer_id))
    ]
    if customer_rows.empty:
        raise ValueError(f"Customer {customer_id} was not found.")
    customer_cluster = customer_rows.iloc[0]["Cluster"]
    purchased = set(
        transactions.loc[
            transactions["CustomerID"].astype(str).eq(str(customer_id)), "ProductID"
        ].astype(str)
    )

    other_customers = set(
        clustered_customers.loc[
            clustered_customers["Cluster"].eq(customer_cluster), "CustomerID"
        ].astype(str)
    ) - {str(customer_id)}
    same_cluster = transactions.loc[
        transactions["CustomerID"].astype(str).isin(other_customers)
    ]

    def rank_products(source: pd.DataFrame) -> pd.DataFrame:
        unseen = source.loc[~source["ProductID"].astype(str).isin(purchased)]
        if unseen.empty:
            return pd.DataFrame(
                columns=[
                    "ProductID",
                    "ProductName",
                    "Category",
                    "Price",
                    "DistinctCustomers",
                    "UnitsPurchased",
                ]
            )
        return (
            unseen.groupby(
                ["ProductID", "ProductName", "Category", "Price"], as_index=False
            )
            .agg(
                DistinctCustomers=("CustomerID", "nunique"),
                UnitsPurchased=("Quantity", "sum"),
            )
            .sort_values(
                ["DistinctCustomers", "UnitsPurchased", "ProductName"],
                ascending=[False, False, True],
            )
        )

    cluster_ranked = rank_products(same_cluster)
    cluster_ranked["RecommendationSource"] = "Same-cluster customers"
    cluster_ranked["Reason"] = cluster_ranked.apply(
        lambda row: (
            f"Purchased by {row['DistinctCustomers']} other customers in your "
            f"cluster ({row['UnitsPurchased']} units)."
        ),
        axis=1,
    )
    recommendations = cluster_ranked.head(recommendation_count).copy()

    if len(recommendations) < recommendation_count:
        global_ranked = rank_products(transactions)
        global_ranked = global_ranked.loc[
            ~global_ranked["ProductID"].isin(recommendations["ProductID"])
        ].head(recommendation_count - len(recommendations)).copy()
        global_ranked["RecommendationSource"] = "Global popularity fallback"
        global_ranked["Reason"] = global_ranked.apply(
            lambda row: (
                f"Popular across all customers: {row['DistinctCustomers']} distinct "
                f"customers purchased {row['UnitsPurchased']} units."
            ),
            axis=1,
        )
        recommendations = pd.concat(
            [recommendations, global_ranked], ignore_index=True
        )

    return recommendations.reset_index(drop=True)
