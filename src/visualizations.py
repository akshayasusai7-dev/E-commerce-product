"""Matplotlib charts for sales exploration and customer segments."""

import matplotlib.pyplot as plt
import pandas as pd


BACKGROUND = "#111018"
PURPLE = "#a78bfa"
PALETTE = ["#a78bfa", "#c084fc", "#818cf8", "#e879f9", "#8b5cf6"]


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors="#ded9eb")
    axis.xaxis.label.set_color("#ded9eb")
    axis.yaxis.label.set_color("#ded9eb")
    axis.title.set_color("#f4f0ff")
    for spine in axis.spines.values():
        spine.set_color("#393343")
    axis.grid(axis="y", color="#393343", alpha=0.45)


def category_revenue_chart(transactions: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4), facecolor=BACKGROUND)
    if not transactions.empty:
        revenue = (
            transactions.assign(Revenue=transactions["Price"] * transactions["Quantity"])
            .groupby("Category")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        axis.bar(revenue.index, revenue.values, color=PURPLE)
    axis.set(title="Revenue by category", xlabel="Category", ylabel="Revenue (INR)")
    _style_axis(axis)
    figure.tight_layout()
    return figure


def top_products_chart(transactions: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5), facecolor=BACKGROUND)
    if not transactions.empty:
        products = (
            transactions.groupby("ProductName")["Quantity"]
            .sum()
            .nlargest(10)
            .sort_values()
        )
        axis.barh(products.index, products.values, color=PURPLE)
    axis.set(title="Top 10 products by units sold", xlabel="Units sold", ylabel="Product")
    _style_axis(axis)
    figure.tight_layout()
    return figure


def cluster_count_chart(clustered_customers: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 4), facecolor=BACKGROUND)
    if not clustered_customers.empty:
        counts = clustered_customers["Cluster"].value_counts().sort_index()
        axis.bar(
            [f"Cluster {cluster}" for cluster in counts.index],
            counts.values,
            color=PALETTE[: len(counts)],
        )
    axis.set(title="Customer count by cluster", xlabel="Cluster", ylabel="Customers")
    _style_axis(axis)
    figure.tight_layout()
    return figure


def customer_scatter_chart(clustered_customers: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 5), facecolor=BACKGROUND)
    if not clustered_customers.empty:
        for cluster, customers in clustered_customers.groupby("Cluster"):
            axis.scatter(
                customers["TotalSpending"],
                customers["TransactionCount"],
                label=f"Cluster {cluster}",
                color=PALETTE[int(cluster) % len(PALETTE)],
                alpha=0.8,
                edgecolors="#17131f",
                linewidths=0.4,
            )
        axis.legend(facecolor=BACKGROUND, labelcolor="#ded9eb", framealpha=1)
    axis.set(
        title="Customer spending vs. transaction count",
        xlabel="Total spending (INR)",
        ylabel="Transaction count",
    )
    _style_axis(axis)
    figure.tight_layout()
    return figure
