"""Generate reproducible synthetic e-commerce purchase data."""

import csv
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "purchases.csv"

PRODUCTS = {
    "Electronics": [
        ("Wireless Earbuds", 2499),
        ("Bluetooth Speaker", 1899),
        ("Smart Watch", 4999),
        ("USB-C Hub", 1299),
        ("Wireless Mouse", 899),
        ("Mechanical Keyboard", 3299),
        ("Power Bank", 1599),
        ("HD Webcam", 2799),
    ],
    "Fashion": [
        ("Cotton T-Shirt", 699),
        ("Denim Jeans", 1799),
        ("Everyday Sneakers", 2499),
        ("Canvas Backpack", 1399),
        ("Linen Shirt", 1599),
        ("Classic Sunglasses", 1199),
        ("Knit Sweater", 2199),
        ("Leather Wallet", 999),
    ],
    "Home": [
        ("Ceramic Table Lamp", 1899),
        ("Cotton Bed Sheet Set", 2499),
        ("Insulated Water Bottle", 799),
        ("Throw Pillow", 899),
        ("Bamboo Cutting Board", 699),
        ("Storage Basket", 1099),
        ("Non-stick Frying Pan", 1499),
        ("Scented Candle Set", 599),
    ],
    "Books": [
        ("The Art of Learning", 499),
        ("Everyday Cooking", 599),
        ("Mystery at Pine Lake", 399),
        ("Practical Python", 799),
        ("A Brief History of Time", 549),
        ("Mindful Living", 449),
        ("The Design Handbook", 699),
        ("Stories from the City", 379),
    ],
    "Beauty": [
        ("Daily Face Cleanser", 599),
        ("Moisturising Face Cream", 899),
        ("Mineral Sunscreen", 749),
        ("Rose Lip Balm", 299),
        ("Hair Care Set", 1099),
        ("Gentle Body Lotion", 499),
        ("Nourishing Hair Oil", 399),
        ("Makeup Brush Set", 999),
    ],
}


def build_product_catalog() -> list[dict[str, object]]:
    """Return the fixed product catalog used by every generated purchase."""
    products = []
    product_number = 1
    for category, category_products in PRODUCTS.items():
        for name, price in category_products:
            products.append(
                {
                    "ProductID": f"P{product_number:03d}",
                    "ProductName": name,
                    "Category": category,
                    "Price": price,
                }
            )
            product_number += 1
    return products


def generate_dataset(
    customer_count: int = 160,
    seed: int = 42,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
) -> list[dict[str, object]]:
    """Create a reproducible dataset with varied customer category preferences."""
    if customer_count < 1:
        raise ValueError("customer_count must be at least 1.")

    rng = random.Random(seed)
    catalog = build_product_catalog()
    categories = list(PRODUCTS)
    catalog_by_category = {
        category: [product for product in catalog if product["Category"] == category]
        for category in categories
    }
    first_date = date.fromisoformat(start_date)
    last_date = date.fromisoformat(end_date)
    day_count = (last_date - first_date).days
    if day_count < 0:
        raise ValueError("end_date must be on or after start_date.")
    rows = []
    transaction_number = 1

    for customer_number in range(1, customer_count + 1):
        preferred = rng.randrange(len(categories))
        weights = [0.9 if index == preferred else 0.025 for index in range(len(categories))]
        purchase_count = rng.randint(12, 30)

        for _ in range(purchase_count):
            category = rng.choices(categories, weights=weights, k=1)[0]
            product = rng.choice(catalog_by_category[category])
            rows.append(
                {
                    "TransactionID": f"T{transaction_number:06d}",
                    "CustomerID": f"C{customer_number:04d}",
                    "ProductID": product["ProductID"],
                    "ProductName": product["ProductName"],
                    "Category": product["Category"],
                    "Price": float(product["Price"]),
                    "Quantity": rng.choice([1, 1, 1, 2, 2, 3]),
                    "PurchaseDate": (
                        first_date + timedelta(days=rng.randrange(day_count + 1))
                    ).isoformat(),
                }
            )
            transaction_number += 1

    return rows


def ensure_dataset(path: Path = DATA_PATH) -> Path:
    """Create the demo CSV only when it is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        rows = generate_dataset()
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return path


if __name__ == "__main__":
    output_path = ensure_dataset()
    print(f"Synthetic demo dataset saved to {output_path}")
