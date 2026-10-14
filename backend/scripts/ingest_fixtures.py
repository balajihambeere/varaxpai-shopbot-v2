"""Ingest the fixture catalog (top-level products.py) into Qdrant.
Run with: python -m scripts.ingest_fixtures
"""

import sys
import time

from products import PRODUCTS
from src.ingestion.ingest import ingest_catalog


def main() -> None:
    start = time.time()
    count = ingest_catalog(PRODUCTS)
    elapsed = time.time() - start
    print(f"Ingested {count} chunks from {len(PRODUCTS)} catalog items in {elapsed:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
