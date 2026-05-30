#!/usr/bin/env python3
"""Create PostgreSQL database and tables (optional manual setup)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.database import ensure_database_exists, init_db


def main() -> None:
    settings = get_settings()
    print(f"DATABASE_URL={settings.database_url}")
    ensure_database_exists(settings.database_url)
    init_db(settings)
    print("Database ready.")


if __name__ == "__main__":
    main()
