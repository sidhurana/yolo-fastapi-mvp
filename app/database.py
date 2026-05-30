from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )

    issue_found: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    message: Mapped[str] = mapped_column(Text, default="")
    summary_suggestion: Mapped[str] = mapped_column(Text, default="")

    model_preset: Mapped[str] = mapped_column(String(64), default="")
    model_path: Mapped[str] = mapped_column(String(512), default="")
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    detection_count: Mapped[int] = mapped_column(Integer, default=0)

    issue_classes_matched: Mapped[list] = mapped_column(JSON, default=list)
    detections: Mapped[list] = mapped_column(JSON, default=list)
    advisories: Mapped[list] = mapped_column(JSON, default=list)

    annotated_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    source_ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)


_engine = None
_SessionLocal = None


def ensure_database_exists(database_url: str) -> None:
    """Create the PostgreSQL database if the server is up but the DB is missing."""
    url = make_url(database_url)
    if not url.drivername.startswith("postgresql"):
        return

    db_name = url.database
    if not db_name:
        return

    # Connect to maintenance DB (postgres) to create the target database.
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")

    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": db_name},
        ).scalar()
        if not exists:
            print(f'Creating PostgreSQL database "{db_name}" …')
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))

    admin_engine.dispose()


def init_db(settings: Optional[Settings] = None) -> None:
    global _engine, _SessionLocal
    settings = settings or get_settings()

    try:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        Base.metadata.create_all(bind=_engine)
    except OperationalError as exc:
        if "does not exist" not in str(exc).lower():
            raise
        ensure_database_exists(settings.database_url)
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        Base.metadata.create_all(bind=_engine)

    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        init_db()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
