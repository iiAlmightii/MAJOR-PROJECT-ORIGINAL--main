from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    async with engine.begin() as conn:
        from app.models import user, match, video, player, tracking, actions, analytics, logs  # noqa
        from app.models import annotations, rotations, speech_events  # noqa
        await conn.run_sync(Base.metadata.create_all)

    # Lightweight incremental migrations for columns added after initial schema
    # These are safe to run repeatedly (IF NOT EXISTS).
    await _apply_incremental_migrations()


async def _apply_incremental_migrations():
    """
    Add new columns to existing tables without Alembic.
    Each statement is safe to run on a fresh DB (column already exists → no-op).
    """
    statements = [
        # Added: actions.source to tag CV vs speech-derived events
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'cv'",
        # Added: actions.spike enum value — handled by recreating enum is complex;
        # instead we accept 'spike' stored as string via SQLAlchemy native enum
        # Phase A — ball speed
        "ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS speed_kmh FLOAT",
        "ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS vx FLOAT",
        "ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS vy FLOAT",
        # Phase A — action landing zones
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS landing_x FLOAT",
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS landing_y FLOAT",
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS ball_speed_kmh FLOAT",
        # Phase C — reception quality
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS reception_quality SMALLINT",
        # Phase B — player movement
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS distance_covered_m FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_speed_kmh FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS max_speed_kmh FLOAT",
        # Phase C — speed aggregates
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS reception_quality_avg FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_serve_speed_kmh FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_attack_speed_kmh FLOAT",
    ]
    import logging
    logger = logging.getLogger(__name__)
    async with engine.begin() as conn:
        for stmt in statements:
            try:
                await conn.execute(__import__("sqlalchemy").text(stmt))
                logger.debug(f"Migration OK: {stmt[:60]}")
            except Exception as e:
                logger.debug(f"Migration skipped (already applied?): {e}")
