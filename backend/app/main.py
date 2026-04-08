from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.database import create_tables
from app.routers import auth, users, videos, matches, analytics
from app.routers import processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Volleyball Analytics API...")
    await create_tables()
    await seed_admin()
    logger.info("Database tables ready.")
    yield
    logger.info("Shutting down...")


async def seed_admin():
    """Create default admin user if not exists."""
    from app.database import AsyncSessionLocal
    from app.models.user import User, UserRole
    from sqlalchemy import select
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(User.email == settings.ADMIN_EMAIL)
        )
        if not result.scalar_one_or_none():
            try:
                admin = User(
                    email=settings.ADMIN_EMAIL,
                    username=settings.ADMIN_USERNAME,
                    full_name="System Administrator",
                    password_hash=pwd_context.hash(settings.ADMIN_PASSWORD),
                    role=UserRole.admin,
                    is_active=True,
                )
                db.add(admin)
                await db.commit()
                logger.info(f"Admin user created: {settings.ADMIN_EMAIL}")
            except IntegrityError:
                await db.rollback()
                logger.info(f"Admin user already exists: {settings.ADMIN_EMAIL}")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Based Volleyball Match Analytics Platform API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static files for uploads/thumbnails
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RALLIES_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/rallies", StaticFiles(directory=settings.RALLIES_DIR), name="rallies")

# Routers
app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(matches.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(processing.router, prefix="/api")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
