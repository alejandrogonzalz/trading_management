import os

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import scoped_session, sessionmaker

from app.db.models import Base

# Use DATABASE_URL from environment or default to local file
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/data/trading.db")

# connect_args={"check_same_thread": False} is required for SQLite and FastAPI
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


# Enable WAL mode for better concurrency in SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)


def init_db():
    """Initializes the database by creating all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ SQLite Database Initialized Successfully")
        return True
    except Exception as e:
        print(f"❌ SQLite Database Initialization Failed: {e}")
        return False


def get_db():
    """FastAPI dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ping_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ SQLite Connection Successful")
        return True
    except Exception as e:
        print(f"❌ SQLite Connection Failed: {e}")
        return False
