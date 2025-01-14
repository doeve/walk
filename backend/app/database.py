# app/database.py
from sqlalchemy import create_engine, text, exc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import event
from contextlib import contextmanager

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://gaituser:gaitpass@db/gaitdb"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=10,
    # Add this to ensure proper transaction handling during table creation
    isolation_level='AUTOCOMMIT'
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Add this function for initial table creation
def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@event.listens_for(engine, "engine_connect")
def ping_connection(connection, branch):
    if branch:
        return
    try:
        connection.scalar(text("SELECT 1"))
    except exc.DBAPIError as err:
        if err.connection_invalidated:
            connection.connection = connection.engine.raw_connection()
        else:
            raise