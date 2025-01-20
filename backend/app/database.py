from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from contextlib import contextmanager

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://gaituser:gaitpass@db/gaitdb"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=10,
    isolation_level='READ COMMITTED'  # Changed from AUTOCOMMIT
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    """Initialize database tables"""
    # Create a new engine specifically for initialization
    init_engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        isolation_level='AUTOCOMMIT'
    )
    Base.metadata.create_all(bind=init_engine)
    init_engine.dispose()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@event.listens_for(engine, "connect")
def do_connect(dbapi_connection, connection_record):
    dbapi_connection.ping(reconnect=True)