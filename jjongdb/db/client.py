from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseSettings
from config import PROJECT_FOLDER

class DBSettings(BaseSettings):
    """database settings"""

    postgresql: str = ""
    redis: str = ""
    azure_postgresql: str = ""
    oracle2: str = ""
    jjong: str = ""

    class Config:
        """env config file"""

        env_file = f"{PROJECT_FOLDER}/.env"


engine = create_engine(
    url=DBSettings().jjong,
    pool_size=10,
    max_overflow=2,
    pool_recycle=300,
    pool_pre_ping=True,
    pool_use_lifo=True,
    echo=False,
)
session_maker = sessionmaker(engine)


@contextmanager
def session_local():
    """local session maker"""
    session = session_maker()
    try:
        yield session
    except:
        session.rollback()
        raise
    finally:
        session.close()
