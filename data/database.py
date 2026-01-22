from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this is data/
DB_PATH = os.path.join(BASE_DIR, "confessions.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=True)


SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
