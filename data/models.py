from sqlalchemy import Column, Integer, String, Text
from .database import Base

class Confession(Base):
    __tablename__ = "confessions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)

    approve_count = Column(Integer, default=0)
    disapprove_count = Column(Integer, default=0)

    timestamp_raw = Column(String)
    source = Column(String, default="ispovesti.com")