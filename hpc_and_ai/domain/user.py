from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from domain.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    age = Column(Integer)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)