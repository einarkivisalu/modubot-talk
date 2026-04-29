from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime
from domain.base import Base

class UserMemory(Base):
    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    key = Column(String)
    value = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)