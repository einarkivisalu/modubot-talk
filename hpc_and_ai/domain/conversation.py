from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime
from domain.base import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    summary = Column(String)
    last_topic = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)