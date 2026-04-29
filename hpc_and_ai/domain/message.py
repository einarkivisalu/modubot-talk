from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime
from domain.base import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)
    content = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)