import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_STRING

Base = declarative_base()
engine = create_engine(DB_STRING)
Session = sessionmaker(bind=engine)

logging.basicConfig(
    filename="data_load.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class LoadLog(Base):
    __tablename__ = 'load_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(100))
    status = Column(String(20))
    records_loaded = Column(Integer)
    message = Column(Text)

class MatchData(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True)
    match_id = Column(String(50), unique=True, nullable=False)
    match_date = Column(DateTime, nullable=False)
    opponent = Column(String(100), nullable=False)
    tournament = Column(String(100))
    attendance = Column(Integer, nullable=True)
    
    __table_args__ = (UniqueConstraint('match_id', name='_match_id_uc'),)

class WeatherNewsData(Base):
    __tablename__ = 'weather_news'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, nullable=False)
    temperature = Column(Float)
    precipitation_prob = Column(Float)
    conditions = Column(String(100))
    news_text = Column(Text) # Слабоструктурированный текстовый источник (новости/анонсы)

def init_db():
    Base.metadata.create_all(engine)
