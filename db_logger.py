import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_STRING

Base = declarative_base()
engine = create_engine(DB_STRING)
Session = sessionmaker(bind=engine)

# Настройка логирования в файл
logging.basicConfig(
    filename="data_load.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Сводный журнал загрузок в БД (требование методички)
class LoadLog(Base):
    __tablename__ = 'load_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(50))
    status = Column(String(20))
    records_loaded = Column(Integer)
    message = Column(Text)

# Таблица 1: Матчи (Структурированные данные)
class MatchData(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True)
    match_id = Column(String(50), unique=True, nullable=False) # Бизнес-ключ для идемпотентности
    match_date = Column(DateTime, nullable=False)
    opponent = Column(String(100), nullable=False)
    tournament = Column(String(100))
    ticket_sales_start = Column(DateTime)
    attendance = Column(Integer, nullable=True) # Заполняется по факту сыгранного матча
    
    __table_args__ = (UniqueConstraint('match_id', name='_match_id_uc'),)

# Таблица 2: Погода (Слабоструктурированные/Метео данные)
class WeatherData(Base):
    __tablename__ = 'weather'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, nullable=False) # Один прогноз/факт на день матча
    temperature = Column(Float)
    precipitation_prob = Column(Float) # Вероятность осадков
    conditions = Column(String(100)) # Текстовое описание (ясно, ливень, снегопад)

def init_db():
    Base.metadata.create_all(engine)
