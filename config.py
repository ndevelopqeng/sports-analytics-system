import os
from dotenv import load_dotenv

load_dotenv()

DB_STRING = os.getenv("DB_CONNECTION_STRING", "sqlite:///sports_analytics.db")
TEAM_ID = int(os.getenv("TEAM_ID", 777))
CITY = os.getenv("CITY", "Moscow")

# Пути к файлам и моделям
EXT_FEED_PATH = os.getenv("EXT_FEED_PATH", "data/external_weather_news.json")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_attendance_model.pkl")
COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Бизнес- и ML-параметры
AVG_TICKET_PRICE = 800
TEST_SIZE = 0.2
RANDOM_STATE = 42
