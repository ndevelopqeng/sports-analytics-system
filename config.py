import os
from dotenv import load_dotenv

load_dotenv()

SPORTS_API_KEY = os.getenv("SPORTS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
DB_STRING = os.getenv("DB_CONNECTION_STRING", "sqlite:///sports_analytics.db")

# Параметры команды (например, ID футбольного клуба)
TEAM_ID = 42  
CITY = "Moscow"
