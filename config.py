import os
from dotenv import load_dotenv

load_dotenv()

# Файл конфигурации переменных окружения (требование конфигурируемости)
DB_STRING = os.getenv("DB_CONNECTION_STRING", "sqlite:///sports_analytics.db")

# Глобальные параметры для генерации и обработки данных по спортивной индустрии
TEAM_ID = 777  
CITY = "Moscow"
