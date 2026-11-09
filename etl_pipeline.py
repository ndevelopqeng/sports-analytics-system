import random
from datetime import datetime, timedelta
from db_logger import Session, MatchData, WeatherData, LoadLog, init_db
import config

def log_to_db(session, source, status, records, message):
    log_entry = LoadLog(source=source, status=status, records_loaded=records, message=message)
    session.add(log_entry)
    session.commit()

def generate_synthetic_data():
    session = Session()
    init_db()
    
    source_name = "Synthetic_Sports_Weather_API"
    last_match = session.query(MatchData).order_by(MatchData.match_date.desc()).first()
    
    if last_match:
        start_date = last_match.match_date + timedelta(days=7)
        num_weeks = 5  
        print(f"База данных уже содержит данные. Инкрементальная догрузка с {start_date.date()}")
    else:
        start_date = datetime.now() - timedelta(days=730)
        num_weeks = 110 
        print(f"Инициализация исторической генерации за 2 года...")

    opponents = ["Зенит", "Спартак", "ЦСКА", "Локомотив", "Динамо", "Краснодар", "Ростов", "Рубин"]
    tournaments = ["РПЛ", "РПЛ", "Кубок России"]
    weather_types = [
        {"cond": "Ясно, солнечно", "temp_mod": 5, "rain_prob": 0},
        {"cond": "Переменная облачность", "temp_mod": 2, "rain_prob": 10},
        {"cond": "Небольшой дождь", "temp_mod": -2, "rain_prob": 60},
        {"cond": "Проливной ливень", "temp_mod": -5, "rain_prob": 95},
        {"cond": "Снегопад", "temp_mod": -12, "rain_prob": 80}
    ]

    new_matches, new_weather_days = 0, 0
    current_date = start_date

    for i in range(num_weeks):
        days_to_weekend = (5 - current_date.weekday()) % 7
        match_date = current_date + timedelta(days=days_to_weekend, hours=random.choice([14, 16, 19]))
        match_id = f"MATCH-{match_date.strftime('%Y%m%d')}-{config.TEAM_ID}"
        
        exists_match = session.query(MatchData).filter_by(match_id=match_id).first()
        if not exists_match:
            opponent = random.choice(opponents)
            tournament = random.choice(tournaments)
            
            base_attendance = random.randint(18000, 26000) if opponent in ["Зенит", "Спартак", "ЦСКА"] else random.randint(8000, 15000)
            
            w_choice = random.choice(weather_types)
            month = match_date.month
            base_temp = -7 if month in [12, 1, 2] else (22 if month in [6, 7, 8] else 8)
            
            final_temp = base_temp + random.randint(-4, 4) + w_choice["temp_mod"]
            final_rain = w_choice["rain_prob"]
            conditions = w_choice["cond"]
            
            if "ливень" in conditions: base_attendance = int(base_attendance * 0.65)
            elif "дождь" in conditions: base_attendance = int(base_attendance * 0.85)
            if final_temp < 0: base_attendance = int(base_attendance * 0.80)
            
            final_attendance = min(max(1500, base_attendance), 30000)
            if match_date > datetime.now():
                final_attendance = None

            try:
                session.add(MatchData(match_id=match_id, match_date=match_date, opponent=opponent, tournament=tournament, attendance=final_attendance))
                new_matches += 1
                
                date_norm = match_date.replace(hour=0, minute=0, second=0, microsecond=0)
                if not session.query(WeatherData).filter_by(date=date_norm).first():
                    session.add(WeatherData(date=date_norm, temperature=float(final_temp), precipitation_prob=float(final_rain), conditions=conditions))
                    new_weather_days += 1
            except Exception as e:
                session.rollback()
                continue

        current_date += timedelta(days=7)

    if new_matches > 0:
        session.commit()
        msg = f"Успешно добавлено {new_matches} матчей в SQLite."
        log_to_db(session, source_name, "SUCCESS", new_matches, msg)
        print(f" {msg}")
    else:
        print("База данных уже заполнена.")
    session.close()

if __name__ == "__main__":
    generate_synthetic_data()
