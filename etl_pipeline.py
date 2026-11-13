import os
import json
import random
from datetime import datetime, timedelta
from db_logger import Session, MatchData, WeatherNewsData, LoadLog, init_db
import config

def log_to_db(session, source, status, records, message):
    log_entry = LoadLog(source=source, status=status, records_loaded=records, message=message)
    session.add(log_entry)
    session.commit()

def generate_external_json_feed(dates_list, opponents_map):
    """Источник 2: Генерация слабоструктурированного JSON-фида (погода + текстовые новости)"""
    os.makedirs(os.path.dirname(config.EXT_FEED_PATH), exist_ok=True)
    
    existing_feed = {}
    if os.path.exists(config.EXT_FEED_PATH):
        with open(config.EXT_FEED_PATH, "r", encoding="utf-8") as f:
            try:
                existing_feed = json.load(f)
            except json.JSONDecodeError:
                existing_feed = {}

    weather_types = [
        {"cond": "Ясно, солнечно", "temp_mod": 5, "rain_prob": 0},
        {"cond": "Переменная облачность", "temp_mod": 2, "rain_prob": 10},
        {"cond": "Небольшой дождь", "temp_mod": -2, "rain_prob": 60},
        {"cond": "Проливной ливень", "temp_mod": -5, "rain_prob": 95},
        {"cond": "Снегопад", "temp_mod": -12, "rain_prob": 80}
    ]
    
    news_templates = [
        "Главный тренер пообещал атакующий футбол против {opp}. Ожидается аншлаг и шоу перед игрой!",
        "Обычная подготовка команды к матчу с {opp}. Травмированных нет, тренировка прошла в штатном режиме.",
        "Скандал и серия поражений перед матчем с {opp}. Болельщики недовольны игрой, часть фанатов бойкотирует игру.",
        "Юбилейный матч клуба против {opp}! Скидки на билеты, автограф-сессия звезд и розыгрыш призов."
    ]

    for dt in dates_list:
        date_str = dt.strftime("%Y-%m-%d")
        if date_str not in existing_feed:
            w_choice = random.choice(weather_types)
            month = dt.month
            base_temp = -7 if month in [12, 1, 2] else (22 if month in [6, 7, 8] else 8)
            final_temp = base_temp + random.randint(-4, 4) + w_choice["temp_mod"]
            opp = opponents_map.get(date_str, "соперником")
            
            existing_feed[date_str] = {
                "temperature": float(final_temp),
                "precipitation_prob": float(w_choice["rain_prob"]),
                "conditions": w_choice["cond"],
                "news_headline": random.choice(news_templates).format(opp=opp)
            }

    with open(config.EXT_FEED_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_feed, f, ensure_ascii=False, indent=2)

def run_etl():
    session = Session()
    init_db()
    
    # --- ИСТОЧНИК 1: Расписание и билетная система (БД) ---
    last_match = session.query(MatchData).order_by(MatchData.match_date.desc()).first()
    if last_match:
        start_date = last_match.match_date + timedelta(days=7)
        num_weeks = 4
        print(f"Инкрементальная догрузка с {start_date.date()}")
    else:
        start_date = datetime.now() - timedelta(days=730)
        num_weeks = 108
        print("Первичная загрузка исторических данных за 2 года...")

    opponents = ["Зенит", "Спартак", "ЦСКА", "Локомотив", "Динамо", "Краснодар", "Ростов", "Рубин"]
    tournaments = ["РПЛ", "РПЛ", "Кубок России"]
    
    generated_dates = []
    opponents_map = {}
    current_date = start_date
    
    for _ in range(num_weeks):
        days_to_weekend = (5 - current_date.weekday()) % 7
        match_date = current_date + timedelta(days=days_to_weekend, hours=random.choice([14, 16, 19]))
        generated_dates.append(match_date)
        opponents_map[match_date.strftime("%Y-%m-%d")] = random.choice(opponents)
        current_date += timedelta(days=7)

    # Генерируем внешний JSON-фид для этих дат
    generate_external_json_feed(generated_dates, opponents_map)

    # Парсинг внешнего JSON-фида (Источник 2)
    new_ext_records = 0
    try:
        with open(config.EXT_FEED_PATH, "r", encoding="utf-8") as f:
            feed_data = json.load(f)
            
        for date_str, info in feed_data.items():
            date_norm = datetime.strptime(date_str, "%Y-%m-%d")
            if not session.query(WeatherNewsData).filter_by(date=date_norm).first():
                session.add(WeatherNewsData(
                    date=date_norm,
                    temperature=info["temperature"],
                    precipitation_prob=info["precipitation_prob"],
                    conditions=info["conditions"],
                    news_text=info["news_headline"]
                ))
                new_ext_records += 1
        session.commit()
        if new_ext_records > 0:
            log_to_db(session, "External_JSON_Weather_News", "SUCCESS", new_ext_records, f"Загружено {new_ext_records} записей из JSON.")
    except Exception as e:
        session.rollback()
        log_to_db(session, "External_JSON_Weather_News", "ERROR", 0, f"Сбой парсинга JSON: {str(e)}")

    # Запись матчей с учетом влияния погоды и новостей
    new_matches = 0
    for match_date in generated_dates:
        match_id = f"MATCH-{match_date.strftime('%Y%m%d')}-{config.TEAM_ID}"
        if not session.query(MatchData).filter_by(match_id=match_id).first():
            date_str = match_date.strftime("%Y-%m-%d")
            opponent = opponents_map[date_str]
            tournament = random.choice(tournaments)
            
            base_attendance = random.randint(18500, 26000) if opponent in ["Зенит", "Спартак", "ЦСКА"] else random.randint(8500, 15000)
            ext_info = feed_data.get(date_str, {})
            cond = ext_info.get("conditions", "")
            temp = ext_info.get("temperature", 10)
            news = ext_info.get("news_headline", "").lower()

            if "ливень" in cond: base_attendance = int(base_attendance * 0.65)
            elif "дождь" in cond: base_attendance = int(base_attendance * 0.85)
            if temp < 0: base_attendance = int(base_attendance * 0.80)
            
            # Влияние новостного фона на посещаемость
            if "аншлаг" in news or "юбилейный" in news:
                base_attendance = int(base_attendance * 1.25)
            elif "скандал" in news or "бойкотирует" in news:
                base_attendance = int(base_attendance * 0.70)

            final_attendance = min(max(1500, base_attendance), 30000)
            if match_date > datetime.now():
                final_attendance = None

            session.add(MatchData(
                match_id=match_id,
                match_date=match_date,
                opponent=opponent,
                tournament=tournament,
                attendance=final_attendance
            ))
            new_matches += 1

    if new_matches > 0:
        session.commit()
        msg = f"Добавлено {new_matches} матчей в БД."
        log_to_db(session, "Matches_Schedule_DB", "SUCCESS", new_matches, msg)
        print(msg)
    else:
        print("Новых матчей не обнаружено (идемпотентность подтверждена).")
        
    session.close()

if __name__ == "__main__":
    run_etl()
