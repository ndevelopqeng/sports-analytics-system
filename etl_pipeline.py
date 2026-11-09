import requests
import time
from datetime import datetime, timedelta
from db_logger import Session, MatchData, WeatherData, LoadLog, init_db
import config

def log_to_db(session, source, status, records, message):
    """Запись результатов запуска в журнал загрузок"""
    log_entry = LoadLog(source=source, status=status, records_loaded=records, message=message)
    session.add(log_entry)
    session.commit()

def fetch_matches():
    """Сбор матчей: Устойчивость к сбоям + Инкрементальность + Идемпотентность"""
    session = Session()
    source_name = "Sports_API"
    
    # Инкрементальность: ищем дату последнего сохраненного матча
    last_match = session.query(MatchData).order_by(MatchData.match_date.desc()).first()
    start_date = (last_match.match_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_match else "2025-01-01"
    
    url = f"https://sportsdata.io{config.TEAM_ID}"
    params = {"apiKey": config.SPORTS_API_KEY, "fromDate": start_date}
    
    try:
        # Устойчивость: ставим timeout для предотвращения зависания системы при сбое сети
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        matches = response.json()
        
        new_records = 0
        for m in matches:
            # Идемпотентность: проверяем, нет ли уже такого ID в базе (защита от дубликатов)
            exists = session.query(MatchData).filter_by(match_id=str(m['MatchId'])).first()
            if not exists:
                new_match = MatchData(
                    match_id=str(m['MatchId']),
                    match_date=datetime.strptime(m['DateTime'], "%Y-%m-%dT%H:%M:%S"),
                    opponent=m['OpponentName'],
                    tournament=m['CompetitionName'],
                    attendance=m.get('Attendance') # Будет None для будущих матчей
                )
                session.add(new_match)
                new_records += 1
                
        session.commit()
        msg = f"Успешно загружено {new_records} новых матчей начиная с {start_date}."
        log_to_db(session, source_name, "SUCCESS", new_records, msg)
        print(msg)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Ошибка сети/API при сборе матчей: {str(e)}"
        log_to_db(session, source_name, "FAILED", 0, error_msg)
        print(error_msg)
    finally:
        session.close()

def fetch_weather():
    """Сбор метеоданных (Слабоструктурированный источник)"""
    session = Session()
    source_name = "Weather_API"
    
    # Собираем погоду только для дней, когда есть матчи, но данных о погоде еще нет (Инкрементальность)
    match_dates = session.query(MatchData.match_date).all()
    dates_to_fetch = []
    
    for row in match_dates:
        date_normalized = row.match_date.replace(hour=0, minute=0, second=0, microsecond=0)
        weather_exists = session.query(WeatherData).filter_by(date=date_normalized).first()
        if not weather_exists:
            dates_to_fetch.append(date_normalized)
            
    if not dates_to_fetch:
        log_to_db(session, source_name, "SUCCESS", 0, "Все данные о погоде актуальны. Новых дат нет.")
        session.close()
        return

    new_records = 0
    url = f"https://weatherapi.com" # Пример эндпоинта архива/прогноза
    
    for target_date in dates_to_fetch:
        params = {
            "key": config.WEATHER_API_KEY,
            "q": config.CITY,
            "dt": target_date.strftime("%Y-%m-%d")
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            # Обработка ограничений частоты запросов (Rate Limiting) - защита от сбоев
            if response.status_code == 429:
                time.sleep(2) # Пауза при превышении лимита
                continue
                
            response.raise_for_status()
            w_data = response.json()
            
            # Извлечение слабоструктурированных параметров дня
            day_info = w_data['forecast']['forecastday']['day']
            
            new_weather = WeatherData(
                date=target_date,
                temperature=day_info['avgtemp_c'],
                precipitation_prob=day_info.get('daily_chance_of_rain', 0.0),
                conditions=day_info['condition']['text'] # Текстовое описание: "Ливень", "Ясно"
            )
            session.add(new_weather)
            new_records += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Не удалось получить погоду за {target_date}: {e}")
            # Не прерываем весь цикл, обрабатываем ошибку точечно (устойчивость)
            continue
            
    session.commit()
    msg = f"Пайплайн погоды завершен. Добавлено дней: {new_records}."
    log_to_db(session, source_name, "SUCCESS", new_records, msg)
    print(msg)
    session.close()

if __name__ == "__main__":
    print("Запуск инициализации БД...")
    init_db()
    print("Запуск ETL пайплайна...")
    fetch_matches()
    fetch_weather()
    print("Сбор данных успешно завершен.")
