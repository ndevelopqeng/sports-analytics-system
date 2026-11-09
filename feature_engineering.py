import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from config import DB_STRING

class FeaturePipeline:
    def __init__(self, db_string=DB_STRING):
        self.engine = create_engine(db_string)
        
    def load_raw_data(self) -> pd.DataFrame:
        """Слияние таблиц матчей и погоды из SQLite БД"""
        query = """
        SELECT m.match_id, m.match_date, m.opponent, m.tournament, m.attendance,
               w.temperature, w.precipitation_prob, w.conditions
        FROM matches m
        LEFT JOIN weather w ON DATE(m.match_date) = DATE(w.date)
        ORDER BY m.match_date ASC
        """
        df = pd.read_sql(query, self.engine)
        df['match_date'] = pd.to_datetime(df['match_date'])
        return df

    def process_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка слабоструктурированных метеоданных"""
        df['temperature'] = df['temperature'].fillna(15.0)  
        df['precipitation_prob'] = df['precipitation_prob'].fillna(0.0)
        df['conditions'] = df['conditions'].fillna('unknown').str.lower()
        
        # Текстовый парсинг слабоструктурированного источника
        df['is_rainy'] = df['conditions'].apply(lambda x: 1 if any(word in x for word in ['rain', 'ливень', 'дождь']) else 0)
        df['is_snowy'] = df['conditions'].apply(lambda x: 1 if any(word in x for word in ['snow', 'снегопад', 'снег']) else 0)
        
        # Аналитический комплексный инженерный признак (Индекс суровости климата)
        df['weather_severity_index'] = (30.0 - df['temperature']) * (1.0 + df['precipitation_prob'] / 100.0)
        return df

    def process_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Формирование календарных признаков"""
        df['month'] = df['match_date'].dt.month
        df['day_of_week'] = df['match_date'].dt.day_of_week  # <-- Исправлено на day_of_week
        df['hour'] = df['match_date'].dt.hour
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
        return df

    def process_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Формирование лаговых и оконных признаков СТРОГО БЕЗ УТЕЧКИ целевой переменной"""
        # Сдвиг на 1 позицию назад (.shift(1)) — залог отсутствия утечек!
        df['lag_attendance_1'] = df['attendance'].shift(1)
        df['rolling_mean_attendance_3'] = df['attendance'].shift(1).rolling(window=3, min_periods=1).mean()
        df['ewm_attendance_3'] = df['attendance'].shift(1).ewm(span=3, adjust=False).mean()
        
        global_median = df['attendance'].median()
        if pd.isna(global_median): global_median = 12000  
            
        df['lag_attendance_1'] = df['lag_attendance_1'].fillna(global_median)
        df['rolling_mean_attendance_3'] = df['rolling_mean_attendance_3'].fillna(global_median)
        df['ewm_attendance_3'] = df['ewm_attendance_3'].fillna(global_median)
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование категориальных шкал"""
        # Частотное кодирование категорий (Обоснование: отражает ранг и популярность соперника)
        freq = df['opponent'].value_counts(normalize=True)
        df['opponent_freq_encoded'] = df['opponent'].map(freq)
        
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=['tournament'], prefix='tour', drop_first=True)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        return df

    def build_features_matrix(self) -> pd.DataFrame:
        """Единый конвейер генерации признаковой витрины"""
        df = self.load_raw_data()
        if df.empty: raise ValueError("База пуста! Запустите etl_pipeline.py.")
        df = self.process_weather_features(df)
        df = self.process_calendar_features(df)
        df = self.process_lag_and_rolling_features(df)
        df = self.encode_categorical(df)
        return df

    def split_data(self, df: pd.DataFrame, target_column='attendance', test_size=0.2):
        """Временное разбиение (Time-Series Split) без перемешивания (Предотвращает утечку будущего)"""
        historical_df = df[df[target_column].notna()].copy()
        future_idx = df[df[target_column].isna()].index
        
        split_idx = int(len(historical_df) * (1 - test_size))
        train_df = historical_df.iloc[:split_idx]
        val_df = historical_df.iloc[split_idx:]
        
        drop_cols = ['match_id', 'match_date', 'opponent', 'conditions', target_column]
        feature_cols = [col for col in train_df.columns if col not in drop_cols]
        
        X_train, y_train = train_df[feature_cols], train_df[target_column]
        X_val, y_val = val_df[feature_cols], val_df[target_column]
        
        # Проверяем, есть ли будущие матчи
        if len(future_idx) > 0:
            X_future = df.loc[future_idx, feature_cols]
        else:
            X_future = pd.DataFrame(columns=feature_cols)
            
        return X_train, y_train, X_val, y_val, X_future, feature_cols

if __name__ == "__main__":
    pipeline = FeaturePipeline()
    try:
        features_df = pipeline.build_features_matrix()
        print("Признаковое пространство успешно сформировано!")
        X_train, _, X_val, _, _, cols = pipeline.split_data(features_df)
        print(f"Сгенерировано признаков для ML: {len(cols)}")
        print(f"Строк для обучения: {X_train.shape[0]} | Строк для валидации: {X_val.shape[0]}")
    except Exception as e:
        print(f"Ошибка пайплайна: {e}")
