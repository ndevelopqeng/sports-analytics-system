import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from config import DB_STRING, TEST_SIZE

class FeaturePipeline:
    def __init__(self, db_string=DB_STRING):
        self.engine = create_engine(db_string)
        
    def load_raw_data(self) -> pd.DataFrame:
        query = """
        SELECT m.match_id, m.match_date, m.opponent, m.tournament, m.attendance,
               w.temperature, w.precipitation_prob, w.conditions, w.news_text
        FROM matches m
        LEFT JOIN weather_news w ON DATE(m.match_date) = DATE(w.date)
        ORDER BY m.match_date ASC
        """
        df = pd.read_sql(query, self.engine)
        df['match_date'] = pd.to_datetime(df['match_date'])
        return df

    def process_unstructured_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка слабоструктурированных данных (погода + NLP-признаки из новостей)"""
        df['temperature'] = df['temperature'].fillna(15.0)
        df['precipitation_prob'] = df['precipitation_prob'].fillna(0.0)
        df['conditions'] = df['conditions'].fillna('unknown').str.lower()
        df['news_text'] = df['news_text'].fillna('').str.lower()
        
        # Метео-признаки
        df['is_rainy'] = df['conditions'].apply(lambda x: 1 if any(w in x for w in ['rain', 'ливень', 'дождь']) else 0)
        df['is_snowy'] = df['conditions'].apply(lambda x: 1 if any(w in x for w in ['snow', 'снегопад', 'снег']) else 0)
        df['weather_severity_index'] = (30.0 - df['temperature']) * (1.0 + df['precipitation_prob'] / 100.0)
        
        # Текстовые признаки из новостного фона (Sentiment / Event extraction)
        pos_words = ['аншлаг', 'шоу', 'юбилейный', 'скидки', 'призов']
        neg_words = ['скандал', 'поражений', 'недовольны', 'бойкотирует']
        
        df['news_positive_score'] = df['news_text'].apply(lambda t: sum(1 for w in pos_words if w in t))
        df['news_negative_score'] = df['news_text'].apply(lambda t: sum(1 for w in neg_words if w in t))
        df['news_length'] = df['news_text'].apply(len)
        return df

    def process_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['month'] = df['match_date'].dt.month
        df['day_of_week'] = df['match_date'].dt.day_of_week
        df['hour'] = df['match_date'].dt.hour
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
        return df

    def process_lag_and_rolling_features(self, df: pd.DataFrame, fill_median: float) -> pd.DataFrame:
        """Лаговые и оконные признаки со сдвигом shift(1) и заполнением строго по train-медиане"""
        df['lag_attendance_1'] = df['attendance'].shift(1).fillna(fill_median)
        df['rolling_mean_attendance_3'] = df['attendance'].shift(1).rolling(window=3, min_periods=1).mean().fillna(fill_median)
        df['ewm_attendance_3'] = df['attendance'].shift(1).ewm(span=3, adjust=False).mean().fillna(fill_median)
        return df

    def encode_categorical(self, df: pd.DataFrame, train_opponent_freq: pd.Series) -> pd.DataFrame:
        """Кодирование категорий без утечки из будущего"""
        default_freq = train_opponent_freq.min() if not train_opponent_freq.empty else 0.05
        df['opponent_freq_encoded'] = df['opponent'].map(train_opponent_freq).fillna(default_freq)
        
        # Сохраняем копию tournament для фильтрации на дашборде
        df['tournament_raw'] = df['tournament']
        df = pd.get_dummies(df, columns=['tournament'], prefix='tour', drop_first=True)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        return df

    def build_features_matrix(self) -> pd.DataFrame:
        df = self.load_raw_data()
        if df.empty:
            raise ValueError("База пуста! Запустите etl_pipeline.py.")
            
        # Определяем границу train для расчета статистик без утечки
        hist_df = df[df['attendance'].notna()]
        split_idx = int(len(hist_df) * (1 - TEST_SIZE))
        train_slice = hist_df.iloc[:split_idx]
        
        train_median = train_slice['attendance'].median()
        if pd.isna(train_median):
            train_median = 12000.0
        train_opp_freq = train_slice['opponent'].value_counts(normalize=True)

        df = self.process_unstructured_features(df)
        df = self.process_calendar_features(df)
        df = self.process_lag_and_rolling_features(df, fill_median=train_median)
        df = self.encode_categorical(df, train_opponent_freq=train_opp_freq)
        return df

    def split_data(self, df: pd.DataFrame, target_column='attendance'):
        historical_df = df[df[target_column].notna()].copy()
        future_idx = df[df[target_column].isna()].index
        
        split_idx = int(len(historical_df) * (1 - TEST_SIZE))
        train_df = historical_df.iloc[:split_idx]
        val_df = historical_df.iloc[split_idx:]
        
        drop_cols = ['match_id', 'match_date', 'opponent', 'tournament_raw', 'conditions', 'news_text', target_column]
        feature_cols = [col for col in train_df.columns if col not in drop_cols]
        
        X_train, y_train = train_df[feature_cols], train_df[target_column]
        X_val, y_val = val_df[feature_cols], val_df[target_column]
        X_future = df.loc[future_idx, feature_cols] if len(future_idx) > 0 else pd.DataFrame(columns=feature_cols)
            
        return X_train, y_train, X_val, y_val, X_future, feature_cols
