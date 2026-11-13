import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import plotly.graph_objects as go
from datetime import datetime

# Импортируем конфигурацию и пайплайн признаков
from config import DB_STRING
from feature_engineering import FeaturePipeline

# Настройка страницы Streamlit
st.set_page_config(page_title="Спортивная Аналитика: Посещаемость", layout="wide")

@st.cache_resource
def get_db_engine():
    return create_engine(DB_STRING)

def load_dashboard_data():
    """Загрузка сырых и обработанных данных для графиков"""
    engine = get_db_engine()
    # Читаем матчи
    df_matches = pd.read_sql("SELECT * FROM matches ORDER BY match_date ASC", engine)
    df_matches['match_date'] = pd.to_datetime(df_matches['match_date'])
    
    # Читаем логи для демонстрации автоматизации контроля качества
    df_logs = pd.read_sql("SELECT * FROM load_logs ORDER BY timestamp DESC LIMIT 5", engine)
    return df_matches, df_logs

# Заголовок дашборда
st.title("⚽ Система прогнозирования посещаемости матчей и продаж билетов")
st.markdown("Пользователь: *Коммерческий департамент и операционные менеджеры стадиона*")

# Проверка наличия обученной модели
MODEL_PATH = "models/best_attendance_model.pkl"
COLS_PATH = "models/feature_columns.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLS_PATH):
    st.error("❌ Ошибка: Файлы обученной модели не найдены! Сначала запустите в терминале: `python model_core.py`")
    st.stop()

# Загрузка модели и данных
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(COLS_PATH)
df_matches, df_logs = load_dashboard_data()

# Рассчитываем признаки для будущих матчей через единый пайплайн признаков (Шаг 3)
pipeline = FeaturePipeline()
try:
    features_df = pipeline.build_features_matrix()
    # Выделяем строки, где посещаемость еще не известна (будущие матчи)
    future_rows = features_df[features_df['attendance'].isna()].copy()
    if not future_rows.empty:
        X_future = future_rows[feature_cols]
        # Записываем предсказания модели обратно в датафрейм для визуализации
        preds = model.predict(X_future)
        features_df.loc[features_df['attendance'].isna(), 'predicted_attendance'] = np.clip(preds, 0, None).astype(int)
    else:
        features_df['predicted_attendance'] = np.nan
except Exception as e:
    st.warning(f"Не удалось рассчитать динамический прогноз: {e}")

# ==========================================
# БОКОВАЯ ПАНЕЛЬ: ФИЛЬТРЫ И РАЗРЕЗЫ (По методичке)
# ==========================================
st.sidebar.header("🎛️ Фильтры и разрезы анализа")

# Фильтр по турнирам
all_tournaments = ["Все турниры"] + list(df_matches['tournament'].unique())
selected_tournament = st.sidebar.selectbox("Выберите турнир:", all_tournaments)

# Фильтр по соперникам
all_opponents = ["Все соперники"] + list(df_matches['opponent'].unique())
selected_opponent = st.sidebar.selectbox("Выберите соперника:", all_opponents)

# Применение фильтров к данным
df_filtered = df_matches.copy()
if selected_tournament != "Все турниры":
    df_filtered = df_filtered[df_filtered['tournament'] == selected_tournament]
if selected_opponent != "Все соперники":
    df_filtered = df_filtered[df_filtered['opponent'] == selected_opponent]

# ==========================================
# КЛЮЧЕВЫЕ МЕТРИКИ (KPI)
# ==========================================
st.subheader("📊 Текущие операционные показатели")
col1, col2, col3, col4 = st.columns(4)

avg_fact = df_filtered[df_filtered['attendance'].notna()]['attendance'].mean()
col1.metric("Средний факт посещаемости", f"{int(avg_fact) if not pd.isna(avg_fact) else 0} чел.")

# Берем ближайший будущий матч для вывода оперативного прогноза
future_matches = features_df[features_df['attendance'].isna()]
if not future_matches.empty:
    next_match = future_matches.iloc[0]
    col2.metric(f"Ближайший матч ({next_match['opponent']})", f"{int(next_match['predicted_attendance'])} чел.")
    
    # Бизнес-метрика принятия решений: Расчет кейтеринга (сосиски/напитки из расчета 1.2 единицы на человека)
    catering_needed = int(next_match['predicted_attendance'] * 1.2)
    col3.metric("План закупки кейтеринга (ед.)", f"{catering_needed} шт.", help="Оптимальный объем закупок фудкорта на основе прогноза спроса")
    
    # Расчет стюардов безопасности (1 стюард на 50 зрителей по регламенту)
    stewards_needed = int(next_match['predicted_attendance'] / 50)
    col4.metric("Необходимо стюардов (безопасность)", f"{stewards_needed} чел.", help="Оптимальный вывод персонала безопасности")
else:
    col2.metric("Ближайший матч", "Нет игр в расписании")
    col3.metric("План закупки кейтеринга", "0 шт.")
    col4.metric("Необходимо стюардов", "0 чел.")

st.markdown("---")

# ==========================================
# ОСНОВНОЙ ГРАФИК: ФАКТ И ПРОГНОЗ (По методичке)
# ==========================================
st.subheader("📈 Совмещенный график фактической посещаемости и прогнозных значений")

# Исторические данные (Факт)
hist_data = features_df[features_df['attendance'].notna()]
# Будущие данные (Прогноз)
fut_data = features_df[features_df['attendance'].isna()]

fig = go.Figure()

# Линия Факта
fig.add_trace(go.Scatter(
    x=hist_data['match_date'], 
    y=hist_data['attendance'],
    mode='lines+markers',
    name='Реальный факт (Прошлые матчи)',
    line=dict(color='#1f77b4', width=2)
))

# Точки Прогноза
if not fut_data.empty:
    fig.add_trace(go.Scatter(
        x=fut_data['match_date'], 
        y=fut_data['predicted_attendance'],
        mode='markers+lines',
        name='Прогноз модели (Будущие матчи)',
        line=dict(color='#ff7f0e', dash='dash', width=2),
        marker=dict(size=8, symbol='diamond')
    ))

fig.update_layout(
    xaxis_title="Дата проведения матча",
    yaxis_title="Количество зрителей (чел.)",
    legend=dict(x=0, y=1.1, orientation="h"),
    margin=dict(l=20, r=20, t=20, b=20),
    height=450
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ТАБЛИЦА БУДУЩИХ МАТЧЕЙ И ЖУРНАЛ СИСТЕМЫ
# ==========================================
col_tab1, col_tab2 = st.columns([3, 2])

with col_tab1:
    st.subheader("🗓️ Расписание и прогнозы на ближайшие туры")
    if not fut_data.empty:
        display_fut = fut_data[['match_date', 'opponent', 'temperature', 'conditions', 'predicted_attendance']].copy()
        display_fut.columns = ['Дата матча', 'Соперник', 'Температура (°C)', 'Погода', 'Прогноз посещаемости (чел.)']
        st.dataframe(display_fut, use_container_width=True, hide_index=True)
    else:
        st.info("Все матчи в базе данных уже сыграны. Запустите генератор для добавления новых будущих туров.")

with col_tab2:
    st.subheader("📋 Журнал автоматизированного сбора (ETL)")
    # Показываем системные логи работы, подтверждающие инкрементальность и устойчивость (Шаг 2)
    if not df_logs.empty:
        display_logs = df_logs[['timestamp', 'source', 'status', 'records_loaded']].copy()
        display_logs.columns = ['Время запуска', 'Источник', 'Статус сессии', 'Записано строк']
        st.dataframe(display_logs, use_container_width=True, hide_index=True)
    else:
        st.info("Журнал загрузок пуст.")
