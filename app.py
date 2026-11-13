import os
import json
import joblib
import shap
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.inspection import permutation_importance

import config
from feature_engineering import FeaturePipeline

st.set_page_config(page_title="Спортивная Аналитика: Посещаемость", layout="wide")

@st.cache_resource
def get_db_engine():
    return create_engine(config.DB_STRING)

def load_dashboard_data():
    engine = get_db_engine()
    df_matches = pd.read_sql("SELECT * FROM matches ORDER BY match_date ASC", engine)
    df_matches['match_date'] = pd.to_datetime(df_matches['match_date'])
    df_logs = pd.read_sql("SELECT * FROM load_logs ORDER BY timestamp DESC LIMIT 10", engine)
    return df_matches, df_logs

st.title("Система прогнозирования посещаемости матчей и продаж билетов")
st.markdown("Пользователь: *Коммерческий департамент и операционные менеджеры стадиона*")

if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.COLS_PATH):
    st.error("Ошибка: файлы обученной модели не найдены. Запустите: python model_core.py")
    st.stop()

model = joblib.load(config.MODEL_PATH)
feature_cols = joblib.load(config.COLS_PATH)
with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

df_matches, df_logs = load_dashboard_data()

# Расчет витрины признаков
pipeline = FeaturePipeline()
features_df = pipeline.build_features_matrix()

future_mask = features_df['attendance'].isna()
if future_mask.any():
    preds = model.predict(features_df.loc[future_mask, feature_cols])
    features_df.loc[future_mask, 'predicted_attendance'] = np.clip(preds, 0, None).astype(int)
else:
    features_df['predicted_attendance'] = np.nan

# ==========================================
# БОКОВАЯ ПАНЕЛЬ: ФИЛЬТРЫ
# ==========================================
st.sidebar.header("Фильтры и разрезы анализа")
all_tournaments = ["Все турниры"] + list(features_df['tournament_raw'].dropna().unique())
selected_tournament = st.sidebar.selectbox("Выберите турнир:", all_tournaments)

all_opponents = ["Все соперники"] + list(features_df['opponent'].dropna().unique())
selected_opponent = st.sidebar.selectbox("Выберите соперника:", all_opponents)

# Применяем фильтры ко всей витрине
df_filtered = features_df.copy()
if selected_tournament != "Все турниры":
    df_filtered = df_filtered[df_filtered['tournament_raw'] == selected_tournament]
if selected_opponent != "Все соперники":
    df_filtered = df_filtered[df_filtered['opponent'] == selected_opponent]

# Вкладки интерфейса
tab1, tab2, tab3 = st.tabs([
    "Операционный дашборд", 
    "Интерпретируемость модели (SHAP)", 
    "Качество моделей и ETL-логи"
])

# ==========================================
# ВКЛАДКА 1: ОПЕРАЦИОННЫЙ ДАШБОРД
# ==========================================
with tab1:
    st.subheader("Ключевые показатели (KPI)")
    col1, col2, col3, col4 = st.columns(4)

    avg_fact = df_filtered[df_filtered['attendance'].notna()]['attendance'].mean()
    col1.metric("Средний факт посещаемости", f"{int(avg_fact) if not pd.isna(avg_fact) else 0} чел.")

    future_filtered = df_filtered[df_filtered['attendance'].isna()]
    if not future_filtered.empty:
        next_match = future_filtered.iloc[0]
        pred_val = int(next_match['predicted_attendance'])
        col2.metric(f"Ближайший матч ({next_match['opponent']})", f"{pred_val} чел.")
        col3.metric("План закупки кейтеринга", f"{int(pred_val * 1.2)} ед.", help="Из расчета 1.2 порции на зрителя")
        col4.metric("Необходимо стюардов", f"{int(pred_val / 50)} чел.", help="Регламент: 1 стюард на 50 зрителей")
    else:
        col2.metric("Ближайший матч", "Нет игр")
        col3.metric("План закупки кейтеринга", "0 ед.")
        col4.metric("Необходимо стюардов", "0 чел.")

    st.markdown("---")
    st.subheader("Динамика фактической посещаемости и прогноза")

    hist_data = df_filtered[df_filtered['attendance'].notna()]
    fut_data = df_filtered[df_filtered['attendance'].isna()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_data['match_date'], y=hist_data['attendance'],
        mode='lines+markers', name='Факт (Прошлые матчи)',
        line=dict(color='#1f77b4', width=2)
    ))
    if not fut_data.empty:
        fig.add_trace(go.Scatter(
            x=fut_data['match_date'], y=fut_data['predicted_attendance'],
            mode='markers+lines', name='Прогноз модели (Будущие матчи)',
            line=dict(color='#ff7f0e', dash='dash', width=2),
            marker=dict(size=9, symbol='diamond')
        ))
    fig.update_layout(xaxis_title="Дата матча", yaxis_title="Зрители (чел.)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Расписание будущих матчей и новостной фон")
    if not fut_data.empty:
        display_fut = fut_data[['match_date', 'opponent', 'tournament_raw', 'temperature', 'conditions', 'news_text', 'predicted_attendance']].copy()
        display_fut.columns = ['Дата', 'Соперник', 'Турнир', 'Темп. (°C)', 'Погода', 'Анонс/Новости', 'Прогноз (чел.)']
        st.dataframe(display_fut, use_container_width=True, hide_index=True)
    else:
        st.info("Нет будущих матчей под выбранные фильтры.")

# ==========================================
# ВКЛАДКА 2: ИНТЕРПРЕТИРУЕМОСТЬ МОДЕЛИ (SHAP)
# ==========================================
with tab2:
    st.header("Объяснимый искусственный интеллект (XAI)")
    st.markdown("Модуль демонстрирует внутреннюю логику принятия решений моделью согласно требованиям проектной работы.")

    X_all = features_df[feature_cols].astype(float)
    explainer = shap.Explainer(model, X_all)
    shap_values = explainer(X_all, check_additivity=False)


    # 1. Оценка важности признаков двумя независимыми способами
    st.subheader("1. Сравнение глобальной важности признаков (2 независимых метода)")
    c_imp1, c_imp2 = st.columns(2)
    
    with c_imp1:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        df_shap_imp = pd.DataFrame({'Признак': feature_cols, 'SHAP важность': mean_abs_shap}).sort_values('SHAP важность', ascending=True)
        fig_shap = px.bar(df_shap_imp, x='SHAP важность', y='Признак', orientation='h', title="Метод 1: Средний вклад SHAP")
        st.plotly_chart(fig_shap, use_container_width=True)

    with c_imp2:
        hist_mask = features_df['attendance'].notna()
        perm_res = permutation_importance(model, features_df.loc[hist_mask, feature_cols], features_df.loc[hist_mask, 'attendance'], n_repeats=5, random_state=42)
        df_perm_imp = pd.DataFrame({'Признак': feature_cols, 'Permutation важность': perm_res.importances_mean}).sort_values('Permutation важность', ascending=True)
        fig_perm = px.bar(df_perm_imp, x='Permutation важность', y='Признак', orientation='h', title="Метод 2: Permutation Importance")
        st.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("---")

    # 2. Объяснение отдельного прогноза
    st.subheader("2. Объяснение прогноза для конкретного матча (Локальная интерпретация)")
    match_options = features_df.apply(lambda r: f"{r['match_id']} | {r['match_date'].strftime('%Y-%m-%d')} | vs {r['opponent']}", axis=1).tolist()
    selected_match_str = st.selectbox("Выберите любой матч для детального разбора факторов:", match_options, index=len(match_options)-1)
    selected_idx = match_options.index(selected_match_str)

    row_info = features_df.iloc[selected_idx]
    pred_single = int(model.predict(X_all.iloc[[selected_idx]])[0])
    fact_single = f"{int(row_info['attendance'])} чел." if pd.notna(row_info['attendance']) else "Будущий матч"
    st.info(f"Матч: {row_info['opponent']} ({row_info['match_date'].strftime('%Y-%m-%d')}) | Прогноз модели: {pred_single} чел. | Факт: {fact_single}")

    fig_waterfall, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[selected_idx], show=False)
    st.pyplot(fig_waterfall, clear_figure=True)

    st.markdown("---")

    # 3. Зависимость прогноза от значения отдельного признака
    st.subheader("3. Зависимость прогноза от значения отдельного признака (Partial Dependence)")
    chosen_feat = st.selectbox("Выберите признак для анализа зависимости:", feature_cols, index=0)
    fig_dep = px.scatter(
        x=X_all[chosen_feat], 
        y=shap_values[:, chosen_feat].values,
        labels={'x': f"Значение признака: {chosen_feat}", 'y': "Влияние на прогноз (SHAP, чел.)"},
        title=f"Как изменение '{chosen_feat}' влияет на посещаемость",
        trendline="lowess"
    )
    st.plotly_chart(fig_dep, use_container_width=True)

    st.markdown("---")

    # 4. Сопоставление структуры важности признаков между различными периодами
    st.subheader("4. Сравнение важности признаков между периодами (Год 1 vs Год 2)")
    mid_point = len(features_df) // 2
    shap_p1 = np.abs(shap_values.values[:mid_point]).mean(axis=0)
    shap_p2 = np.abs(shap_values.values[mid_point:]).mean(axis=0)
    
    df_periods = pd.DataFrame({
        'Признак': feature_cols,
        'Период 1 (Ранний)': shap_p1,
        'Период 2 (Поздний)': shap_p2
    }).melt(id_vars='Признак', var_name='Период', value_name='Важность (SHAP)')
    
    fig_periods = px.bar(df_periods, x='Признак', y='Важность (SHAP)', color='Период', barmode='group', title="Сдвиг значимости факторов во времени")
    st.plotly_chart(fig_periods, use_container_width=True)

# ==========================================
# ВКЛАДКА 3: МЕТРИКИ МОДЕЛЕЙ И ETL-ЛОГИ
# ==========================================
with tab3:
    st.subheader("Сравнение моделей и отбор признаков")
    st.write(f"Выбранная модель: **{metrics_data['best_model']}**")
