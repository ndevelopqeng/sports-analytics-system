import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

# Импортируем пайплайн признаков из Шага 3
from feature_engineering import FeaturePipeline

def mean_absolute_percentage_error(y_true, y_pred):
    """Расчет относительной ошибки MAPE в процентах"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Защита от деления на ноль
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_model_core():
    print("=== [Шаг 4] Запуск Модельного Ядра и Валидации ===")
    
    # 1. Загрузка признаковой матрицы и разделение по времени (Time-Based Split)
    pipeline = FeaturePipeline()
    try:
        features_df = pipeline.build_features_matrix()
        X_train, y_train, X_val, y_val, X_future, feature_cols = pipeline.split_data(features_df)
    except Exception as e:
        print(f" Ошибка при подготовке признаков: {e}")
        print("Пожалуйста, убедитесь, что вы предварительно запустили etl_pipeline.py для наполнения СУБД.")
        return

    # Бизнес-константа для перевода ошибки в понятную пользователю величину (по методичке)
    AVG_TICKET_PRICE = 800  # Средняя цена одного билета в рублях
    
    # 2. Инициализация трех моделей различных классов по методичке
    models = {
        "Базовая модель (Линейная регрессия)": LinearRegression(),
        "Случайный лес (Ансамбль деревьев)": RandomForestRegressor(n_estimators=100, random_state=42),
        "Градиентный бустинг (LightGBM)": LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    }
    
    best_model_name = None
    best_mae = float('inf')
    best_model_obj = None
    results_list = []

    print(f"\n--- Старт обучения на {len(X_train)} матчах и временной валидации на {len(X_val)} матчах ---")
    
    for name, model in models.items():
        # Обучение строго на хронологическом прошлом (Train)
        model.fit(X_train, y_train)
        
        # Прогноз на хронологическом будущем (Validation)
        preds = model.predict(X_val)
        
        # Расчет обязательных метрик точности
        mae = mean_absolute_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)
        
        # Перевод сухой математической ошибки в финансовый показатель (понятная величина)
        financial_risk = mae * AVG_TICKET_PRICE
        
        results_list.append({
            "Модель": name,
            "MAE (в билетах)": round(mae, 1),
            "MAPE (%)": round(mape, 2),
            "Финансовый риск (руб)": round(financial_risk, 0)
        })
        
        # Поиск лучшей модели по MAE
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_model_obj = model

    # 3. Вывод сравнительного отчета (выполнение требования методички)
    df_results = pd.DataFrame(results_list)
    print("\nСравнительная таблица качества моделей (Временная валидация):")
    print("=" * 85)
    print(df_results.to_string(index=False))
    print("=" * 85)
    print(f" Лучшее качество продемонстрировала: {best_model_name}\n")
    
    # 4. Сохранение лучшей модели для интерактивного дашборда
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_obj, "models/best_attendance_model.pkl")
    joblib.dump(feature_cols, "models/feature_columns.pkl")
    print(" Лучшая модель и метаданные признаков успешно сохранены в папку 'models/'.")

    # 5. Демонстрационный расчет прогнозов для будущих матчей (Inference)
    if not X_future.empty:
        future_predictions = best_model_obj.predict(X_future)
        # Ограничиваем снизу нулем и округляем до целых билетов
        future_predictions = np.clip(future_predictions, 0, None).astype(int)
        print(f" Успешно рассчитан прогноз посещаемости для {len(future_predictions)} предстоящих игр.")
        print(f"Прогнозные значения: {future_predictions.tolist()}")
    else:
        print(" В базе данных нет будущих матчей для прогнозирования.")

if __name__ == "__main__":
    run_model_core()
