import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor

import config
from feature_engineering import FeaturePipeline

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def run_model_core():
    print("=== Запуск Модельного Ядра, Отбора Признаков и Валидации ===")
    
    pipeline = FeaturePipeline()
    features_df = pipeline.build_features_matrix()
    X_train, y_train, X_val, y_val, X_future, feature_cols = pipeline.split_data(features_df)

    models = {
        "Базовая (Линейная регрессия)": LinearRegression(),
        "Случайный лес (Random Forest)": RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
        "Градиентный бустинг (LightGBM)": LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=config.RANDOM_STATE, verbose=-1)
    }
    
    results_list = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)
        financial_risk = mae * config.AVG_TICKET_PRICE
        
        results_list.append({
            "model": name,
            "MAE": round(float(mae), 1),
            "RMSE": round(float(rmse), 1),
            "MAPE": round(float(mape), 2),
            "financial_risk_rub": round(float(financial_risk), 0)
        })
        trained_models[name] = (model, mae)

    # Выбираем лучшую модель по MAE
    best_model_name = min(trained_models, key=lambda k: trained_models[k][1])
    best_model_obj, best_mae_before = trained_models[best_model_name]

    # --- ОТБОР ПРИЗНАКОВ (требование методички) ---
    perm = permutation_importance(best_model_obj, X_val, y_val, n_repeats=10, random_state=config.RANDOM_STATE, scoring='neg_mean_absolute_error')
    selected_cols = [col for col, imp in zip(feature_cols, perm.importances_mean) if imp > 0]
    
    # Если отсеялось слишком много, оставляем минимум топ-5 признаков
    if len(selected_cols) < 5:
        top_idx = np.argsort(perm.importances_mean)[::-1][:5]
        selected_cols = [feature_cols[i] for i in top_idx]

    # Переобучаем лучшую модель на отобранных признаках и проверяем прирост качества
    best_model_obj.fit(X_train[selected_cols], y_train)
    preds_after = best_model_obj.predict(X_val[selected_cols])
    best_mae_after = mean_absolute_error(y_val, preds_after)

    # Если после отбора стало хуже, возвращаем полный набор признаков
    if best_mae_after > best_mae_before:
        selected_cols = feature_cols
        best_model_obj.fit(X_train[selected_cols], y_train)
        final_mae = best_mae_before
    else:
        final_mae = best_mae_after

    print("\nСравнительная таблица моделей:")
    print(pd.DataFrame(results_list).to_string(index=False))
    print(f"\nЛучшая модель: {best_model_name}")
    print(f"Отбор признаков: оставлено {len(selected_cols)} из {len(feature_cols)} | MAE до: {best_mae_before:.1f} -> MAE после: {final_mae:.1f}")

    # Сохраняем модель, список признаков и метрики для дашборда
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(best_model_obj, config.MODEL_PATH)
    joblib.dump(selected_cols, config.COLS_PATH)
    
    metrics_payload = {
        "best_model": best_model_name,
        "mae_before_selection": round(float(best_mae_before), 1),
        "mae_after_selection": round(float(final_mae), 1),
        "features_total": len(feature_cols),
        "features_selected": len(selected_cols),
        "comparison": results_list
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_model_core()
