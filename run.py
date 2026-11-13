import subprocess
import sys

def main():
    print("[1/3] Запуск конвейера сбора и генерации данных (ETL)...")

    subprocess.run([sys.executable, "etl_pipeline.py"], check=True)

    print("\n[2/3] Обучение моделей, отбор признаков и валидация...")
    subprocess.run([sys.executable, "model_core.py"], check=True)

    print("\n[3/3] Запуск интерактивного дашборда Streamlit...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"])

if __name__ == "__main__":
    main()
