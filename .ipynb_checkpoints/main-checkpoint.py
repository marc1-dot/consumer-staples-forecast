import subprocess
import sys
import os

# Liste des scripts dans l'ordre exact d'exécution
scripts = [
    "src/data_loader.py",
    "src/preprocessing.py",
    "src/create_train_validation_test_split.py",
    "src/model_evaluate.py",
    "src/train_all.py",
    "src/test_all.py",
    "src/feature_importance.py",
    "src/backtesting.py",
    "src/monte_carlo.py"
]

def run_pipeline():
    print("🚀 Starting End-to-End Consumer Staples Forecasting Pipeline...")
    print(f"📂 Working Directory: {os.getcwd()}")
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"❌ Error: Script not found: {script}")
            sys.exit(1)
            
        print(f"\n▶️  Running {script}...")
        # Lancer le script via subprocess
        result = subprocess.run([sys.executable, script], capture_output=False)
        
        if result.returncode != 0:
            print(f"❌ Critical Error in {script}. Pipeline stopped.")
            sys.exit(result.returncode)
            
    print("\n✅ Pipeline completed successfully! Check 'results/' folder for outputs.")

if __name__ == "__main__":
    run_pipeline()
