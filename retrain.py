def retrain_pipeline():
    import subprocess

    subprocess.run(["python", "train_xgb.py"])

    print("Model retrained successfully")