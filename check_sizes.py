import os

def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

paths = [
    "models/parent_layer/parent_model_rf.joblib",
    "models/preprocessors/parent_preprocessor.joblib",
]

for p in paths:
    if os.path.exists(p):
        print(f"{p}: {get_size_mb(p):.2f} MB")
    else:
        print(f"{p}: NOT FOUND")
