# Makefile

.PHONY: install run start_mlflow

# Variables
PYTHON=python3
MAIN=main.py

# Install dependencies
install:
	pip install torch torch_geometric pandas numpy opencv-python scikit-learn matplotlib ultralytics mlflow

# Start MLflow server
start_mlflow:
	nohup mlflow ui --host 0.0.0.0 --port 5005 &

# Run the project
run: start_mlflow
	$(PYTHON) $(MAIN)
