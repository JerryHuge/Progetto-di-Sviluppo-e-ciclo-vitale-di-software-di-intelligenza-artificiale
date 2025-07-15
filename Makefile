.PHONY: install lint test eda train evaluate all clean

# Installazione delle dipendenze
install:
	python -m pip install --upgrade pip && pip install -r requirements.txt
	@echo "✔ Installation complete. You can now run the project."

# Linting del codice (stile PEP8)
lint:
	pylint --disable=R,C src/*.py tests/*.py eda.py
	@echo "✔ Linting complete. No issues found."

# Esecuzione dei test (con copertura)
test:
	python -m pytest -vv --cov=src tests/
	@echo "✔ Testing complete. All tests passed."

# Analisi esplorativa dei dati
eda:
	python eda.py
	@echo "✔ EDA complete."

# Training del modello
train:
	python src/train_mnist.py
	@echo "✔ Model training complete."

# Valutazione del modello (opzionale, nel caso tu voglia solo fare evaluation)
evaluate:
	python -c "from src.train_mnist import evaluate_mnist; evaluate_mnist()"
	@echo "✔ Model evaluation complete."

# Target default: esegue training + evaluation
all: train evaluate

# Pulizia dei file generati
clean:
	rm -f *.pth confusion_matrix.png
	@echo "✔ Cleaned up generated files."