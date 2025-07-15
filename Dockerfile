# 1) Base image: usa Python leggero e affidabile
FROM python:3.9-slim
RUN apt-get update && apt-get install -y make && apt-get clean

# 2) Metadati (opzionali)
LABEL maintainer="Tuo Nome <tuo@email>"

# 3) Imposta la cartella di lavoro dentro il container
WORKDIR /app

# 4) Copia e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copia il codice
COPY src/ ./src
COPY src/eda.py ./src/eda.py
COPY tests/ ./tests
COPY Makefile .

# 6) Pre‑crea cartelle di lavoro
RUN mkdir -p data outputs

# 7) Comando di default all’avvio del container
CMD ["make", "all"]