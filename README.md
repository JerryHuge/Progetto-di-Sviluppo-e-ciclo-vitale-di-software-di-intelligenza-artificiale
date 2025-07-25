# Progetto-di-Sviluppo-e-ciclo-vitale-di-software-di-intelligenza-artificiale
Il progetto è stato realizzato seguendo un approccio strutturato, partendo dalla fase iniziale fino alla deployment automation. Ecco le fasi principali:  

Acquisizione e Preparazione dei Dati  
Ho selezionato il dataset MNIST dalla piattaforma Kaggle: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv  
Per non eccedere i 100MB di push, i file dei dati sono stati esclusi mediante il file .gitignore  

Analisi Esplorativa  
È stata realizzata un’analisi esplorativa dei dati (EDA) tramite un file eda.py  

Sviluppo del Modello  
L'architettura scelta è una CNN (Convolutional Neural Network) implementata con PyTorch   
Il codice per l'addestramento e la validazione si basa su modelli di riferimento forniti durante il corso  
Tutti i risultati, comprese le metriche di accuratezza e la matrice di confusione, sono stati archiviati nella directory outputs/  

Gestione del repository Git  
Dopo un primo tentativo in cui i dati erano stati involontariamente inclusi nel commit (senza .gitignore), è stato creato un nuovo repository pulito.  
Sono state create le cartelle src/, tests/, data/, outputs/ per organizzare meglio il progetto.  
Per semplificare l'esecuzione, ho implementato un Makefile con i comandi principali  

Dockerizzazione  
Il progetto è stato dockerizzato mediante un Dockerfile completo  
Ho utilizzato .dockerignore per escludere file non necessari nel build context  
È stato installato Docker Desktop, e sono stati testati i comandi docker build e docker run per eseguire la pipeline interamente nel container.  


Automazione CI/CD  
Ho configurato GitHub Actions tramite il file .github/workflows/ci.yml  
Il workflow automatizzato include:  
linting del codice (pylint)  
test automatici (pytest)  
Build dell'immagine Docker  
esportazione dell’immagine come file .tar.gz (artifact)  
Il sistema è stato perfezionato iterativamente, correggendo gli errori rilevati durante l'esecuzione degli workflow  