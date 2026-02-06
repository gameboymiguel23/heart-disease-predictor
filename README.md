# Heart Disease Predictor (Python + ML)

## Syfte
Projektet tränar en maskininlärningsmodell på Heart Disease Dataset och använder modellen i en terminalbaserad applikation för att förutsäga risk för hjärtsjukdom (target 0/1).

## Dataset
Heart Disease Dataset (Kaggle: johnsmith88/heart-disease-dataset).  
Kolumner: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target.

## Funktioner
- Datainläsning och grundläggande rensning (dubbletter, saknade värden)
- Enkel EDA:
  - target_distribution.png (fördelning av målvariabeln)
  - correlation_heatmap.png (korrelationsmatris)
- Modellering:
  - Train/test split (80/20)
  - Logistic Regression (Pipeline med StandardScaler)
  - Utvärdering: accuracy, confusion matrix, classification report + ROC AUC
- Terminal-app för prediktion med den tränade modellen

## Projektstruktur
- data/heart.csv
- outputs/ (EDA-bilder)
- src/
  - data_processing.py
  - model_training.py
  - utils.py
  - app.py
  - __init__.py
- main.py
- requirements.txt

## Installation
Skapa och aktivera venv (Windows PowerShell i projektmappen):
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
