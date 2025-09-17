# NFL Predictor

Local tool + Streamlit app to predict NFL scores and probabilities using nflverse data.

## Quick start
```bash
git clone https://github.com/<YOU>/nfl-predictor.git
cd nfl-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/predict.py --season 2024 --weeks 1 2
