# Workspace instructions

## Project

This workspace contains an E-Commerce Product Recommendation System built with
Python, Pandas, NumPy, scikit-learn, Matplotlib, and Streamlit. The root
`index.html` is a separate existing project and must not be replaced or changed
when working on the Python dashboard.

## Python dashboard

- `app.py` is the Streamlit user interface.
- `generate_data.py` creates reproducible synthetic purchase data in
  `data/purchases.csv` only when that file does not already exist.
- Reusable validation, feature engineering, clustering, recommendations, and
  chart code belongs in `src/`.
- Keep generated demo data clearly identified as synthetic; do not use external
  APIs, databases, credentials, or real personal information.
- Run the app with `python -m streamlit run app.py`.
- Keep customer cluster labels descriptive only by their measured features;
  cluster numbers are arbitrary.
- Recommendation counts are popularity measures, not prediction probabilities.

## Changes and validation

- Preserve unrelated files and existing user changes.
- Prefer small, typed, readable changes and use the project dependencies already
  listed in `requirements.txt`.
- Validate Python changes with `python -m compileall app.py generate_data.py src`
  and exercise data generation, preprocessing, clustering, and recommendations
  with a small smoke check.
