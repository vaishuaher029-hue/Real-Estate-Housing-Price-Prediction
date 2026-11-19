# Real Estate Housing Price Prediction

**Internship project:** Predict median house values using Ridge Regression and Lasso Regression.

## Project contents
- `Real_Estate_Housing_Price_Prediction.ipynb` — notebook with data cleaning, EDA, feature engineering, training, evaluation and plots.
- `train_model.py` — trains and compares both regression models, then saves the best model.
- `app.py` — Streamlit web app to enter housing details and get a prediction.
- `requirements.txt` — required Python packages.

## Run on Windows
1. Extract the ZIP.
2. Open PowerShell in the extracted project folder.
3. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Start the app:

```powershell
streamlit run app.py
```

The first run downloads the public California Housing dataset and trains the models. Keep an internet connection active for this first run. The model and comparison CSV are saved in the project folder.

## Run the notebook
```powershell
jupyter notebook
```
Open `Real_Estate_Housing_Price_Prediction.ipynb` and run cells from top to bottom. The first dataset load requires internet access.

## Dataset and target
The project uses scikit-learn's California Housing dataset. `MedHouseVal` is measured in units of **$100,000**. This is a public educational dataset, not live property listing data or a professional valuation service.

## Models and evaluation
- Ridge Regression (`alpha=1.0`)
- Lasso Regression (`alpha=0.01`)
- StandardScaler inside each model pipeline
- 80/20 train-test split (`random_state=42`)
- MAE, MSE, RMSE and R² metrics
- Residual and actual-vs-predicted visual analysis

## Submission checklist
- [ ] Install dependencies
- [ ] Run the notebook from top to bottom
- [ ] Launch the Streamlit app and test a prediction
- [ ] Keep screenshots of the app and model comparison for your internship submission
- [ ] Include a short explanation of the dataset, algorithms and evaluation metrics in your viva/report

**Note:** Predictions are estimates based on historical district-level data and should not be treated as current market prices.
