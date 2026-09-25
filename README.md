# 🏠 House Price Prediction Using Linear Regression

## Project Objective

Build a supervised machine-learning model that predicts house sale prices from property characteristics using Linear Regression.

## Features Used

- OverallQual
- GrLivArea
- YearBuilt
- TotalBsmtSF
- FullBath
- BedroomAbvGr
- GarageCars

**Target:** `SalePrice`

## Workflow

Dataset → Cleaning → EDA → Feature Selection → Train/Test Split → Linear Regression → RMSE & R² → Coefficient Interpretation → Save Model → Example Prediction

## Project Structure

```text
House-Price-Prediction/
├── data/
│   └── train.csv
├── models/
│   └── house_price_model.pkl
├── outputs/
│   ├── correlation_heatmap.png
│   ├── area_vs_price.png
│   └── actual_vs_predicted.png
├── src/
│   └── house_price_prediction.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/house_price_prediction.py
```


## Visualizations

The project generates the following visualizations in the `outputs/` folder:

1. **Price Distribution** – shows the distribution of house sale prices.
2. **Living Area vs House Price** – shows the relationship between living area and price.
3. **Overall Quality vs House Price** – compares prices across quality levels.
4. **Bedrooms vs House Price** – compares prices across bedroom counts.
5. **Year Built vs House Price** – explores the relationship between construction year and price.
6. **Correlation Heatmap** – shows correlations among the selected variables.
7. **Actual vs Predicted Prices** – evaluates prediction performance visually.
8. **Residual Plot** – helps identify systematic prediction errors.

## Evaluation

The project reports:

- **RMSE:** Root Mean Squared Error. Lower values indicate smaller prediction errors.
- **R²:** Proportion of variance in the target explained by the fitted model on the test set.

## Coefficient Interpretation

Each coefficient estimates the change in predicted `SalePrice` associated with a one-unit increase in that feature while holding the other selected features constant.

## Model Saving

The trained model is saved as:

```text
models/house_price_model.pkl
```

## Dataset

The included `data/train.csv` is a small runnable sample with the same columns expected by the code.

For the full Kaggle **House Prices – Advanced Regression Techniques** dataset, download `train.csv` and replace the included file. The full Kaggle dataset contains many more columns; this project intentionally selects seven features.

Dataset source:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

## Academic Use

This project demonstrates:

- Data preprocessing
- Exploratory data analysis
- Feature selection
- Supervised learning
- Linear regression
- Model evaluation
- Model persistence
- New-data prediction
