"""
House Price Prediction using Linear Regression
Run from project root:
    python src/house_price_prediction.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "models" / "house_price_model.pkl"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "YearBuilt",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "GarageCars",
]
TARGET = "SalePrice"


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)
    print("\n--- Dataset ---")
    print("Shape:", df.shape)
    print(df.head())
    print("\nMissing values:")
    print(df[FEATURES + [TARGET]].isnull().sum())

    df = df[FEATURES + [TARGET]].copy()
    df = df.drop_duplicates()
    df = df.dropna()

    return df


def perform_eda(df):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Distribution of house prices
    plt.figure(figsize=(9, 6))
    sns.histplot(df[TARGET], kde=True)
    plt.title("Distribution of House Prices")
    plt.xlabel("Sale Price")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=150)
    plt.close()

    # 2. House price vs living area
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="GrLivArea", y=TARGET)
    plt.title("Living Area vs House Price")
    plt.xlabel("Above-Ground Living Area")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "area_vs_price.png", dpi=150)
    plt.close()

    # 3. House price by overall quality
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="OverallQual", y=TARGET)
    plt.title("House Price by Overall Quality")
    plt.xlabel("Overall Quality")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quality_vs_price.png", dpi=150)
    plt.close()

    # 4. House price by number of bedrooms
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="BedroomAbvGr", y=TARGET)
    plt.title("House Price by Number of Bedrooms")
    plt.xlabel("Bedrooms")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bedrooms_vs_price.png", dpi=150)
    plt.close()

    # 5. House price vs year built
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="YearBuilt", y=TARGET)
    plt.title("Year Built vs House Price")
    plt.xlabel("Year Built")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "year_vs_price.png", dpi=150)
    plt.close()

    # 6. Correlation heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")

    coefficients = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_
    })
    print("\n--- Coefficients ---")
    print(coefficients.to_string(index=False))
    print(f"\nIntercept: {model.intercept_:,.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Residual plot
    residuals = y_test - predictions

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=150)
    plt.close()

    return model, rmse, r2


def save_model(model):
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)


def example_prediction(model):
    example = pd.DataFrame({
        "OverallQual": [7],
        "GrLivArea": [1800],
        "YearBuilt": [2000],
        "TotalBsmtSF": [1000],
        "FullBath": [2],
        "BedroomAbvGr": [3],
        "GarageCars": [2],
    })

    prediction = model.predict(example)[0]

    print("\n--- Example Prediction ---")
    print(example.to_string(index=False))
    print(f"\nPredicted House Price: {prediction:,.2f}")


def main():
    print("=" * 60)
    print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
    print("=" * 60)

    df = load_and_clean_data()
    perform_eda(df)
    model, rmse, r2 = train_model(df)
    save_model(model)
    example_prediction(model)

    print("\nProject completed successfully.")
    print("Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
