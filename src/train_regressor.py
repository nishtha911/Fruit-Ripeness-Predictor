from sklearn.metrics import mean_absolute_error
import os
import joblib
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def generate_synthetic_decay_data(num_samples=1000):

    np.random.seed(42)

    stages = ['Unripe', 'Ripe', 'Overripe']
    ripeness_base_days = {'Unripe': 8.0, 'Ripe': 4.0, 'Overripe': 1.0}

    data = []
    for _ in range(num_samples):

        stage = np.random.choice(stages)
        temp = np.random.uniform(5.0, 35.0)
        humidity = np.random.uniform(30.0, 90.0)

        base_days = ripeness_base_days[stage]

        temp_factor = np.exp(-0.05 * (temp - 15.0))

        humidity_factor = 1.0 - 0.002 * abs(humidity - 65.0)

        days_left = max(0.0, base_days * temp_factor * humidity_factor + np.random.normal(0, 0.3))

        data.append({
            'ripeness_stage': stage,
            'storage_temp_c': round(temp, 1),
            'humidity_pct': round(humidity, 1),
            'remaining_shelf_life_days': round(days_left, 1)
        })

    return pd.DataFrame(data)

def train_regressor():
    print("Generating training dataset for shelf-life estimation...")
    df = generate_synthetic_decay_data(num_samples=2000)

    X = df[['ripeness_stage', 'storage_temp_c', 'humidity_pct']]
    y = df['remaining_shelf_life_days']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['ripeness_stage']
    numerical_features = ['storage_temp_c', 'humidity_pct']

    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    model_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor), 
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("Training the Random Forest Regressor...")
    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    

    print(f"\nEvaluation Results:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"  R² Score: {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = "models/shelflife_regressor.pkl"
    joblib.dump(model_pipeline, save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    train_regressor()

    
    
    