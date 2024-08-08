import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def prediction_model(csv_file, test_size=0.2, random_state=42):
    data = pd.read_csv(csv_file)

    # Extract features and target
    X = data[['mean_utm_easting', 'mean_utm_northing', 'mean_depth']].values
    y = data['particle_count'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Use RationalQuadratic kernel
    kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=random_state)
    gpr.fit(X_train, y_train)

    # Predict on test data
    y_pred, sigma = gpr.predict(X_test, return_std=True)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]

    return {"mse": mse, "r2": r2, "correlation": correlation}


metrics = prediction_model('/Users/manzifabriceniyigaba/Desktop/game/pythonProject/.venv/voxelized_data.csv',
                           test_size=0.2, random_state=42)

# printing the performance
print("RationalQuadratic kernel:")
print(f"  MSE: {metrics['mse']:.4f}")
print(f"  R²: {metrics['r2']:.4f}")
print(f"  Correlation: {metrics['correlation']:.2f}\n")
