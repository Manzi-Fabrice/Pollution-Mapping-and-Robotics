import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def random_forest_correlation(csv_file_path, surface_threshold=3, num_bins=10, test_size=0.2, random_state=42):
    voxel_data = pd.read_csv(csv_file_path)

    # Separate surface and bottom data
    surface_data = voxel_data[voxel_data['mean_depth'] >= surface_threshold].copy()
    bottom_data = voxel_data[voxel_data['mean_depth'] < surface_threshold].copy()

    # Define equal bin edges for both surface and bottom
    surface_bins = np.linspace(surface_data['mean_depth'].min(), surface_data['mean_depth'].max(), num_bins + 1)
    bottom_bins = np.linspace(bottom_data['mean_depth'].min(), bottom_data['mean_depth'].max(), num_bins + 1)

    # Bin the data
    surface_data['bin'] = pd.cut(surface_data['mean_depth'], bins=surface_bins, labels=False)
    bottom_data['bin'] = pd.cut(bottom_data['mean_depth'], bins=bottom_bins, labels=False)

    matched_data = []

    # For each surface bin, find corresponding bottom data
    for _, surface_row in surface_data.iterrows():
        s_bin = surface_row['bin']
        matched = False

        # Check for corresponding bottom data starting from the same bin
        for offset in range(num_bins):
            # Create bin indices to check, based on offset
            bins_to_check = [s_bin - offset, s_bin + offset] if offset != 0 else [s_bin]

            for b_bin in bins_to_check:
                if b_bin < 0 or b_bin >= num_bins:
                    continue  # Skip out-of-range bins

                candidates = bottom_data[bottom_data['bin'] == b_bin]

                if not candidates.empty:
                    # Choose the closest spatial candidate
                    distances = np.sqrt((candidates['mean_utm_easting'] - surface_row['mean_utm_easting'])**2 +
                                        (candidates['mean_utm_northing'] - surface_row['mean_utm_northing'])**2)
                    closest_idx = distances.idxmin()
                    closest_match = bottom_data.loc[closest_idx]
                    matched_data.append((surface_row, closest_match))
                    matched = True
                    break

            if matched:
                break

    if not matched_data:
        print("No matching data found between surface and bottom.")
        return None, None

    # Extract features and target for Random Forest
    X_surface = np.array([[s.mean_utm_easting, s.mean_utm_northing, s.mean_depth] for s, b in matched_data])
    y_bottom = np.array([b.particle_count for s, b in matched_data])

    # Scaling the features
    scaler = StandardScaler()
    X_surface_scaled = scaler.fit_transform(X_surface)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_surface_scaled, y_bottom, test_size=test_size, random_state=random_state)

    # Define and train the Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)

    # Predict bottom particle counts
    y_pred = rf.predict(X_test)

    # Calculate correlation between actual and predicted bottom data
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]

    print(f"Random Forest - MSE using binning: {mse:.4f}, R²: {r2:.4f}")
    print(f"Correlation between actual and predicted bottom data using binning: {correlation:.2f}")

    return rf, correlation

rf_model, correlation = random_forest_correlation('/Users/manzifabriceniyigaba/Desktop/game/pythonProject/.venv/voxelized_data.csv')


