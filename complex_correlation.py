import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

# This complex correlation model predicts particle counts at a specified location using an existing correlation model
# then after it dynamically explores surrounding areas based on predictions to identify potential hotspots.

def explore_surrounding_points(gpr, scaler, point, search_radius=1000, num_points=100):
    # Generate surrounding points for exploration
    surrounding_points = []
    offsets = np.linspace(-search_radius, search_radius, int(np.sqrt(num_points)))

    for dx in offsets:
        for dy in offsets:
            if dx == 0 and dy == 0:
                continue
            surrounding_point = [point[0] + dx, point[1] + dy, point[2]]  # Same depth
            surrounding_points.append(surrounding_point)

    # Scale the surrounding points
    surrounding_points_scaled = scaler.transform(surrounding_points)

    # Predict at surrounding points
    preds, stds = gpr.predict(surrounding_points_scaled, return_std=True)

    # Identify points with high uncertainty or high prediction
    high_prediction_indices = np.argsort(preds)[-5:]  # Top 5 with highest predictions

    high_prediction_points = [surrounding_points[i] for i in high_prediction_indices]

    print("\nPoints with high predicted particle count:")
    for i, point in enumerate(high_prediction_points):
        print(f"{i + 1}: Point {point}, Predicted count: {preds[high_prediction_indices[i]]:.2f}")

    return high_prediction_points


def complex_correlation_model(lat, lon, depth, correlation_model, scaler, threshold=0.1, search_radius=1000,
                              num_points=100):
    # Create a point from input latitude, longitude, and depth
    input_point = np.array([[lon, lat, depth]])

    # Scale the input point
    input_point_scaled = scaler.transform(input_point)

    # Predict particle count at the input point using the correlation model
    predicted_count = correlation_model.predict(input_point_scaled)[0]

    print(f"Predicted particle count at ({lat}, {lon}, {depth}): {predicted_count:.2f}")

    # Check if the prediction suggests a hotspot
    if predicted_count >= threshold:
        print("Detected hotspot at the input point. Consider moving there.")
        return

    # If not a hotspot, explore surrounding points
    print("No significant prediction at the input point. Exploring surrounding points.")

    # Define and train the Gaussian Process Regressor for surrounding exploration
    kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)


    X_data, y_data = scaler.transform([[lon, lat, depth] for _, _ in voxel_data.iterrows()]), voxel_data[
        'particle_count'].values
    gpr.fit(X_data, y_data)

    # Explore surrounding points for better predictions
    high_prediction_points = explore_surrounding_points(gpr, scaler, input_point[0], search_radius, num_points)

    # Decide to explore points with high predictions
    if high_prediction_points:
        print("Consider exploring the following high-prediction surrounding points:")
        for point in high_prediction_points:
            print(f"Explore at: {point}")

