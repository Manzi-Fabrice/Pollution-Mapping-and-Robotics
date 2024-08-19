import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import heapq

path = '/Users/manzifabriceniyigaba/Desktop/game/pythonProject/.venv/voxelized_data.csv'


def prediction_model(path, test_size=0.2, random_state=42):
    data = pd.read_csv(path)
    x = data[['mean_utm_easting', 'mean_utm_northing', 'mean_depth']].values
    y = data['particle_count'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=random_state)
    gpr.fit(x_train, y_train)

    y_pred, sigma = gpr.predict(x_test, return_std=True)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"mse for prediction model is {mse:4f} and r2 is {r2:4f}")

    return gpr

def simple_correlation(path, surface_threshold=3, num_bins=10, test_size=0.2, random_state=42):
    voxel_data = pd.read_csv(path)
    surface_data = voxel_data[voxel_data['mean_depth'] <= surface_threshold].copy()
    bottom_data = voxel_data[voxel_data['mean_depth'] > surface_threshold].copy()

    surface_bins = np.linspace(surface_data["mean_depth"].min(), surface_data["mean_depth"].max(), num_bins + 1)
    bottom_bins = np.linspace(bottom_data["mean_depth"].min(), bottom_data["mean_depth"].max(), num_bins + 1)

    surface_data['bin'] = pd.cut(surface_data["mean_depth"], bins=surface_bins, labels=False)
    bottom_data['bin'] = pd.cut(bottom_data['mean_depth'], bins=bottom_bins, labels=False)

    matched_data = []
    for _, surface_row in surface_data.iterrows():
        s_bin = surface_row['bin']
        matched = False
        for offset in range(num_bins):
            bins_to_check = [s_bin - offset, s_bin + offset] if offset != 0 else [s_bin]
            for b_bins in bins_to_check:
                if b_bins < 0 or b_bins >= num_bins:
                    continue
                candidate = bottom_data[bottom_data['bin'] == b_bins]

                if not candidate.empty:
                    distances = np.sqrt((candidate['mean_utm_easting'] - surface_row['mean_utm_easting']) ** 2 +
                                        (candidate['mean_utm_northing'] - surface_row['mean_utm_northing']) ** 2)

                    closest_idx = distances.idxmin()
                    closest_match = bottom_data.loc[closest_idx]
                    matched_data.append((surface_row, closest_match))
                    matched = True
                    break
            if matched:
                break

    if not matched_data:
        print("There are no matched points between surface and bottom")
        return None, None, None

    X_surface = np.array([[s.mean_utm_easting, s.mean_utm_northing, s.particle_count] for s, b in matched_data])
    y_bottom = np.array([b.particle_count for s, b in matched_data])

    scaler = StandardScaler()
    X_surface_scaled = scaler.fit_transform(X_surface)

    X_train, X_test, y_train, y_test = train_test_split(X_surface_scaled, y_bottom, test_size=test_size,
                                                        random_state=random_state)

    kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))


    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=random_state)
    gpr.fit(X_train, y_train)

    y_pred, y_std = gpr.predict(X_test, return_std=True)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]

    print(f"Gaussian Process Regressor for correlation model is  - MSE: {mse:.4f}, R²: {r2:.4f}")
    print(f"Correlation between actual and predicted bottom data: {correlation:.2f}")

    return gpr, correlation, scaler

def explore_surrounding_points(gpr, scaler, input_point, search_radius, num_points):
    surrounding_points = []
    step_size = search_radius / np.sqrt(num_points)

    for i in range(-int(np.sqrt(num_points)), int(np.sqrt(num_points))):
        for j in range(-int(np.sqrt(num_points)), int(np.sqrt(num_points))):
            if i == 0 and j == 0:
                continue

            new_point = np.array([input_point[0] + i * step_size, input_point[1] + j * step_size, input_point[2]])
            new_point_scaled = scaler.transform([new_point])
            predicted_count = gpr.predict(new_point_scaled)[0]

            if predicted_count >= 5:  # setting the  threshold
                surrounding_points.append(new_point)

    return surrounding_points

def complex_correlation_model(lat, lon, depth, correlation_model, prediction_model, scaler, threshold=5, search_radius=1000, num_points=100):
    # Create input point
    input_point = np.array([[lon, lat, depth]])
    # Scale input point
    input_point_scaled = scaler.transform(input_point)

    # Predict the surface particle count using the prediction model
    predicted_surface_count = prediction_model.predict(input_point_scaled)[0]
    print(f"Predicted surface particle count at ({lat}, {lon}, {depth}): {predicted_surface_count:.2f}")

    # Use the surface particle count to predict bottom particle count
    input_point_with_surface_count = np.array([[lon, lat, predicted_surface_count]])
    input_point_with_surface_count_scaled = scaler.transform(input_point_with_surface_count)

    predicted_bottom_count = correlation_model.predict(input_point_with_surface_count_scaled)[0]
    print(f"Predicted bottom particle count: {predicted_bottom_count:.2f}")

    if predicted_bottom_count >= threshold:
        print("Detected hotspot at the input point. Consider moving there.")
        return [input_point]  # Return a list of high-prediction points

    print("No significant prediction at the input point. Exploring surrounding points.")

    # Explore surrounding points if no hotspot detected
    high_prediction_points = explore_surrounding_points(prediction_model, scaler, input_point[0], search_radius, num_points)

    if high_prediction_points:
        print("Consider exploring the following high-prediction surrounding points:")
        for point in high_prediction_points:
            print(f"Explore at: {point}")

    return high_prediction_points

def decide_next_surface_point(current_point, prediction_model, visited, threshold, gpr,scaler):
    neighbors = explore_surrounding_points(gpr, scaler, current_point,1000,100)

    best_point = None
    highest_score = -float('inf')

    for point in neighbors:
        if point in visited:
            continue

        predicted_value = prediction_model.predict([point])[0]
        uncertainty = prediction_model.predict([point], return_std=True)[1][0]

        score = predicted_value if predicted_value > threshold else uncertainty

        if score > highest_score:
            highest_score = score
            best_point = point

    return best_point


def importance_score(predicted_value, uncertainty, threshold=5, prediction_weight=0.7, uncertainty_weight=0.3):
    prediction_score = prediction_weight * max(predicted_value - threshold, 0)

    uncertainty_score = uncertainty_weight * uncertainty

    total_score = prediction_score + uncertainty_score

    return total_score



def main_driver(path):
    # Initialize models
    prediction_model_instance = prediction_model(path)
    correlation_model_instance, _, scaler = simple_correlation(path)

    # Priority Queue for exploration
    pq = []
    visited = set()

    # Initialize starting point (example coordinates from the dataset)
    initial_point = [500000, 4500000, 2]  # example UTM easting, northing, depth

    # Predict surface particle count and calculate importance
    initial_point_scaled = scaler.transform([initial_point])
    predicted_surface_count = prediction_model_instance.predict(initial_point_scaled)[0]
    surface_uncertainty = prediction_model_instance.predict(initial_point_scaled, return_std=True)[1][0]
    initial_importance = importance_score(predicted_surface_count, surface_uncertainty)

    # Add initial point to the priority queue with its importance score
    heapq.heappush(pq, (-initial_importance, tuple(initial_point)))

    next_point = initial_point

    while next_point:
        # Get the most important point from the priority queue (but don't pop it yet)
        queue_importance, top_point = pq[0]  # Peek at the top of the queue
        top_point = list(top_point)  # Convert back to list for processing

        # Mark current point as visited
        visited.add(tuple(top_point))

        if top_point[2] <= 3:  # If at the surface
            # Predict the bottom particle count using the correlation model
            bottom_point = [top_point[0], top_point[1], None]  # Depth is managed by the model
            bottom_point_scaled = scaler.transform([[top_point[0], top_point[1], predicted_surface_count]])
            bottom_pred, bottom_uncertainty = correlation_model_instance.predict(bottom_point_scaled, return_std=True)

            bottom_importance = importance_score(bottom_pred[0], bottom_uncertainty[0])

            # Explore surrounding points
            surrounding_points = explore_surrounding_points(correlation_model_instance, scaler, top_point,
                                                            search_radius=1000, num_points=100)

            top_importance = max(
                importance_score(
                    prediction_model_instance.predict(scaler.transform([point]))[0],
                    prediction_model_instance.predict(scaler.transform([point]), return_std=True)[1][0]
                )
                for point in surrounding_points
            )

            # Compare queue_importance, top_importance, and bottom_importance
            if queue_importance >= max(top_importance, bottom_importance):
                heapq.heappop(pq)  # Pop and explore the top of the queue
                next_point = top_point
                # Add the other two importances to the queue
                heapq.heappush(pq, (-top_importance, tuple([top_point[0], top_point[1], 3])))  # Push top_importance point
                heapq.heappush(pq, (-bottom_importance, tuple([top_point[0], top_point[1], None])))  # Push bottom_importance point
            elif top_importance >= max(queue_importance, bottom_importance):
                next_point = max(surrounding_points, key=lambda point: importance_score(
                    prediction_model_instance.predict(scaler.transform([point]))[0],
                    prediction_model_instance.predict(scaler.transform([point]), return_std=True)[1][0]
                ))
                heapq.heappush(pq, (-bottom_importance, tuple([top_point[0], top_point[1], None])))  # Push bottom_importance point
            else:
                next_point = [top_point[0], top_point[1], None]  # Explore bottom_importance point
                heapq.heappush(pq, (-top_importance, tuple([top_point[0], top_point[1], 3])))  # Push top_importance point

            print(f"Exploring point at {next_point} with importance {max(top_importance, bottom_importance, queue_importance):.2f}")

        else:  # If at the bottom
            # Call decide_next_surface_point to find the next surface point
            next_surface_point = decide_next_surface_point(top_point, prediction_model_instance, visited, threshold=5,
                                                           gpr=correlation_model_instance, scaler=scaler)

            # Explore surrounding points
            surrounding_points = explore_surrounding_points(correlation_model_instance, scaler, top_point,
                                                            search_radius=1000, num_points=100)

            # Calculate the importance of going back to the surface and the surrounding
            surface_point_importance = importance_score(
                prediction_model_instance.predict(scaler.transform([next_surface_point]))[0],
                prediction_model_instance.predict(scaler.transform([next_surface_point]), return_std=True)[1][0]
            ) if next_surface_point else -float('inf')

            surrounding_importance = max(
                importance_score(
                    prediction_model_instance.predict(scaler.transform([point]))[0],
                    prediction_model_instance.predict(scaler.transform([point]), return_std=True)[1][0]
                )
                for point in surrounding_points
            )

            # Compare queue_importance, surface_point_importance, and surrounding_importance
            if queue_importance >= max(surface_point_importance, surrounding_importance):
                heapq.heappop(pq)  
                next_point = top_point
                if next_surface_point:
                    heapq.heappush(pq, (-surface_point_importance, tuple(next_surface_point)))
                for point in surrounding_points:
                    heapq.heappush(pq, (-importance_score(
                        prediction_model_instance.predict(scaler.transform([point]))[0],
                        prediction_model_instance.predict(scaler.transform([point]), return_std=True)[1][0]
                    ), tuple(point)))
            elif surface_point_importance >= max(queue_importance, surrounding_importance):
                next_point = next_surface_point
                for point in surrounding_points:
                    heapq.heappush(pq, (-importance_score(
                        prediction_model_instance.predict(scaler.transform([point]))[0],
                        prediction_model_instance.predict(scaler.transform([point]), return_std=True)[1][0]
                    ), tuple(point)))
            else:
                next_point = max(surrounding_points, key=lambda p: importance_score(
                    prediction_model_instance.predict(scaler.transform([p]))[0],
                    prediction_model_instance.predict(scaler.transform([p]), return_std=True)[1][0]
                ))
                if next_surface_point:
                    heapq.heappush(pq, (-surface_point_importance, tuple(next_surface_point)))

            print(f"Exploring point at {next_point} with importance {max(surface_point_importance, surrounding_importance, queue_importance):.2f}")

        visited.add(tuple(next_point))
        if not pq:
            print("No more points to explore.")
            break

main_driver('/Users/manzifabriceniyigaba/Desktop/game/pythonProject/.venv/voxelized_data.csv')
