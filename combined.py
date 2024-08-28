import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import heapq


def load_data(path):
    data = pd.read_csv(path)
    return data


def train_prediction_model(data, test_size=0.2, random_state=42):
    x = data[['mean_utm_easting', 'mean_utm_northing', 'mean_depth']].values
    y = data['particle_count'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=random_state)
    gp.fit(x_train, y_train)
    y_pred, sigma = gp.predict(x_test, return_std=True)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f" the prediction model result is MSE: {mse:.4} and r2: {r2:.4f}")

    return gp


def train_depth_prediction_model(data):
    top_data = data[data['voxel_z'] == 0]
    bottom_data = data[data['voxel_z'] == 1]
    merged_data = pd.merge(top_data, bottom_data, on=['voxel_x', 'voxel_y'], suffixes=('_top', '_bottom'))

    x = merged_data[['voxel_x', 'voxel_y', 'mean_depth_top']].values
    y = merged_data['mean_depth_bottom'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=42)
    gp.fit(x_train, y_train)
    y_pred, sigma = gp.predict(x_test, return_std=True)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Depth Prediction Model - MSE: {mse:.4f}, R²: {r2:.4f}")

    return gp, mse, r2


def train_correlation_model_rf(data):
    top_data = data[data['voxel_z'] == 0]
    bottom_data = data[data['voxel_z'] == 1]
    merged_data = pd.merge(top_data, bottom_data, on=['voxel_x', 'voxel_y'], suffixes=('_top', '_bottom'))
    feature_columns = ['mean_utm_easting_top', 'mean_utm_northing_top', 'particle_count_top']

    x = merged_data[feature_columns].values
    y = merged_data['particle_count_bottom'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation_coefficient = np.corrcoef(y_test, y_pred)[0, 1]
    print(f"Random Forest Correlation Model - MSE: {mse:.4f}, R²: {r2:.4f}, Correlation: {correlation_coefficient:.2f}")

    return rf, correlation_coefficient


def explore_surrounding_points(gpr_model, input_point, search_radius, num_points, threshold):
    explore_surrounding_points = []
    step_size = search_radius / np.sqrt(num_points)

    for i in range(-int(np.sqrt(num_points)), int(np.sqrt(num_points))):
        for j in range(-int(np.sqrt(num_points)), int(np.sqrt(num_points))):
            if i == 0 and j == 0:
                continue
            new_point = np.array([input_point[0] + i * step_size, input_point[1] + j * step_size, input_point[2]])
            predicted_count = gpr_model.predict([new_point])[0]

            if predicted_count >= threshold:
                explore_surrounding_points.append(new_point)
    return explore_surrounding_points


def decide_next_move(current_point, prediction_model, correlation_model, depth_prediction_model, visited, threshold, explore_surrounding_points, heap):
    if tuple(current_point) not in visited:
        if current_point[2] == 0:  # At the top
            # First Option: Explore the surrounding point
            new_points = explore_surrounding_points(prediction_model, current_point, search_radius=1000, num_points=100, threshold=threshold)

            for point in new_points:
                if tuple(point) not in visited:
                    predicted_value, uncertainty = prediction_model.predict([point[:3]], return_std=True)
                    distance_cost = calculate_distance_cost(current_point, point)
                    coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + len(new_points))
                    importance = importance_score(predicted_value, uncertainty, threshold)
                    objective_value = objective_function(importance, distance_cost, coverage_factor)

                    heapq.heappush(heap, (-objective_value, tuple(point)))

            # Second Option: Check correlation to decide on going to the bottom
            predicted_bottom_count = correlation_model.predict([current_point])[0]
            if predicted_bottom_count >= threshold:
                predicted_depth = depth_prediction_model.predict([current_point[:2]])[0]
                point_at_bottom = [current_point[0], current_point[1], predicted_depth]
                predicted_value, uncertainty = prediction_model.predict([point_at_bottom], return_std=True)
                importance = importance_score(predicted_value, uncertainty, threshold)
                distance_cost = calculate_distance_cost(current_point, point_at_bottom)
                coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + 1)
                objective_value_at_bottom = objective_function(importance, distance_cost, coverage_factor)

                if heap and -heap[0][0] > objective_value_at_bottom:
                    next_move = heapq.heappop(heap)[1]
                else:
                    next_move = point_at_bottom

                update_tracking(current_point, heap)
                visited.add(tuple(current_point))
                return next_move

        else:  # At the bottom
            # First Option: Explore Surrounding
            new_points = explore_surrounding_points(prediction_model, current_point, search_radius=1000, num_points=100, threshold=threshold)

            for point in new_points:
                if tuple(point) not in visited:
                    predicted_value, uncertainty = prediction_model.predict([point[:3]], return_std=True)
                    distance_cost = calculate_distance_cost(current_point, point)
                    coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + len(new_points))
                    importance = importance_score(predicted_value, uncertainty, threshold)
                    objective_value = objective_function(importance, distance_cost, coverage_factor)

                    heapq.heappush(heap, (-objective_value, tuple(point)))

            # Second Option: Move back to the top at the current lat/long
            point_at_top = [current_point[0], current_point[1], 0]
            if tuple(point_at_top) not in visited:
                predicted_value, uncertainty = prediction_model.predict([point_at_top], return_std=True)
                importance = importance_score(predicted_value, uncertainty, threshold)
                distance_cost = calculate_distance_cost(current_point, point_at_top)
                coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + 1)
                objective_value_at_top = objective_function(importance, distance_cost, coverage_factor)

                if heap and -heap[0][0] > objective_value_at_top:
                    next_move = heapq.heappop(heap)[1]
                else:
                    next_move = point_at_top

                update_tracking(current_point, heap)
                visited.add(tuple(current_point))
                return next_move

            # Third Option: If neither move to top nor surrounding is better, check neighbors of top not visited
            surrounding_top_points = explore_surrounding_points(prediction_model, point_at_top, search_radius=1000, num_points=100, threshold=threshold)
            for point in surrounding_top_points:
                if tuple(point) not in visited:
                    predicted_value, uncertainty = prediction_model.predict([point[:3]], return_std=True)
                    distance_cost = calculate_distance_cost(current_point, point)
                    coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + len(surrounding_top_points))
                    importance = importance_score(predicted_value, uncertainty, threshold)
                    objective_value = objective_function(importance, distance_cost, coverage_factor)

                    heapq.heappush(heap, (-objective_value, tuple(point)))

            # Finally decide on the next move: top of heap, back to top, or nearby bottom point
            if heap:
                next_move = heapq.heappop(heap)[1]
            else:
                next_move = None

            update_tracking(current_point, heap)
            visited.add(tuple(current_point))
            return next_move


def importance_score(predicted_value, uncertainty, threshold=5, prediction_weight=0.7, uncertainty_weight=0.3):
    prediction_score = prediction_weight * max(predicted_value - threshold, 0)
    uncertainty_score = uncertainty_weight * uncertainty
    total_score = prediction_score + uncertainty_score

    return total_score


def objective_function(importance_score, distance_cost, coverage_factor, alpha=2.0, beta=0.5):
    return importance_score - (alpha * distance_cost + beta * coverage_factor)


def update_tracking(current_point, heap):
    updated_heap = []

    while heap:
        importance_score, point = heapq.heappop(heap)
        new_distance_cost = calculate_distance_cost(current_point, point)
        new_importance_score = importance_score - new_distance_cost
        heapq.heappush(updated_heap, (new_importance_score, point))
    heap[:] = updated_heap


def calculate_distance_cost(point1, point2):
    dis_cost = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    return dis_cost ** 0.5


def calculate_depth_cost(current_depth, target_depth):
    return abs(target_depth - current_depth)


def calculate_coverage_factor(visited_set, total_area):
    coverage = (len(visited_set) / total_area) * 100
    return coverage


def main_driver(path):

    data = load_data(path)

    prediction_model_instance = train_prediction_model(data)
    depth_prediction_model_instance, _, _ = train_depth_prediction_model(data)
    correlation_model_instance, _ = train_correlation_model_rf(data)

    heap = []
    visited = set()

    starting_point = data.iloc[0][['mean_utm_easting', 'mean_utm_northing', 'mean_depth']].values.tolist()
    current_point = starting_point

    predicted_value, uncertainty = prediction_model_instance.predict([current_point], return_std=True)

    importance = importance_score(predicted_value[0], uncertainty[0])
    heapq.heappush(heap, (-importance, tuple(current_point)))

    while heap:
        next_point_to_explore = decide_next_move(current_point, prediction_model_instance,
                                                 correlation_model_instance, depth_prediction_model_instance,
                                                 visited, threshold=5,
                                                 explore_surrounding_points=explore_surrounding_points,
                                                 heap=heap)
        if next_point_to_explore is None or not heap:
            print("Exploration complete.")
            break
        # update  current_point
        current_point = next_point_to_explore

    print("Finished exploring all points.")


