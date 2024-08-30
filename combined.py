import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


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


def generate_random_point(data):
    min_easting = data['mean_utm_easting'].min()
    max_easting = data['mean_utm_easting'].max()
    min_northing = data['mean_utm_northing'].min()
    max_northing = data['mean_utm_northing'].max()
    min_depth = data['mean_depth'].min()
    max_depth = data['mean_depth'].max()

    random_easting = random.uniform(min_easting, max_easting)
    random_northing = random.uniform(min_northing, max_northing)
    random_depth = random.uniform(min_depth, max_depth)

    return [random_easting, random_northing, random_depth]


def get_voxel_indices(point, voxel_size, voxel_size_z):
    x_index = int(point[0] // voxel_size)
    y_index = int(point[1] // voxel_size)
    z_index = int(point[2] // voxel_size_z)
    return (x_index, y_index, z_index)


def decide_next_move(current_point, prediction_model, correlation_model, depth_prediction_model, visited, threshold,
                     heap, data, voxel_size=100, voxel_size_z=5):
    print(f"Moving to : {current_point}")
    visited.add(tuple(current_point))

    current_voxel = get_voxel_indices(current_point, voxel_size, voxel_size_z)

    if current_point[2] == 0:
        next_point = explore_from_surface(current_point, prediction_model, correlation_model, depth_prediction_model,
                                          visited, threshold, heap)
    else:
        next_point = explore_from_bottom(current_point, prediction_model, visited, threshold, heap)

    # Voxel-based Proximity Check
    if next_point:
        next_voxel = get_voxel_indices(next_point, voxel_size, voxel_size_z)

        # If the next point is in the same voxel, choose a different point
        if current_voxel == next_voxel:
            print(
                f"Next point {next_point} is within the same voxel as the current point {current_point}. Exploring randomly.")
            next_point = explore_randomly(data, heap)

    return next_point


def explore_from_surface(current_point, prediction_model, correlation_model, depth_prediction_model, visited,
                         threshold, heap):
    new_points = explore_surrounding_points(prediction_model, current_point, search_radius=1000, num_points=100,
                                            threshold=threshold)

    add_points_to_heap(new_points, current_point, prediction_model, visited, threshold, heap)

    predicted_bottom_count = correlation_model.predict([current_point])[0]
    if predicted_bottom_count >= threshold:
        return move_to_bottom(current_point, prediction_model, depth_prediction_model, visited, threshold, heap)

    return get_next_point_from_heap(heap)


def explore_from_bottom(current_point, prediction_model, visited,
                        threshold, heap):
    new_points = explore_surrounding_points(prediction_model, current_point, search_radius=1000, num_points=100,
                                            threshold=threshold)

    add_points_to_heap(new_points, current_point, prediction_model, visited, threshold, heap)

    point_at_top = [current_point[0], current_point[1], 0]

    if tuple(point_at_top) not in visited:
        return move_to_top(point_at_top, current_point, prediction_model, visited, threshold, heap)

    print("Both the neighbors of the bottom and the top points are low; checking neighbors of top.")
    surrounding_top_points = explore_surrounding_points(prediction_model, point_at_top, search_radius=1000,
                                                        num_points=100, threshold=threshold)
    add_points_to_heap(surrounding_top_points, current_point, prediction_model, visited, threshold, heap)

    if not heap:
        return explore_randomly(data, heap)

    return get_next_point_from_heap(heap)


def add_points_to_heap(new_points, current_point, prediction_model, visited, threshold, heap):
    for point in new_points:
        if tuple(point) not in visited:
            predicted_value, uncertainty = prediction_model.predict([point[:3]], return_std=True)
            distance_cost = calculate_distance_cost(current_point, point)
            coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + len(new_points))
            importance = importance_score(predicted_value, uncertainty, threshold)
            objective_value = objective_function(importance, distance_cost, coverage_factor)
            heapq.heappush(heap, (-objective_value, tuple(point)))


def move_to_bottom(current_point, prediction_model, depth_prediction_model, visited, threshold, heap):
    predicted_depth = depth_prediction_model.predict([current_point[:2]])[0]
    point_at_bottom = [current_point[0], current_point[1], predicted_depth]
    predicted_value, uncertainty = prediction_model.predict([point_at_bottom], return_std=True)
    importance = importance_score(predicted_value, uncertainty, threshold)
    distance_cost = calculate_distance_cost(current_point, point_at_bottom)
    coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + 1)
    objective_value_at_bottom = objective_function(importance, distance_cost, coverage_factor)

    if heap and -heap[0][0] > objective_value_at_bottom:
        heapq.heappush(heap, (-objective_value_at_bottom, tuple(point_at_bottom)))
        return heapq.heappop(heap)[1]

    else:
        return point_at_bottom


def move_to_top(point_at_top, current_point, prediction_model, visited, threshold, heap):
    predicted_value, uncertainty = prediction_model.predict([point_at_top], return_std=True)
    importance = importance_score(predicted_value, uncertainty, threshold)
    distance_cost = calculate_distance_cost(current_point, point_at_top)
    coverage_factor = calculate_coverage_factor(visited, total_area=len(visited) + 1)
    objective_value_at_top = objective_function(importance, distance_cost, coverage_factor)

    if heap and -heap[0][0] > objective_value_at_top:
        heapq.heappush(heap, (-objective_value_at_top, tuple(point_at_top)))
        return heapq.heappop(heap)[1]
    else:
        return point_at_top


def get_next_point_from_heap(heap):
    if heap:
        return heapq.heappop(heap)[1]
    else:
        return None


def explore_randomly(data, heap):
    random_point = generate_random_point(data)
    heapq.heappush(heap, (-importance_score(0, 0), tuple(random_point)))
    return random_point


def importance_score(predicted_value, uncertainty, threshold=5, prediction_weight=0.8, uncertainty_weight=0.2):
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


def plot_initial_data(ax, voxel_data, threshold):
    hotspots = voxel_data[voxel_data['particle_count'] > threshold]
    ax.scatter(hotspots['mean_utm_easting'], hotspots['mean_utm_northing'], hotspots['mean_depth'],
               c='green', marker='o', label='Hotspots')
    ax.set_xlabel('UTM Easting')
    ax.set_ylabel('UTM Northing')
    ax.set_zlabel('Depth')
    ax.set_title('Initial Hotspots and Robot Path')
    ax.legend()


def animate_robot_movement(frame, path_traveled, ax):
    if frame < len(path_traveled):
        current_point = path_traveled[frame]
        ax.scatter(current_point[0], current_point[1], current_point[2], color='blue', s=50)
        ax.plot(path_traveled[:frame + 1, 0], path_traveled[:frame + 1, 1], path_traveled[:frame + 1, 2], color='red')


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

    print(f"Starting at point {current_point}")

    path_traveled = [current_point]  # To record the robot's path

    iteration_limit = 30
    iteration_count = 0

    while heap:
        current_point = decide_next_move(current_point, prediction_model_instance, correlation_model_instance,
                                         depth_prediction_model_instance, visited, threshold=5, heap=heap, data=data)

        if current_point is None:
            print("No valid moves found.")
            break

        if not heap:
            print("Exploration complete.")
            break

        iteration_count += 1
        if iteration_count > iteration_limit:
            print("Reached iteration limit, terminating exploration.")
            break

        path_traveled.append(current_point)

    print("Finished exploring all points.")

    path_traveled = np.array(path_traveled)

    # Initialize plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_initial_data(ax, data, 5)

    ani = FuncAnimation(fig, animate_robot_movement, frames=len(path_traveled), fargs=(path_traveled, ax), interval=500,
                        repeat=False)

    plt.show()


main_driver("/Users/manzifabriceniyigaba/Desktop/game/pythonProject/.venv/voxelized_data.csv")
