from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import utm
from opendrift.models.oceandrift import OceanDrift

def initialize_ocean_drift():
    o = OceanDrift(loglevel=30)
    o.add_readers_from_list(['https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be'])
    o.set_config('drift:horizontal_diffusivity', 10)  # m2/s
    return o

def seed_particles(o, num_particles):


    z = np.random.rand(num_particles) * 20  # Depths between 0 and 20 meters

    # Generate random lon and lat for all particles
    lon = 4.8 + np.random.rand(num_particles) * 0.01
    lat = 60.0 + np.random.rand(num_particles) * 0.01

    # Seed the elements into the simulation
    o.seed_elements(lon=lon, lat=lat, z=z, radius=0, number=num_particles, time=datetime.utcnow())

def run_simulation(o, duration, time_step):
    o.run(duration=duration, time_step=time_step)
    return o.elements, o.get_time_array()[0]

def create_dataframe(elements, times):
    data_list = []
    for time in times:
        for lon, lat, z in zip(elements.lon, elements.lat, elements.z):
            data_list.append([time, lon, lat, z])
    df = pd.DataFrame(data_list, columns=['time', 'longitude', 'latitude', 'depth'])
    df['time'] = pd.to_datetime(df['time'])
    df['timestamp'] = df['time'].astype(np.int64) // 10 ** 6
    return df

def convert_to_utm(df):
    utm_coords = df.apply(lambda row: utm.from_latlon(row['latitude'], row['longitude']), axis=1)
    df[['utm_easting', 'utm_northing', 'utm_zone_number', 'utm_zone_letter']] = pd.DataFrame(utm_coords.tolist(), index=df.index)
    return df


def voxelize_data(df, voxel_size, voxel_size_z):
    # Calculate the min and max values for each dimension
    min_easting, max_easting = df['utm_easting'].agg(['min', 'max'])
    min_northing, max_northing = df['utm_northing'].agg(['min', 'max'])
    min_depth, max_depth = df['depth'].agg(['min', 'max'])

    # Determine the maximum indices for the voxel grid
    max_index_x = int(np.ceil((max_easting - min_easting) / voxel_size))
    max_index_y = int(np.ceil((max_northing - min_northing) / voxel_size))
    max_index_z = int(np.ceil((max_depth - min_depth) / voxel_size_z))

    # Initialize a dictionary for voxel data with zero particle counts
    voxel_data = {
        (x, y, z): {
            'sum_utm_easting': 0,
            'sum_utm_northing': 0,
            'sum_depth': 0,
            'particle_count': 0
        }
        for x in range(max_index_x)
        for y in range(max_index_y)
        for z in range(max_index_z)
    }

    # Populate the voxel data
    for idx, row in df.iterrows():
        # Calculate voxel indices
        voxel_x = int((row['utm_easting'] - min_easting) // voxel_size)
        voxel_y = int((row['utm_northing'] - min_northing) // voxel_size)
        voxel_z = int((row['depth'] - min_depth) // voxel_size_z)

        # Update sums and particle count
        voxel_data[(voxel_x, voxel_y, voxel_z)]['sum_utm_easting'] += row['utm_easting']
        voxel_data[(voxel_x, voxel_y, voxel_z)]['sum_utm_northing'] += row['utm_northing']
        voxel_data[(voxel_x, voxel_y, voxel_z)]['sum_depth'] += row['depth']
        voxel_data[(voxel_x, voxel_y, voxel_z)]['particle_count'] += 1

    # Prepare the output data
    data = {
        'voxel_x': [],
        'voxel_y': [],
        'voxel_z': [],
        'mean_utm_easting': [],
        'mean_utm_northing': [],
        'mean_depth': [],
        'particle_count': []
    }

    for (voxel_x, voxel_y, voxel_z), values in voxel_data.items():
        particle_count = values['particle_count']

        # Calculate mean values only if there are particles
        if particle_count > 0:
            mean_utm_easting = values['sum_utm_easting'] / particle_count
            mean_utm_northing = values['sum_utm_northing'] / particle_count
            mean_depth = values['sum_depth'] / particle_count
        else:
            mean_utm_easting = 0
            mean_utm_northing = 0
            mean_depth = 0

        # Append to output data
        data['voxel_x'].append(voxel_x)
        data['voxel_y'].append(voxel_y)
        data['voxel_z'].append(voxel_z)
        data['mean_utm_easting'].append(mean_utm_easting)
        data['mean_utm_northing'].append(mean_utm_northing)
        data['mean_depth'].append(mean_depth)
        data['particle_count'].append(particle_count)

    return pd.DataFrame(data)

def main():
    o = initialize_ocean_drift()
    seed_particles(o, num_particles=1000)
    elements, times = run_simulation(o, duration=timedelta(minutes=20), time_step=20)
    df = create_dataframe(elements, times)
    df = convert_to_utm(df)
    specific_time = df['time'].unique()[14]
    df_selected = df[df['time'] == specific_time]
    voxel_size = 100  # Define voxel size in meters

    # Save original data to CSV
    df_selected.to_csv('original_data.csv', index=False)

    # Voxelize and save voxelized data to CSV
    voxelized_data = voxelize_data(df_selected, voxel_size, 10)
    voxelized_data.to_csv('voxelized_data.csv', index=False)

    print(f"Original data saved to 'original_data.csv'")
    print(f"Voxelized data saved to 'voxelized_data.csv'")

if __name__ == "__main__":
    main()
