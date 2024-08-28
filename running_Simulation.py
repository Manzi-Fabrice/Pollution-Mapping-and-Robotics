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
    num_pairs = num_particles // 2
    z_surface = np.random.uniform(0, 1, num_pairs)  
    z_bottom = np.random.uniform(7, 8, num_pairs)  
    lon = 5.0 + np.random.rand(num_pairs) * 0.01
    lat = 60.0 + np.random.rand(num_pairs) * 0.01
    lon = np.concatenate((lon, lon))
    lat = np.concatenate((lat, lat))
    z = np.concatenate((z_surface, z_bottom))
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
    df[['utm_easting', 'utm_northing', 'utm_zone_number', 'utm_zone_letter']] = pd.DataFrame(utm_coords.tolist(),
                                                                                             index=df.index)
    return df


def voxelize_data(df, voxel_size, voxel_size_z):
    min_easting, max_easting = df['utm_easting'].agg(['min', 'max'])
    min_northing, max_northing = df['utm_northing'].agg(['min', 'max'])
    min_depth, max_depth = df['depth'].agg(['min', 'max'])

    num_voxels_x = int(np.ceil((max_easting - min_easting) / voxel_size))
    num_voxels_y = int(np.ceil((max_northing - min_northing) / voxel_size))
    num_voxels_z = int(np.ceil((max_depth - min_depth) / voxel_size_z))

    voxel_data = []

    for x in range(num_voxels_x):
        for y in range(num_voxels_y):
            for z in range(num_voxels_z):
                voxel_min_easting = min_easting + x * voxel_size
                voxel_max_easting = voxel_min_easting + voxel_size
                voxel_min_northing = min_northing + y * voxel_size
                voxel_max_northing = voxel_min_northing + voxel_size
                voxel_min_depth = min_depth + z * voxel_size_z
                voxel_max_depth = voxel_min_depth + voxel_size_z

                in_voxel = df[
                    (df['utm_easting'] >= voxel_min_easting) & (df['utm_easting'] < voxel_max_easting) &
                    (df['utm_northing'] >= voxel_min_northing) & (df['utm_northing'] < voxel_max_northing) &
                    (df['depth'] >= voxel_min_depth) & (df['depth'] < voxel_max_depth)
                ]

                if len(in_voxel) > 0:
                    mean_utm_easting = in_voxel['utm_easting'].mean()
                    mean_utm_northing = in_voxel['utm_northing'].mean()
                    mean_depth = in_voxel['depth'].mean()
                    particle_count = len(in_voxel)
                else:
                    mean_utm_easting = (voxel_min_easting + voxel_max_easting) / 2
                    mean_utm_northing = (voxel_min_northing + voxel_max_northing) / 2
                    mean_depth = (voxel_min_depth + voxel_max_depth) / 2
                    particle_count = 0
                voxel_data.append([x, y, z, mean_utm_easting, mean_utm_northing, mean_depth, particle_count])
    voxel_df = pd.DataFrame(voxel_data, columns=['voxel_x', 'voxel_y', 'voxel_z', 'mean_utm_easting', 'mean_utm_northing', 'mean_depth', 'particle_count'])

    return voxel_df




def main():
    o = initialize_ocean_drift()
    seed_particles(o, num_particles=1000)
    elements, times = run_simulation(o, duration=timedelta(minutes=20), time_step=20)
    df = create_dataframe(elements, times)
    df = convert_to_utm(df)
    specific_time = df['time'].unique()[14]
    df_selected = df[df['time'] == specific_time]
    voxel_size = 100  
    df_selected.to_csv('original_data.csv', index=False)

    voxelized_data = voxelize_data(df_selected, voxel_size, 5)
    voxelized_data.to_csv('voxelized_data.csv', index=False)

    print(f"Original data saved to 'original_data.csv'")
    print(f"Voxelized data saved to 'voxelized_data.csv'")


if __name__ == "__main__":
    main()
