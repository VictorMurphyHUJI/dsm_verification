import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import multiprocessing as mp

def fun_dynsys_univariate_analysis(data_matrix, quanti):
    print('Computing dynamical quantities')
    n_samples = data_matrix.shape[0]
    D1 = np.zeros(n_samples)
    theta = np.zeros(n_samples)

    for j in range(n_samples):
        # Compute the observables
        distances = np.linalg.norm(data_matrix[j, :] - data_matrix, axis=1)
        distances[distances == 0] = np.finfo(float).eps  # Replace zero distances with a small positive value
        logdista = -np.log(distances)

        # Extract the threshold corresponding to the quantile defined
        thresh = np.quantile(logdista, quanti)

        # Compute the extremal index using the Sueveges formula
        theta[j] = fun_extremal_index_sueveges(logdista, quanti, thresh)

        # Sort the time series and find all the PoTs (peaks over threshold)
        logextr = logdista[logdista > thresh]
        logextr = logextr[np.isfinite(logextr)]

        # Extract the GPD parameters
        if len(logextr) > 0:
            D1[j] = 1.0 / np.mean(logextr - thresh)
        else:
            D1[j] = np.nan  # Handle cases with no exceedances

    return D1, theta


def fun_extremal_index_sueveges(Y, p, u=None):
    if u is None:
        u = np.quantile(Y, p)

    q = 1 - p
    Li = np.where(Y > u)[0]
    Ti = np.diff(Li)
    Si = Ti - 1
    Nc = np.sum(Si > 0)
    N = len(Ti)

    if len(Si) == 0 or Nc == 0:
        return np.nan

    sum_qSi = np.sum(q * Si)
    numerator = sum_qSi + N + Nc
    denominator = 2 * sum_qSi
    discriminant = numerator ** 2 - 8 * Nc * sum_qSi

    if discriminant < 0:
        return np.nan

    theta = (numerator - np.sqrt(discriminant)) / denominator
    return theta


# Load the dataset
ds = xr.open_dataset("combined_geopotential_500hPa_daily.nc")

# dataset is ERA5 hourly average data from 1950 to 2024 end of year Northern Hemisphere.


# Geographic subset: 75°N to 5°N latitude, -80°W to 45°E longitude
print("Original dataset shape:", ds['z'].shape)
print("Original latitude range:", ds['latitude'].min().values, "to", ds['latitude'].max().values)
print("Original longitude range:", ds['longitude'].min().values, "to", ds['longitude'].max().values)

# Select the geographic region
ds_subset = ds.sel(
    latitude=slice(75, 5),      # 75°N to 5°N (assuming latitude is in descending order)
    longitude=slice(-80, 45)    # -80°W to 45°E
)

print("Subset dataset shape:", ds_subset['z'].shape)
print("Subset latitude range:", ds_subset['latitude'].min().values, "to", ds_subset['latitude'].max().values)
print("Subset longitude range:", ds_subset['longitude'].min().values, "to", ds_subset['longitude'].max().values)


# Coarsen to 1° resolution
ds_coarse = ds.coarsen(latitude=4, longitude=4, boundary="trim").mean()

print("Coarsened dataset shape:", ds_coarse['z'].shape)
print("Coarsened latitude range:", ds_coarse['latitude'].min().values, "to", ds_coarse['latitude'].max().values)
print("Coarsened longitude range:", ds_coarse['longitude'].min().values, "to", ds_coarse['longitude'].max().values)

# Stack spatial dimensions into a time-series matrix
data_matrix = ds_coarse['z'].stack(space=('latitude', 'longitude')).values


print(f"New data matrix shape: {data_matrix.shape}")

# Compute D1 and theta
quanti = 0.95
D1, theta = fun_dynsys_univariate_analysis(data_matrix, quanti)

# Create results folder
results_folder = "dsm_results"
os.makedirs(results_folder, exist_ok=True)
print("Results folder created")


# Save results as a NetCDF file
ds_results = xr.Dataset(
    {
        "D1": (("valid_time"), D1),
        "theta": (("valid_time"), theta),
    },
    coords={
        "valid_time": ds_coarse["valid_time"].values,
        "latitude": ds_coarse["latitude"].values,
        "longitude": ds_coarse["longitude"].values,
    },
    attrs={
        "description": "Dynamical systems analysis results",
        "geographic_region": "75N to 5N, 80W to 45E",
        "quantile_threshold": quanti,
    }
)

output_nc_file = os.path.join(results_folder, "dsm_results.nc")
ds_results.to_netcdf(output_nc_file)
print(f"Results saved to {output_nc_file}")

print(f"Starting the plots")

# Scatter plot: D1 vs Theta
plt.figure(figsize=(10, 6))
plt.scatter(D1, theta, c='blue', alpha=0.5)
plt.title("Scatter Plot: D1 vs Theta")
plt.xlabel("Local Dimension (D1)")
plt.ylabel("Extremal Index (Theta)")
plt.grid(True)
scatter_plot_file = os.path.join(results_folder, "scatter_d1_vs_theta.png")
plt.savefig(scatter_plot_file)
plt.show()
print(f"Scatter plot saved to {scatter_plot_file}")

##########################################################################################
                        ### Some additional Plots ###
##########################################################################################


# Visualization: Spatial distribution of D1 and Theta
latitude = ds_coarse['latitude'].values
longitude = ds_coarse['longitude'].values
lon, lat = np.meshgrid(longitude, latitude)

plt.figure(figsize=(12, 8))

# Local Dimensions (D1)
plt.subplot(2, 1, 1)
plt.scatter(lon, lat, c=D1, cmap='viridis', s=10)
plt.colorbar(label='Local Dimension (D1)')
plt.title(f'Local Dimensions, Average D1 = {np.nanmean(D1):.2f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Extremal Index (Theta)
plt.subplot(2, 1, 2)
plt.scatter(lon, lat, c=theta, cmap='plasma', s=10)
plt.colorbar(label='Extremal Index (Theta)')
plt.title(f'Extremal Index, Average Theta = {np.nanmean(theta):.2f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

results_plot_file = os.path.join(results_folder, "results_plot.png")
plt.tight_layout()
plt.savefig(results_plot_file)
plt.show()
print(f"Results plot saved to {results_plot_file}")