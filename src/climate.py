import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
from src.config import CLIMATE_OUTPUT_DIR, ERA5_PRECIP_DIR, ERA5_TEMP_DIR
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_grid_info(sample_path):
    """Extracts 1D latitude and longitude arrays from an NC file."""
    # Force the netcdf4 engine
    with xr.open_dataset(sample_path, engine='netcdf4') as ds:
        lat_name = next((v for v in ['latitude', 'lat'] if v in ds.coords), None)
        lon_name = next((v for v in ['longitude', 'lon'] if v in ds.coords), None)

        if not lat_name or not lon_name:
            raise ValueError(f"Could not find lat/lon in {sample_path}")

        lats = ds[lat_name].values
        lons = ds[lon_name].values
        
    return lats, lons

def compute_spatial_weights(basins_gdf, lats, lons):
    """Computes intersection weights with CRS and Longitude checks."""
    print(f"⏳ Computing spatial weights for {len(basins_gdf)} basins...")
    
    # ERA5 is often 0-360. Convert to -180 to 180 if needed.
    lons_matching = lons.copy()
    if np.any(lons_matching > 180):
        lons_matching[lons_matching > 180] -= 360

    weights_lookup = {}
    lat_res = abs(lats[1] - lats[0])
    lon_res = abs(lons[1] - lons[0])
    empty_basins = 0

    for _, row in tqdm(basins_gdf.iterrows(), total=len(basins_gdf), desc="Mapping Grid"):
        station_id = row['station_id']
        basin_geom = row.geometry
        minx, miny, maxx, maxy = basin_geom.bounds
        
        # Candidate indices
        lat_mask = (lats >= miny - lat_res) & (lats <= maxy + lat_res)
        lon_mask = (lons_matching >= minx - lon_res) & (lons_matching <= maxx + lon_res)
        
        valid_lat_indices = np.where(lat_mask)[0]
        valid_lon_indices = np.where(lon_mask)[0]
        
        basin_weights = {}
        
        for i in valid_lat_indices:
            for j in valid_lon_indices:
                y_c, x_c = lats[i], lons_matching[j]
                cell = box(x_c - lon_res/2, y_c - lat_res/2, x_c + lon_res/2, y_c + lat_res/2)
                
                if basin_geom.intersects(cell):
                    inter_area = basin_geom.intersection(cell).area
                    if inter_area > 0:
                        basin_weights[(i, j)] = inter_area / cell.area

        if basin_weights:
            total_w = sum(basin_weights.values())
            weights_lookup[station_id] = {k: v/total_w for k, v in basin_weights.items()}
        else:
            empty_basins += 1

    if empty_basins > 0:
        print(f"⚠️ Warning: {empty_basins} basins failed intersection.")
    
    return weights_lookup

def _read_hourly(files, names):
    """Read UTC hours in order, rejecting gaps and overlapping files."""
    previous = None
    for path in sorted(files):
        with xr.open_dataset(path, engine="netcdf4") as ds:
            name = next((name for name in names if name in ds.data_vars), None)
            if name is None:
                raise ValueError(f"No expected variable {names} in {path}")
            coord = "valid_time" if "valid_time" in ds.coords else "time"
            times = pd.DatetimeIndex(pd.to_datetime(ds[coord].values))
            data = ds[name].values
        if len(times) != len(data) or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError(f"Invalid hourly time axis in {path}")
        if len(times) and (not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all()
                           or (previous is not None and times[0] - previous != pd.Timedelta(hours=1))):
            raise ValueError(f"Missing or overlapping hours before/in {path}")
        if len(times):
            previous = times[-1]
            yield times, data


def _increments(times, cumulative, previous_time, previous_value):
    """Convert ERA5-Land forecast accumulations (metres) to hourly amounts.

    01 UTC is the first forecast step; 00 UTC is step 24 of the previous
    forecast. The timestamp denotes the *end* of the accumulation interval.
    """
    result = np.full(cumulative.shape, np.nan, dtype=float)
    for i, time in enumerate(times):
        if time.hour == 1:
            result[i] = cumulative[i]
        elif previous_time is not None and time - previous_time == pd.Timedelta(hours=1):
            result[i] = cumulative[i] - previous_value
        previous_time, previous_value = time, cumulative[i].copy()
    # Small negative differences can arise from GRIB packing precision.
    result = np.maximum(result, 0)
    return result, previous_time, previous_value


def _weighted(data, weights_map):
    out = {}
    for station, weights in weights_map.items():
        lat, lon = zip(*weights)
        w = np.asarray(list(weights.values()))
        pixels = data[:, lat, lon]
        valid = np.isfinite(pixels)
        denominator = (valid * w).sum(axis=1)
        out[station] = np.divide(np.nansum(pixels * w, axis=1), denominator,
                                 out=np.full(len(data), np.nan), where=denominator > 0)
    return pd.DataFrame(out)


def _daily(frames, operation):
    """Combine hours before grouping; retain only complete 24-hour values."""
    hourly = pd.concat(frames).sort_index()
    grouped = hourly.resample("D")
    counts = grouped.count()
    daily = getattr(grouped, operation)()
    return daily.where(counts == 24)


def process_era5_files(files, weights_map, var_type):
    """Produce fixed UTC-7 daily basin averages from ERA5-Land UTC hours."""
    if var_type not in ("precip", "temp"):
        raise ValueError(f"Unknown variable type: {var_type}")
    names = ["tp", "total_precipitation", "precip"] if var_type == "precip" else ["t2m", "2t", "2m_temperature"]
    frames = []
    previous_time = previous_value = None
    for times, data in tqdm(_read_hourly(files, names), desc=f"{var_type} files"):
        if var_type == "precip":
            data, previous_time, previous_value = _increments(times, data, previous_time, previous_value)
        frame = _weighted(data, weights_map)
        # Precipitation belongs to the hour ending at its valid time; temperature
        # is instantaneous and belongs to the hour beginning at its valid time.
        frame.index = times - pd.Timedelta(hours=7 + (var_type == "precip"))
        frames.append(frame)
    if not frames:
        raise ValueError(f"No {var_type} files supplied")
    if var_type == "precip":
        return _daily(frames, "sum") * 1000
    return _daily(frames, "min") - 273.15, _daily(frames, "max") - 273.15


def process_snow_rain_files(precip_files, temp_files, weights_map):
    """Split hourly precipitation using temperature at the interval end."""
    precip = _read_hourly(precip_files, ["tp", "total_precipitation", "precip"])
    temp = _read_hourly(temp_files, ["t2m", "2t", "2m_temperature"])
    from itertools import zip_longest
    snow_frames, rain_frames, frac_frames = [], [], []
    previous_time = previous_value = None
    for p_chunk, t_chunk in tqdm(zip_longest(precip, temp), desc="Snow/rain files"):
        if p_chunk is None or t_chunk is None:
            raise ValueError("Precipitation and temperature file coverage differs")
        times, cumulative = p_chunk
        t_times, temperature = t_chunk
        if not times.equals(t_times) or cumulative.shape != temperature.shape:
            raise ValueError("Precipitation and temperature time axes or grids differ")
        amounts, previous_time, previous_value = _increments(times, cumulative, previous_time, previous_value)
        valid = np.isfinite(amounts) & np.isfinite(temperature)
        snow = np.where(valid, np.where(temperature < 273.15, amounts, 0), np.nan)
        rain = np.where(valid, np.where(temperature >= 273.15, amounts, 0), np.nan)
        fraction = np.where(np.isfinite(temperature), (temperature < 273.15).astype(float), np.nan)
        for data, frames, offset in ((snow, snow_frames, 8), (rain, rain_frames, 8), (fraction, frac_frames, 7)):
            frame = _weighted(data, weights_map)
            frame.index = times - pd.Timedelta(hours=offset)
            frames.append(frame)
    if not snow_frames:
        raise ValueError("No precipitation and temperature files supplied")
    return _daily(snow_frames, "sum") * 1000, _daily(rain_frames, "sum") * 1000, _daily(frac_frames, "mean")


def process_era5_basin_data(basin_gpkg_list, stations_list, split_snow=True):
    """Main Orchestrator"""
    
    # 1. Load Basins
    print("Step 1/5: Loading Basins...")
    gdfs = []
    for p in basin_gpkg_list:
        gdf = gpd.read_file(p, layer='DrainageBasin_BassinDeDrainage')
        id_col = next(c for c in gdf.columns if c.lower() in ['stationnum', 'id', 'station_id'])
        gdf = gdf.rename(columns={id_col: 'station_id'})
        gdfs.append(gdf)
    
    full_gdf = pd.concat(gdfs)
    full_gdf['station_id'] = full_gdf['station_id'].str.strip()
    filtered_gdf = full_gdf[full_gdf['station_id'].isin(stations_list)].copy()
    
    if filtered_gdf.crs != "EPSG:4326":
        print("   🔄 Reprojecting basins to EPSG:4326 (Lat/Lon)...")
        filtered_gdf = filtered_gdf.to_crs("EPSG:4326")
    
    # 2. Map Weights
    print("Step 2/5: Mapping Spatial Weights...")
    precip_files = sorted(list(ERA5_PRECIP_DIR.glob("*.nc")))
    temp_files = sorted(list(ERA5_TEMP_DIR.glob("*.nc")))
    
    if not precip_files: 
        raise FileNotFoundError("No ERA5 Precip files found")
    
    lats, lons = get_grid_info(precip_files[0])
    weights = compute_spatial_weights(filtered_gdf, lats, lons)
    
    if not weights:
        raise ValueError("Weights Dictionary is empty.")
    
    # 3. Process Total Precip
    print("\nStep 3/5: Processing Total Precipitation...")
    df_precip = process_era5_files(precip_files, weights, var_type='precip')
    df_precip.to_csv(CLIMATE_OUTPUT_DIR / "daily_precipitation.csv")
    print("✅ Total Precipitation data saved.")
    
    # 4. Process Temp
    print("\nStep 4/5: Processing Temperature...")
    if temp_files:
        df_min, df_max = process_era5_files(temp_files, weights, var_type='temp')
        df_min.to_csv(CLIMATE_OUTPUT_DIR / "daily_temp_min.csv")
        df_max.to_csv(CLIMATE_OUTPUT_DIR / "daily_temp_max.csv")
        print("✅ Temperature data saved.")
        
    # 5. Split Snow, Rain, and Fraction Below Zero
    if split_snow and precip_files and temp_files:
        print("\nStep 5/5: Splitting Precipitation into Rain/Snow and computing Freezing Fraction...")
        if len(precip_files) != len(temp_files):
            print("⚠️ Warning: Number of Precip and Temp files do not match. Ensure time coverage is identical.")
            
        df_snow, df_rain, df_frac = process_snow_rain_files(precip_files, temp_files, weights)
        df_snow.to_csv(CLIMATE_OUTPUT_DIR / "daily_snowfall.csv")
        df_rain.to_csv(CLIMATE_OUTPUT_DIR / "daily_rainfall.csv")
        df_frac.to_csv(CLIMATE_OUTPUT_DIR / "daily_fraction_below_zero.csv")
        print("✅ Snowfall, Rainfall, and Fraction Below Zero data saved.")
        
    print("\n🎉 Climate processing complete.")
