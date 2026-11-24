import cdsapi

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["2m_dewpoint_temperature", "2m_temperature", "total_precipitation"],
    "year": [str(y) for y in range(1940, 2026)],
    "month": ["01","02","03","04","05","06","07","08","09","10","11","12"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [38, 68, 6, 98]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("data/raw/era5_india.nc")
print("Done! Saved to data/raw/era5_india.nc")
