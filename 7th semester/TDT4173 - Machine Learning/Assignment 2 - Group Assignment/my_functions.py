import numpy as np
import pandas as pd

def full_clean(Y_train, X_train, X_test, location, index_date = False, normalize = False, sqrt = False):
    
    ignore_features = [
        "ceiling_height_agl__m",
        "cloud_base_agl__m",
        "snow_density__kgm3",
    ]
    
    categorical_features = [
        "dew_or_rime__idx",
        "elevation__m",
        "is_day__idx",
        "is_in_shadow__idx",
        "precip_type_5min__idx",
        # "snow_drift__idx", Almost no variation
        "wind_speed_w_1000hPa__ms"
    ]
    
    numeric_features = [
        "absolute_humidity_2m__gm3",
        "air_density_2m__kgm3",
        # "ceiling_height_agl__m", Fucked up
        "clear_sky_energy_1h__J",
        "clear_sky_rad__W",
        # "cloud_base_agl__m", Fucked up
        "dew_point_2m__K",
        "diffuse_rad__W",
        "diffuse_rad_1h__J",
        "direct_rad__W",
        "direct_rad_1h__J",
        "effective_cloud_cover__p",
        "fresh_snow_12h__cm",
        "fresh_snow_1h__cm",
        "fresh_snow_24h__cm",
        "fresh_snow_3h__cm",
        "fresh_snow_6h__cm",
        "msl_pressure__hPa",
        "precip_5min__mm",
        "pressure_100m__hPa",
        "pressure_50m__hPa",
        "prob_rime__p",
        "rain_water__kgm2",
        "relative_humidity_1000hPa__p",
        "sfc_pressure__hPa",
        # "snow_density__kgm3", Fucked up
        "snow_depth__cm",
        "snow_melt_10min__mm",
        "snow_water__kgm2",
        "sun_azimuth__d",
        "sun_elevation__d",
        "super_cooled_liquid_water__kgm2",
        "t_1000hPa__K",
        "total_cloud_cover__p",
        "visibility__m",
        "wind_speed_10m__ms",
        "wind_speed_u_10m__ms",
        "wind_speed_v_10m__ms",
    ]
    
    similar_features = {
        "clear_sky": ["clear_sky_energy_1h__J","clear_sky_rad__W"],
        "dew_point_2m__K_and_t_1000hPa__K": ["dew_point_2m__K", "t_1000hPa__K"],
        "diffuse_rad": ["diffuse_rad__W", "diffuse_rad_1h__J"],
        "direct_rad": ["direct_rad__W", "direct_rad_1h__J"],
        "fresh_snow__cm": ["fresh_snow_1h__cm", "fresh_snow_3h__cm", "fresh_snow_6h__cm", "fresh_snow_12h__cm", "fresh_snow_24h__cm"],
        "pressure__hPa": ["msl_pressure__hPa", "pressure_100m__hPa", "pressure_50m__hPa", "sfc_pressure__hPa"]
    }
    
    # Setting DatetimeIndex:
    Y_train = Y_train.set_index("time")
    if sqrt:
        Y_train = np.sqrt(Y_train)
    
    # Removing NaN values from the Y training data
    Y_train = Y_train.dropna()

    # Setting DatetimeIndex:
    X_train = X_train.set_index("date_forecast")
    # Changing index name
    X_train.index.name = "time"
    # Removing ":" from all column names and replacing with "__":
    X_train.columns = X_train.columns.str.replace(":", "__")
    
    X_train_cat = X_train[categorical_features]
    X_train_cont = X_train[numeric_features]
    for key, value in similar_features.items():
        # X_train_cont[key] = X_train_cont[value].sum(axis = 1)
        X_train_cont.drop(value[1:], axis = 1, inplace = True)
    
    X_train_cat = round(X_train_cat.resample("H").mean(),0).astype("category")
    X_train_cont = (X_train_cont.resample("H").mean()).astype("float")
    
    if normalize:
        X_train_cont = X_train_cont/(X_train_cont.max())
    
    X_train = pd.concat([X_train_cat, X_train_cont], axis = 1)
    
    X_train = X_train.dropna(axis = 0, how = 'all')
    X_train["location"] = location
    X_train["location"] = X_train["location"].astype("category")

    # Setting DatetimeIndex:
    X_test = X_test.set_index("date_forecast")
    # Changing index name
    X_test.index.name = "time"
    # Removing ":" from all column names and replacing with "__":
    X_test.columns = X_test.columns.str.replace(":", "__")
    
    X_test_cat = X_test[categorical_features]
    X_test_cont = X_test[numeric_features]
    for key, value in similar_features.items():
        # X_test_cont[key] = X_test_cont[value].sum(axis = 1)
        X_test_cont.drop(value[1:], axis = 1, inplace = True)
    
    X_test_cat = round(X_test_cat.resample("H").mean(),0).astype("category")
    X_test_cont = (X_test_cont.resample("H").mean()).astype("float")
    
    if normalize:
        X_test_cont = X_test_cont/(X_test_cont.max())
    
    X_test = pd.concat([X_test_cat, X_test_cont], axis = 1)
    
    X_test = X_test.dropna(axis = 0, how = 'all')
    X_test["location"] = location
    X_test["location"] = X_test["location"].astype("category")
    
    Y_train, X_train = Y_train.align(X_train, join='inner', axis=0)
    
    X = pd.concat([X_train, X_test])
    
    train = pd.concat([Y_train, X_train], axis = 1)
    
    if index_date:
        return Y_train, X_train, X_test, X, train
        
    return Y_train.reset_index(), X_train.reset_index(), X_test.reset_index(), X.reset_index(), train.reset_index()


def export_csv(model_pred, name):
    df = pd.DataFrame(model_pred, columns = ["prediction"])
    df.index.name = "id"
    df.to_csv(f"Predictions/{name}.csv")

def export_csv2(model_pred, name):
    df = pd.DataFrame(model_pred, columns = ["prediction"])
    df.index.name = "id"
    df.to_csv(f"Predictions2/{name}.csv")