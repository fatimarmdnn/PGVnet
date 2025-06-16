import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
    
def prep_input_map(source_params, station_coords):
    """
    Prepares input features for a PGV (Peak Ground Velocity) map based on a sparse station grid.

    Parameters:
        source_coords (np.ndarray): Array of shape (N, 3), where N is the number of sources.
        strikes (np.ndarray): Array of shape (N,) with strike angles.
        dips (np.ndarray): Array of shape (N,) with dip angles.
        rakes (np.ndarray): Array of shape (N,) with rake angles.

    Returns:
        np.ndarray: Input tensor of shape (N, n_stations, 8), including:
                    [distance, azimuth, depth, strike, dip, rake, radiation, takeoff]
    """
    source_coords = source_params[:, :3]
    strikes       = source_params[:, 3] 
    dips          = source_params[:, 4]
    rakes         = source_params[:, 5] 
    
    n_stations = station_coords.shape[0]

    # Calculate distances and azimuths between sources and stations
    diffs      = source_coords[:, np.newaxis, :2] - station_coords[np.newaxis, :, :2]
    distances  = np.linalg.norm(diffs, axis=2)
    azimuths   = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])

    # Compute radiation pattern and takeoff angles
    radiation, takeoff = compute_radiation_takeoff(source_coords, station_coords, strikes, dips)

    # Broadcast source attributes to all stations
    depths   = np.repeat(source_coords[:, 2][:, np.newaxis], n_stations, axis=1)
    strikes  = np.repeat(strikes[:, np.newaxis], n_stations, axis=1)
    dips     = np.repeat(dips[:, np.newaxis], n_stations, axis=1)
    rakes    = np.repeat(rakes[:, np.newaxis], n_stations, axis=1)

    # Stack features into final input tensor
    inputs = np.stack([distances, azimuths, depths, strikes, dips, rakes, radiation, takeoff], axis=-1)

    return inputs


def compute_radiation_takeoff(sources, receivers, strikes, dips):
    """
    Computes the P-wave radiation pattern and takeoff angle between each source and receiver.

    Parameters:
        sources (np.ndarray): Array of shape (M, 3) with source coordinates (x, y, z).
        receivers (np.ndarray): Array of shape (N, 3) with receiver coordinates.
        strikes (np.ndarray): Array of shape (M,) with strike angles in degrees.
        dips (np.ndarray): Array of shape (M,) with dip angles in degrees.
        rakes (np.ndarray): Array of shape (M,) with rake angles in degrees.

    Returns:
        tuple:
            - rad_pattern (np.ndarray): Radiation pattern of shape (M, N).
            - takeoff (np.ndarray): Takeoff angles (angle from vertical) of shape (M, N).
    """
    # Ensure inputs are NumPy arrays
    sources   = np.asarray(sources)    # shape: (M, 3)
    receivers = np.asarray(receivers)  # shape: (N, 3)

    strikes_rad = np.radians(strikes).reshape(-1, 1)  # shape: (M, 1)
    dips_rad = np.radians(dips).reshape(-1, 1)

    # Compute source-to-receiver vectors: shape (M, N, 3)
    vec = receivers[np.newaxis, :, :] - sources[:, np.newaxis, :]

    # Normalize to unit vectors
    r = np.linalg.norm(vec, axis=2, keepdims=True)  # shape: (M, N, 1)
    unit_vec = vec / r
    ux, uy, uz = unit_vec[:, :, 0], unit_vec[:, :, 1], unit_vec[:, :, 2]

    # Compute takeoff angle: angle from vertical
    takeoff = np.arccos(uz)  # shape: (M, N)

    # Compute azimuth (angle from north, clockwise)
    dx, dy = vec[:, :, 0], vec[:, :, 1]
    azimuth = np.arctan2(dx, dy) % (2 * np.pi)  # shape: (M, N)

    # Compute phi = azimuth - strike, and use delta = dip
    phi = azimuth - strikes_rad  # shape: (M, N)
    delta = dips_rad  # shape: (M, 1), broadcasted

    # Simplified P-wave radiation pattern formula
    rad_pattern = (
        np.cos(takeoff) * np.sin(delta) * np.sin(2 * phi) +
        np.sin(takeoff) * np.cos(delta) * np.cos(2 * phi))

    return rad_pattern, takeoff



def train_xgb(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model with fixed hyperparameters
    model = XGBRegressor(
        max_depth=10,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        n_estimators=500,
        min_child_weight=5,
        gamma=0,
        objective='reg:pseudohubererror',
        random_state=42
    )

    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # Evaluation
    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test, y_test_pred)

    print(f"Training R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")

    mae_train = np.mean(np.abs(y_train - y_train_pred))
    mse_train = np.mean((y_train - y_train_pred)**2)

    mae_test  = np.mean(np.abs(y_test - y_test_pred))
    mse_test  = np.mean((y_test - y_test_pred)**2)

    metrics = {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "mse_train": mse_train,
        "mae_test": mae_test,
        "mse_test": mse_test,
    }

         
    return y_test, y_test_pred
         