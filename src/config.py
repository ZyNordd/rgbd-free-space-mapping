from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"
POINT_CLOUDS_DIR = OUTPUTS_DIR / "point_clouds"
OCCUPANCY_MAPS_DIR = OUTPUTS_DIR / "occupancy_maps"
METRICS_DIR = OUTPUTS_DIR / "metrics"


# Depth values in the Kaggle NYU2 dataset appear to be stored in millimeters.
DEPTH_SCALE = 1000.0

MIN_DEPTH_M = 0.5
MAX_DEPTH_M = 5.0


# Approximate NYU Depth V2 camera intrinsics.
# These values are commonly used for 640x480 NYU Depth V2 frames.
FX = 5.1885790117450188e02
FY = 5.1946961112127485e02
CX = 3.2558244941119034e02
CY = 2.5373616633400465e02


# Point cloud settings
DOWNSAMPLE_STEP = 2


# Occupancy grid settings
GRID_RESOLUTION_M = 0.10

# Obstacle settings
MIN_OBSTACLE_HEIGHT_M = 0.25
MAX_OBSTACLE_HEIGHT_M = 1.80

# Robot settings
ROBOT_WIDTH_M = 0.30
ROBOT_RADIUS_M = ROBOT_WIDTH_M / 2.0