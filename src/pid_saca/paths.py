"""Repository-relative paths used by scripts and modules."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "log"
PARAM_DIR = REPO_ROOT / "param"
FIGURE_DIR = REPO_ROOT / "figure"

STATION_FILE = DATA_DIR / "stations_30" / "bus_stations.txt"
ROUTING_MODEL_DIR = PACKAGE_ROOT / "routing" / "model"
