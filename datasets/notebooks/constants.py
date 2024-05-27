import os

# Define paths to data directories
DATA_DIR = "../data"

# Define paths to data files

# OSM Caching paths
OSM_DIR = os.path.join(DATA_DIR, "OSM")

FULL_ROAD_NETWORK_PATH = os.path.join(OSM_DIR, "vejnet.graphml")
SIMPLE_ROAD_NETWORK_PATH = os.path.join(OSM_DIR, "vejnet_simple.graphml")

NODE_GDF_PATH = os.path.join(OSM_DIR, "road_nodes.geojson")
EDGE_GDF_PATH = os.path.join(OSM_DIR, "road_edges.geojson")

# Vejman
VEJMAN_PATH = os.path.join(DATA_DIR, "vejman.geojson")
VEJMAN_DIR = os.path.join(DATA_DIR, "vejman")
VEJMAN_FILES = {
    2019: os.path.join(VEJMAN_DIR, "Vejman-dk (3).xls"),
    2020: os.path.join(VEJMAN_DIR, "Vejman-dk.xls"),
    2021: os.path.join(VEJMAN_DIR, "Vejman-dk (1).xls"),
    2022: os.path.join(VEJMAN_DIR, "Vejman-dk (2).xls")
}

# Mastra
MASTRA_PATH = os.path.join(DATA_DIR, "mastra.geojson")
MASTRA_DIR = os.path.join(DATA_DIR, "mastra")
NODE_CSV_PATH = os.path.join(DATA_DIR, "aadt_nodes.csv")

# Finished datasets
TABULAR_INTERSECTIONS = os.path.join(DATA_DIR, "tabular_intersections.geojson")

GRAPH_DATASET_DIR = os.path.join(DATA_DIR, "graph_datasets")

# Bounding boxes
DK_BBOX = (8.08997684086, 54.8000145534, 12.6900061378, 57.730016588)