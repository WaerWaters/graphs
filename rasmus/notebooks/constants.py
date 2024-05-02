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

EDGE_CHUNKS_DIR = os.path.join(OSM_DIR, "edges")

FULL_NETWORK_CHUNKS_DIR = os.path.join(OSM_DIR, "graphs")


# Vejman
VEJMAN_PATH = os.path.join(DATA_DIR, "vejman.geojson")
VEJMAN_DIR = os.path.join(DATA_DIR, "vejman")
VEJMAN_FILES = {
    2019: "Vejman-dk (3).xls",
    2020: "Vejman-dk.xls",
    2021: "Vejman-dk (1).xls",
    2022: "Vejman-dk (2).xls"
}

# Mastra
MASTRA_PATH = os.path.join(DATA_DIR, "mastra.geojson")
MASTRA_DIR = os.path.join(DATA_DIR, "mastra")

# CVF
CVF_PATH = os.path.join(DATA_DIR, "cvf.xml")

# Finished datasets
TABULAR_INTERSECTIONS = os.path.join(DATA_DIR, "tabular_intersections.geojson")

# Bounding boxes
DK_BBOX = (8.08997684086, 54.8000145534, 12.6900061378, 57.730016588)