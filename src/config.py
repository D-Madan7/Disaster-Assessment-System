"""
Global configuration for the Disaster Assessment System.
"""

import os
import torch
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
MODELS_DIR = "models"

### SIAMESE
IMAGE_SIZE_SIAMESE = 512
DILATION_KERNEL_SIZE = 5
SIAMESE_WEIGHTS_PATH = os.path.join(MODELS_DIR,"best_siamese_model.pth")

### CNN
IMAGE_SIZE_CNN = 224
PADDING_RATIO = 0.10
CLASS_NAMES = ["no-damage", "damaged", "destroyed",]
CLASS_TO_IDX = {"no-damage": 0, "damaged": 1, "destroyed": 2,}
MERGE_MAPPING = {"no-damage": "no-damage",  "minor-damage": "damaged", "major-damage": "damaged", "destroyed": "destroyed",}
IGNORED_CLASSES = {"un-classified"}
CLASSIFIER_WEIGHTS_PATH = os.path.join( MODELS_DIR,"best_classifier_model.pth")

### RAG
USE_TRUE_LABELS_FOR_NOTEBOOK_ANALYSIS = True
ENABLE_RAG_RETRIEVAL = True
TOP_K_DOC_CHUNKS = 4
MASK_THRESHOLD = 127
GRID_ROWS = 3
GRID_COLS = 3

CLASS_ORDER = ["no-damage", "damaged", "destroyed"]
CLASS_COLORS = {"no-damage": "lime", "damaged": "orange", "destroyed": "red",}

DOC_SUBFOLDERS = ["Building Assessment", "Earthquake", "Flood_Hurricane",
                  "Tornado", "Tsunami", "Volcano", "Wildfire",]

EVENT_TO_DOC_FOLDER = {
    "portugal-wildfire": "Wildfire",
    "pinery-bushfire": "Wildfire",
    "socal-fire": "Wildfire",
    "woolsey-fire": "Wildfire",
    "santa-rosa-wildfire": "Wildfire",
    "nepal-flooding": "Flood_Hurricane",
    "midwest-flooding": "Flood_Hurricane",
    "hurricane-michael": "Flood_Hurricane",
    "hurricane-florence": "Flood_Hurricane",
    "hurricane-harvey": "Flood_Hurricane",
    "hurricane-matthew": "Flood_Hurricane",
    "tuscaloosa-tornado": "Tornado",
    "moore-tornado": "Tornado",
    "joplin-tornado": "Tornado",
    "lower-puna-volcano": "Volcano",
    "guatemala-volcano": "Volcano",
    "mexico-earthquake": "Earthquake",
    "palu-tsunami": "Tsunami",
    "sunda-tsunami": "Tsunami",
}

ACTION_KEYWORDS = [
    "inspect",
    "inspection",
    "assess",
    "assessment",
    "evaluate",
    "repair",
    "stabilize",
    "monitor",
    "evacuate",
    "rescue",
    "restrict",
    "re-entry",
    "occupancy",
    "unsafe",
    "structural",
    "building",
    "damage",
    "debris",
    "utility",
    "utilities",
    "lifeline",
    "critical infrastructure",
    "temporary shelter",
    "recovery",
    "emergency",
    "response",
    "field inspection",
]

BAD_PATTERNS = [
    "chapter",
    "figure",
    "table",
    "appendix",
    "isbn",
    "contents",
    "copyright",
    "acknowledgement",
    "acknowledgment",
    "references",
    "bibliography",
    "index",
    "page ",
    "section ",
    "manual",
    "field guide",
    "publication",
    "doi",
    "http",
    "https",
    "www.",
    "fema p-",
    "atc-",
    "mbie",
    "available at",
]

IMPORTANT_TERMS = [
    "collapsed",
    "collapse",
    "destroyed",
    "damaged",
    "structural damage",
    "unsafe",
    "inspection",
    "re-entry",
    "evacuation",
    "debris removal",
    "search and rescue",
    "critical facilities",
    "utility restoration",
    "rapid assessment",
]

ACTION_PATTERNS = [
    "inspect",
    "inspection",
    "assess",
    "assessment",
    "evaluate",
    "repair",
    "stabilize",
    "restrict",
    "evacuate",
    "monitor",
    "rescue",
    "occupancy",
    "re-entry",
    "unsafe",
    "debris",
    "utility",
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_DIRECTORY = Path("rag")
DOC_INDICES_PATH = RAG_DIRECTORY / "doc_indices.pkl"