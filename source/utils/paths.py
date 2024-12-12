from pathlib import Path


REPOSITORY_DIR = Path(__file__).parents[2].resolve()

CREDENTIALS_DIR =  REPOSITORY_DIR / 'credentials'
CREDENTIALS_PATH =  CREDENTIALS_DIR / 'credentials.yaml'

SOURCE_DIR = REPOSITORY_DIR / "source"

ASSETS_DIR = REPOSITORY_DIR / "assets"

MODELS_DIR = ASSETS_DIR / "models"
EMBEDDER_MODELS_DIR = MODELS_DIR / "embedders"
GENERATOR_MODELS_DIR = MODELS_DIR / "generators"

DB_DIR = ASSETS_DIR / "db"
CHROMA_DB_DIR = DB_DIR / "chroma"

DOCUMENTS_DIR = ASSETS_DIR / "documents"
VADE_MECUM_PATH = DOCUMENTS_DIR / "2023_vade_mecum.pdf"

