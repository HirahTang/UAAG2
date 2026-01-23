from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import shutil
import tempfile
import hydra
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

# Add project root to sys.path to ensure imports work
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from uaag2.equivariant_diffusion import Trainer
from uaag2.datasets.uaag_dataset import Dataset_Info, UAAG2Dataset_sampling

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Global variables for model and config
MODEL = None
DATASET_INFO = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants (should match training/eval or be loaded from config)
# Ideally these should come from environment variables or a specific deployment config
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_INFO_PATH = os.path.join(_BASE_DIR, "..", "..", "data", "statistic.pkl")
_DEFAULT_CHECKPOINT_PATH = os.path.join(_BASE_DIR, "..", "..", "models", "good_model", "last.ckpt")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", _DEFAULT_CHECKPOINT_PATH)
DATA_INFO_PATH = os.environ.get("DATA_INFO_PATH", _DEFAULT_DATA_INFO_PATH)
CONFIG_PATH = os.path.join(_BASE_DIR, "..", "..", "configs")
CONFIG_NAME = "train"


def load_config():
    with hydra.initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        # Override save_dir specifically for the API to avoid hydra interpolation error
        # We also override test_save_dir as it often refers to save_dir
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=["save_dir=.", "test_save_dir=."])
        OmegaConf.set_struct(cfg, False)  # Globally disable struct mode for the API
    return cfg


def load_data_info(cfg):
    global DATASET_INFO
    if os.path.exists(DATA_INFO_PATH):
        # Override data info path if provided in env
        if "data" not in cfg:
            cfg.data = {}
        cfg.data.data_info_path = DATA_INFO_PATH

        DATASET_INFO = Dataset_Info(cfg, DATA_INFO_PATH)
        return cfg
    else:
        # Fallback if specific file not found, use config's path if available
        if os.path.exists(cfg.data.data_info_path):
            DATASET_INFO = Dataset_Info(cfg, cfg.data.data_info_path)
            return cfg
        raise FileNotFoundError(f"{DATA_INFO_PATH} not found and config path {cfg.data.data_info_path} also missing")


@app.on_event("startup")
async def startup_event():
    global MODEL
    # Allow safe globals for torch.load
    if hasattr(torch.serialization, "add_safe_globals"):
        pass

    try:
        cfg = load_config()
        cfg = load_data_info(cfg)

        print(f"Loading model from {CHECKPOINT_PATH}...")

        if os.path.exists(CHECKPOINT_PATH):
            MODEL = Trainer.load_from_checkpoint(
                CHECKPOINT_PATH,
                hparams=cfg,
                dataset_info=DATASET_INFO,
                strict=False,  # To avoid errors with missing keys if any
            ).to(DEVICE)
            OmegaConf.set_struct(MODEL.hparams, False)
            MODEL.eval()
            print("Model loaded successfully.")
        else:
            print(f"Checkpoint not found at {CHECKPOINT_PATH}. API will start but generation will fail.")

    except Exception as e:
        print(f"Failed to load model: {e}")
        # We don't raise here to allow app to start, but generation will fail
        pass


@app.post("/generate")
def generate(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
        content = file.file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    req_cfg_save_dir = None
    try:
        # Load the data
        try:
            data_list = torch.load(temp_path, weights_only=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid .pt file: {str(e)}")

        if isinstance(data_list, list):
            if len(data_list) == 0:
                raise HTTPException(status_code=400, detail="Empty data list")
            graph = data_list[0]  # Take first
        else:
            graph = data_list  # Assume it's a single Data object

        # Create dataset and loader
        # We use a fresh config or modification of the global one
        # For generation we need specific params that might differ from training

        # Load defaults from global if available, or load fresh
        # But we need to ensure unique ID/save_dir for this request

        # Clone cfg? Trainer has .hparams which is the config
        # But we shouldn't modify the model's hparams in place if it affects others

        # Create a temporary config context
        req_cfg = MODEL.hparams.copy()
        req_cfg.save_dir = tempfile.mkdtemp()
        req_cfg_save_dir = req_cfg.save_dir
        req_cfg.id = "api_gen"
        req_cfg.virtual_node_size = 15  # Default for gen
        req_cfg.num_samples = 1  # Default for gen

        dataset_save_path = os.path.join(req_cfg.save_dir, "samples")

        dataset = UAAG2Dataset_sampling(
            graph,
            req_cfg,
            dataset_save_path,
            DATASET_INFO,
            sample_size=req_cfg.virtual_node_size,
            sample_length=req_cfg.num_samples,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )

        # To be safe(r), let's use absolute path for save_path argument if generate_ligand supports it?
        # Or just risk it for this migration task (cleanup).

        # Trainer.generate_ligand uses self.save_dir.
        # But os.path.join(prefix, /absolute/path) returns /absolute/path.
        # So passing an absolute path to generate_ligand bypasses self.save_dir.

        # Create a unique output directory for this request
        req_save_dir = os.path.join(req_cfg.save_dir, f"run{req_cfg.id}")
        os.makedirs(req_save_dir, exist_ok=True)

        # We pass the absolute path req_save_dir as save_path.
        # This causes generate_ligand to write to req_save_dir/iter_i
        # ignoring MODEL.save_dir
        MODEL.generate_ligand(dataloader, save_path=req_save_dir, verbose=True, device=DEVICE)

        # Find the generated file
        # generate_ligand joins save_path with iter_i, so we look in req_save_dir
        base_output_dir = req_save_dir

        # Collect results
        results = []

        # Iterate over iterations (samples)
        for i in range(req_cfg.num_samples):
            mol_path = os.path.join(base_output_dir, f"iter_{i}", "batch_0", "final", "ligand.mol")
            xyz_path = os.path.join(base_output_dir, f"iter_{i}", "batch_0", "final", "ligand.xyz")

            if os.path.exists(mol_path):
                with open(mol_path, "r") as f:
                    mol_content = f.read()
                results.append({"format": "mol", "content": mol_content, "id": i})
            elif os.path.exists(xyz_path):
                with open(xyz_path, "r") as f:
                    xyz_content = f.read()
                results.append({"format": "xyz", "content": xyz_content, "id": i})

        if not results:
            raise HTTPException(status_code=500, detail="Generation failed to produce output files")

        return JSONResponse(content={"results": results})

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)
        if req_cfg_save_dir and os.path.exists(req_cfg_save_dir):
            shutil.rmtree(req_cfg_save_dir)


# Mount static files
app.mount("/", StaticFiles(directory="src/uaag2/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
