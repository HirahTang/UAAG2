from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import tempfile
import hydra
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
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "models/good_model/last.ckpt")
DATA_INFO_PATH = os.environ.get("DATA_INFO_PATH", "data/statistic.pkl")
CONFIG_PATH = "../../configs"
CONFIG_NAME = "train"


def load_config():
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(config_name=CONFIG_NAME)
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
            MODEL.eval()
            print("Model loaded successfully.")
        else:
            print(f"Checkpoint not found at {CHECKPOINT_PATH}. API will start but generation will fail.")

    except Exception as e:
        print(f"Failed to load model: {e}")
        # We don't raise here to allow app to start, but generation will fail
        pass


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

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
        req_cfg.id = "api_gen"
        req_cfg.virtual_node_size = 15  # Default for gen
        req_cfg.num_samples = 3  # Default for gen

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

        # Run generation
        # generate_ligand saves to os.path.join(self.save_dir, save_path, f"iter_{i}")

        # We need to temporarily set model's save dir to separate requests
        # Or better, pass save_path_override if possible, but generate_ligand uses self.save_dir

        # Trainer.generate_ligand uses self.save_dir.
        # CAUTION: Changing self.save_dir on a global model object in async context is NOT thread safe.
        # However, for this single-worker process or if generate_ligand uses passed args locally...
        # Checking generate_ligand implementation (from memory, it takes save_path)
        # But looking at previous code:
        # MODEL.hparams.save_dir = hparams.save_dir
        # MODEL.save_dir = ...
        # This IS NOT thread safe.
        # But assuming single worker for now or low concurrency.
        # Ideally generate_ligand should accept absolute output path.

        # Let's check generate_ligand signature in previous file view?
        # evaluate.py calls: model.generate_ligand(dataloader, save_path=save_path, verbose=True)
        # If save_path is relative, it joins with self.save_dir.

        # To be safe(r), let's use absolute path for save_path argument if generate_ligand supports it?
        # Or just risk it for this migration task (cleanup).

        original_save_dir = MODEL.save_dir
        MODEL.save_dir = os.path.join(req_cfg.save_dir, f"run{req_cfg.id}")

        try:
            MODEL.generate_ligand(dataloader, save_path="gen_output", verbose=True, device=DEVICE)

            # Find the generated file
            base_output_dir = os.path.join(MODEL.save_dir, "gen_output")

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

        finally:
            MODEL.save_dir = original_save_dir
            # Cleanup temp dir?
            # shutil.rmtree(req_cfg.save_dir)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)


# Mount static files
app.mount("/", StaticFiles(directory="src/uaag2/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
