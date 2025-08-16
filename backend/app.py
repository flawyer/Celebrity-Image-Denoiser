from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import logging
import matplotlib.pyplot as plt
import base64
from typing import Optional, Dict, Tuple

# ---- Optional TensorFlow/Keras (for cGAN KERAS backend) ----
try:
    import tensorflow as tf  # pip install tensorflow
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gan_api")

# ------------------- FastAPI -------------------
app = FastAPI(title="Unified GAN API (Denoise / cGAN / SRGAN / ESRGAN)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ------------------- Device -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ------------------- PyTorch Model Definitions -------------------

# ---- Denoise Generator (U-Net like structure) ----
class DenoiseGenerator(nn.Module):
    def __init__(self):
        super(DenoiseGenerator, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        e1 = self.down1(x)
        p1 = self.pool1(e1)

        e2 = self.down2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        d2 = self.up2(b)
        if d2.size() != e2.size():
            _, _, h_d2, w_d2 = d2.shape
            e2 = e2[:, :, :h_d2, :w_d2]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.upconv2(d2)

        d1 = self.up1(d2)
        if d1.size() != e1.size():
            _, _, h_d1, w_d1 = d1.shape
            e1 = e1[:, :, :h_d1, :w_d1]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.upconv1(d1)

        return torch.tanh(d1)

# ---- cGAN Generator ----
class CGANGenerator(nn.Module):
    def __init__(self, n_classes: int = 10, latent_dim: int = 100):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.init_size = 8
        self.l1 = nn.Linear(latent_dim + latent_dim, 128 * self.init_size * self.init_size)
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, cond=None):
        if cond is None:
            raise ValueError("cGAN requires a condition (label or tensor)")
        
        if isinstance(cond, torch.Tensor) and cond.dim() == 1:
            cond_emb = self.label_emb(cond).view(-1, 100, 1, 1)
            x = x.view(x.size(0), -1)
            x = torch.cat([x, cond_emb.squeeze(-1).squeeze(-1)], dim=1)
            x = self.l1(x)
            x = x.view(x.size(0), 128, self.init_size, self.init_size)
            return torch.tanh(self.model(x))
        else:
            x = torch.cat([x, cond], dim=1)
            x = self.model[0:2](x)
            x = self.model[2:](x)
            return torch.tanh(x)

# ---- SRGAN Generator ----
class SRGANGenerator(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        if scale_factor < 1 or (scale_factor & (scale_factor - 1)) != 0:
            raise ValueError(f"scale_factor must be a power of two (got {scale_factor})")

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        res_blocks = []
        for _ in range(5):
            res_blocks.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
            ))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        num_upsamples = int(np.log2(scale_factor))
        upscale = []
        for _ in range(num_upsamples):
            upscale.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            upscale.append(nn.PixelShuffle(2))
            upscale.append(nn.PReLU())
        self.upscale = nn.Sequential(*upscale)

        self.final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x0 = self.initial(x)
        res = self.res_blocks(x0)
        x = self.mid(res) + x0
        x = self.upscale(x)
        x = self.final(x)
        return torch.tanh(x)

# ---- Residual Block for ESRGAN ----
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

# ---- ESRGAN Generator ----
class ESRGANGenerator(nn.Module):
    def __init__(self, num_residuals=8):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residuals)]
        )
        self.final = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.residuals(x1)
        return self.final(x1 + x2)

# ------------------- Config -------------------
DENOISE_CKPT_PTH = "weights/denoise_epoch_499.pth"
CGAN_CKPT_PTH    = "weights/cgan_epoch_500_converted.pth"
CGAN_CKPT_KERAS  = "weights/cgan_epoch_500.keras"
SRGAN_CKPT_PTH   = "weights/srgan_epoch_499.pth"
ESRGAN_CKPT_PTH  = "weights/esrgan_epoch_500.pth"

ModelCfg = Dict[str, Dict]
MODEL_CFG: ModelCfg = {
    "denoise": {"normalize": ([0.5]*3, [0.5]*3), "activation": "tanh", "pad_divisor": 4, "scale": 1},
    "cgan": {"normalize": ([0.5]*3, [0.5]*3), "activation": "tanh", "pad_divisor": 4, "scale": 1},
    "srgan": {"normalize": ([0.5]*3, [0.5]*3), "activation": "tanh", "pad_divisor": 4, "scale": 4},
    "esrgan": {"normalize": None, "activation": None, "pad_divisor": 4, "scale": 1},
}

# ------------------- Helper functions ---
def load_generator(checkpoint_path, device):
    model = ESRGANGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['G'])  # adjust if your checkpoint differs
    model.eval()
    return model

def preprocess_image(image_path, device):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor

def postprocess_tensor(tensor):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    return img

# ------------------- Other Utils -------------------
def load_state_safely(model: nn.Module, checkpoint_path: str, key_candidates=("generator", "state_dict", "G")):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        for k in key_candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    logger.info(f"Loaded PyTorch weights from {checkpoint_path}")

def get_padding(image: Image.Image, divisor: int, scale: int = 1) -> Tuple[int, int, int, int]:
    w, h = image.size
    effective_divisor = divisor * scale
    pad_w = (effective_divisor - w % effective_divisor) % effective_divisor
    pad_h = (effective_divisor - h % effective_divisor) % effective_divisor
    return (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

def denorm_for_view(t: torch.Tensor, mean, std):
    mean = torch.tensor(mean, device=t.device).view(-1, 1, 1)
    std = torch.tensor(std, device=t.device).view(-1, 1, 1)
    return (t * std) + mean

def to_base64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def make_graphs(input_vis: torch.Tensor, output_vis: torch.Tensor) -> str:
    noise = (input_vis - output_vis)
    noise_np = noise.permute(1, 2, 0).cpu().numpy()
    abs_error = noise.abs().permute(1, 2, 0).cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.clip(noise_np * 0.5 + 0.5, 0, 1))
    axs[0].set_title("Noise Map (Input - Output)"); axs[0].axis('off')
    axs[1].imshow(np.clip(abs_error * 2.0, 0, 1))
    axs[1].set_title("Absolute Error Map"); axs[1].axis('off')
    axs[2].hist(noise.cpu().numpy().flatten(), bins=50)
    axs[2].set_title("Histogram of Noise Values"); axs[2].set_xlabel("Difference"); axs[2].set_ylabel("Frequency")
    return fig_to_base64(fig)

def bicubic_to_size(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.BICUBIC)

# ------------------- Instantiate PyTorch Models -------------------
PT_MODELS: Dict[str, nn.Module] = {
    "denoise": DenoiseGenerator().to(DEVICE),
    "cgan": CGANGenerator().to(DEVICE),
    "srgan": SRGANGenerator(scale_factor=MODEL_CFG["srgan"]["scale"]).to(DEVICE),
    "esrgan": ESRGANGenerator(num_residuals=8).to(DEVICE),
}

# Load .pth weights
for name, path in [
    ("denoise", DENOISE_CKPT_PTH),
    ("cgan", CGAN_CKPT_PTH),
    ("srgan", SRGAN_CKPT_PTH),
    ("esrgan", ESRGAN_CKPT_PTH),
]:
    try:
        load_state_safely(PT_MODELS[name], path)
    except Exception as e:
        logger.warning(f"[{name}] PyTorch ckpt not loaded ({e}). Using random init for that backend.")

# ------------------- Load Keras cGAN -------------------
KERAS_CGAN = None
if TF_AVAILABLE and CGAN_CKPT_KERAS:
    try:
        KERAS_CGAN = tf.keras.models.load_model(CGAN_CKPT_KERAS, compile=False)
        logger.info(f"Loaded Keras cGAN from {CGAN_CKPT_KERAS}")
    except Exception as e:
        logger.warning(f"Keras cGAN not loaded ({e}).")

# ------------------- Routes -------------------
@app.get("/")
async def root():
    backends = {
        "denoise": "torch",
        "cgan": ("keras" if KERAS_CGAN is not None else "torch") + " (configurable)",
        "srgan": "torch",
        "esrgan": "torch",
    }
    return {"message": "Unified GAN API is running", "models": list(PT_MODELS.keys()), "default_backends": backends}

@app.post("/enhance")
async def enhance_image(
    model: str,
    file: UploadFile = File(...),
    cgan_backend: str = "auto",
    label: Optional[int] = Form(default=None),
    cond_file: Optional[UploadFile] = File(default=None),
):
    model = model.lower()
    if model not in PT_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose one of {list(PT_MODELS.keys())}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        cfg = MODEL_CFG[model]
        scale = cfg.get("scale", 1)

        pad_divisor = cfg["pad_divisor"]
        padding = get_padding(image, divisor=pad_divisor, scale=scale)
        pad_tf = transforms.Pad(padding, fill=0)

        if model == "esrgan":
            # Use provided preprocessing for ESRGAN
            use_keras = False
            x_pt = preprocess_image(io.BytesIO(contents), DEVICE)
            net = PT_MODELS[model].eval()
            with torch.no_grad():
                y = net(x_pt)
            y_pil = postprocess_tensor(y)
            x_pil = image  # Original image for visualization
            x_vis = transforms.ToTensor()(x_pil).to(DEVICE)
            y_vis = transforms.ToTensor()(y_pil).to(DEVICE)
        else:
            # Existing preprocessing for other models
            mean, std = cfg["normalize"]
            tfm_pt = transforms.Compose([
                pad_tf,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            x_pt = tfm_pt(image).unsqueeze(0).to(DEVICE)

            cond_pt = None
            cond_np = None
            if model == "cgan":
                if cond_file is not None:
                    cond_bytes = await cond_file.read()
                    cond_img = Image.open(io.BytesIO(cond_bytes)).convert("RGB")
                    cond_img = pad_tf(cond_img)
                    cond_pt = transforms.ToTensor()(cond_img).unsqueeze(0).to(DEVICE)
                    cond_np = np.array(cond_img).astype(np.float32) / 255.0
                elif label is not None:
                    cond_pt = torch.tensor([label], device=DEVICE)
                else:
                    raise HTTPException(status_code=400, detail="cGAN requires either a label or condition image")

            def run_torch_path():
                net = PT_MODELS[model].eval()
                if model == "cgan":
                    if cond_pt is None:
                        raise RuntimeError("cGAN requires a condition")
                    if cond_pt.dim() == 1:
                        z = torch.randn(x_pt.size(0), 100, 1, 1, device=DEVICE)
                        y = net(z, cond=cond_pt)
                    else:
                        y = net(x_pt, cond=cond_pt)
                else:
                    y = net(x_pt)
                x_vis = denorm_for_view(x_pt.squeeze(0), mean, std).clamp(0, 1)
                y_vis = ((y.squeeze(0)) * 0.5 + 0.5).clamp(0, 1) if cfg["activation"] == "tanh" else y.squeeze(0).clamp(0, 1)
                return x_vis, y_vis

            def run_keras_cgan():
                if not TF_AVAILABLE or KERAS_CGAN is None:
                    raise RuntimeError("Keras backend not available or model not loaded.")
                x_img = pad_tf(image)
                x_np = (np.asarray(x_img).astype(np.float32) / 255.0)
                x_np = (x_np - 0.5) / 0.5
                x_np = np.expand_dims(x_np, 0)

                try:
                    if isinstance(KERAS_CGAN.inputs, (list, tuple)) and len(KERAS_CGAN.inputs) >= 2:
                        if cond_np is None and label is None:
                            cond_guess = np.zeros_like(x_np, dtype=np.float32)
                            y_np = KERAS_CGAN.predict([x_np, cond_guess], verbose=0)
                        elif cond_np is not None:
                            cond_in = (cond_np - 0.5) / 0.5
                            cond_in = np.expand_dims(cond_in, 0)
                            y_np = KERAS_CGAN.predict([x_np, cond_in], verbose=0)
                        else:
                            y_np = KERAS_CGAN.predict([x_np, np.array([[label]], dtype=np.float32)], verbose=0)
                    else:
                        y_np = KERAS_CGAN.predict(x_np, verbose=0)
                except Exception as e:
                    raise RuntimeError(f"Keras cGAN forward failed: {e}")

                if cfg["activation"] == "tanh":
                    y_np = (y_np * 0.5 + 0.5)
                y_np = np.clip(y_np, 0.0, 1.0)[0]
                x_vis = transforms.ToTensor()(x_img)
                y_vis = torch.from_numpy(y_np.transpose(2, 0, 1))
                return x_vis.to(DEVICE), y_vis.to(DEVICE)

            use_keras = model == "cgan" and (cgan_backend == "keras" or (cgan_backend == "auto" and KERAS_CGAN is not None))
            x_vis, y_vis = run_keras_cgan() if use_keras else run_torch_path()
            x_pil = transforms.ToPILImage()(x_vis.cpu())
            y_pil = transforms.ToPILImage()(y_vis.cpu())

        if model in ("denoise", "cgan", "esrgan"):
            x_pil = x_pil.crop((padding[0], padding[1],
                                padding[0] + original_size[0],
                                padding[1] + original_size[1]))
            y_pil = y_pil.crop((padding[0], padding[1],
                                padding[0] + original_size[0],
                                padding[1] + original_size[1]))
        else:
            x_pil_cropped = x_pil.crop((padding[0], padding[1],
                                        padding[0] + original_size[0],
                                        padding[1] + original_size[1]))
            x_pil = bicubic_to_size(x_pil_cropped, y_pil.size)

        x_graph_t = transforms.ToTensor()(x_pil).to(DEVICE)
        y_graph_t = transforms.ToTensor()(y_pil).to(DEVICE)
        graph_base64 = make_graphs(x_graph_t, y_graph_t)

        out_b64 = to_base64_png(y_pil)

        return {
            "denoised_image_base64": out_b64,
            "noise_graph_base64": graph_base64,
            "backend": "keras" if use_keras else "torch"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhancement failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image enhancement failed")