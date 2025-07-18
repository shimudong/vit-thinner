import argparse
import os
import re
from collections import OrderedDict
from glob import glob
from types import MethodType

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from imagenet_1k import IMAGENET2012_CLASSES

WNID_TO_IDX = {wnid: i for i, wnid in enumerate(IMAGENET2012_CLASSES.keys())}

# ## Core Decomposition and Masking Logic

def generate_orthogonal_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Generates a random orthogonal matrix of size n x n."""
    H = torch.randn(n, n, device=device)
    Q, _ = torch.linalg.qr(H)

    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def decompose_identity(n: int, rank: int, epochs: int = 5000, lr: float = 0.0001, diag_weight: float = 1.0, device: torch.device = "cuda", verbose: bool = True):
    """
    Finds matrices A and B such that B @ A approximates the identity matrix.

    Args:
        n (int): The dimensionality of the identity matrix.
        rank (int): The rank of the decomposition (inner dimension).
        epochs (int): Number of optimization epochs.
        lr (float): Learning rate for the Adam optimizer.
        diag_weight (float): Weight for the diagonal element loss.
        device (torch.device): The device to perform computations on.
        verbose (bool): Whether to print progress.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The decomposed matrices (A, B).
    """
    print(f"Decomposing Identity({n}) into ({n}x{rank}) and ({rank}x{n}) matrices...")
    I = torch.eye(n, device=device)

    # Initialize A as a fixed orthogonal matrix projection
    A = generate_orthogonal_matrix(n, device)[:rank, :]
    A.requires_grad_(False)

    # Initialize B with random values
    B = 1 / (rank**0.5) * torch.randn(n, rank, device=device)
    B.requires_grad_(True)

    optimizer = optim.Adam([B], lr=lr)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        C = B @ A

        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        off_diag_loss = torch.norm(C.masked_select(mask))**2
        diag_loss = torch.norm(torch.diag(C) - 1.0)**2
        loss = off_diag_loss + diag_weight * diag_loss

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if verbose and (epoch == 0 or (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1):
            with torch.no_grad():
                recon_error = torch.norm(C - I, p="fro").item()
                print(f'Epoch [{epoch+1:5d}/{epochs}] | Loss: {loss.item():.6f} | '
                      f'Recon Err: {recon_error:.6f}')
                
    return A.detach(), B.detach()


def apply_mask_to_linear_layer(linear_layer: nn.Linear, mask: torch.Tensor, side: str):
    """Applies a projection mask to a linear layer's weights and biases."""
    device = linear_layer.weight.device
    mask = mask.to(device).to(linear_layer.weight.dtype)
    new_bias = None

    with torch.no_grad():
        if side == "left":
            new_weight = linear_layer.weight.data @ mask.T
        elif side == "right":
            new_weight = mask.T @ linear_layer.weight.data
            if linear_layer.bias is not None:
                new_bias = mask.T @ linear_layer.bias.data
        else:
            raise ValueError(f"Invalid side '{side}', must be 'left' or 'right'.")

    linear_layer.weight.data = new_weight
    if new_bias is not None:
        linear_layer.bias.data = new_bias


def layer_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x @ self.A.to(x.dtype)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    out = (x - mean) / torch.sqrt(var + self.eps)
    out = self.weight * out + self.bias
    return out @ self.B.to(x.dtype)


def gamma_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x @ self.A.to(x.dtype)
    x = x * self.gamma
    x = x @ self.B.to(x.dtype)
    return x


def mask_model_linear_layers(model: nn.Module, rank: int, device: torch.device):
    """
    Modifies the Vision Transformer model by applying low-rank projections.

    Args:
        model (nn.Module): The ViT model to modify.
        rank (int): The target rank for the projection.
        device (torch.device): The device for computation.

    Returns:
        nn.Module: The modified model.
    """
    feature_dim = model.head.in_features
    print(f"Model feature dimension detected: {feature_dim}")

    total_param = sum(p.numel() for p in model.parameters())
    reduced_param = (feature_dim - rank) / feature_dim * total_param - 2 * feature_dim ** 2
    print(f"Reduced param: {reduced_param / total_param}")

    A, B = decompose_identity(feature_dim, rank, epochs=10000, device=device)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "qkv" in name or "fc1" in name or "head" in name:
                side, projection = "left", A
            elif "attn.proj" in name or "fc2" in name:
                side, projection = "right", B
            else:
                continue
            print(f"Applying '{side}' mask to Linear layer: {name}")
            apply_mask_to_linear_layer(module, projection, side)

        elif "patch_embed" in name and not ("patch_embed." in name):
            print(f"Applying mask to Patch Embedding: {name}")
            proj_layer = module.proj
            c, h, w = proj_layer.weight.shape[1:]
            
            flat_weight = proj_layer.weight.data.flatten(1)
            proj_layer.weight.data = (B.T @ flat_weight).view(-1, c, h, w)
            if proj_layer.bias is not None:
                proj_layer.bias.data = B.T @ proj_layer.bias.data

        elif "norm" in name and not ("norm." in name) and not isinstance(module, nn.Identity):
            print(f"Hooking LayerNorm: {name}")
            module.forward = MethodType(layer_norm_forward, module)
            module.A = A
            module.B = B

        elif isinstance(module, timm.models.vision_transformer.LayerScale):
            print(f"Hooking LayerScale: {name}")
            module.forward = MethodType(gamma_forward, module)
            module.A = A
            module.B = B

    print("Applying mask to Position and Registration Embeddings")
    if hasattr(model, "pos_embed") and model.pos_embed is not None:
        model.pos_embed.data = model.pos_embed.data @ B
        
    if hasattr(model, "reg_token") and model.reg_token is not None:
        model.reg_token.data = model.reg_token.data @ B

    if hasattr(model, "cls_token") and model.cls_token is not None:
        model.cls_token.data = model.cls_token.data @ B
    
    return model


class ImageNetFlat(Dataset):
    """
    Custom PyTorch Dataset for ImageNet validation set in a flat directory.
    Filenames are expected to follow the format: '..._nXXXXXXXX.JPEG'
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob(os.path.join(root_dir, '*.JPEG'))
        self.wnid_pattern = re.compile(r'_(n\d{8})\.JPEG$')
        
        # Filter out paths that don't match the pattern
        self.valid_files = []
        for path in self.image_paths:
            if self.wnid_pattern.search(path):
                self.valid_files.append(path)

        if len(self.image_paths) != len(self.valid_files):
            print(f"Warning: Filtered {len(self.image_paths) - len(self.valid_files)} files that did not match the naming pattern.")

        assert self.valid_files, f"No valid image files found in {root_dir}"

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_path = self.valid_files[idx]
        
        # Extract label from filename
        match = self.wnid_pattern.search(img_path)
        wnid = match.group(1)
        label = WNID_TO_IDX.get(wnid, -1) # Use -1 for unknown classes

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Evaluates the model's performance on the provided dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device for computation.
    """
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            
            # Get top 5 predictions
            _, preds = logits.topk(5, dim=1, largest=True, sorted=True)
            
            # Check for correct predictions
            labels_resized = labels.view(-1, 1)
            top1_correct += torch.eq(preds[:, :1], labels_resized).sum().item()
            top5_correct += torch.eq(preds, labels_resized).sum().item()
            total += images.size(0)

            print((top1_correct / total) * 100)
    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100

    print("\n--- Evaluation Results ---")
    print(f'Images Evaluated: {total}')
    print(f'Top-1 Accuracy:   {top1_acc:.2f}%')
    print(f'Top-5 Accuracy:   {top5_acc:.2f}%')


def main():
    parser = argparse.ArgumentParser(description="Apply low-rank projection to a ViT model and evaluate.")
    parser.add_argument('--model-name', type=str, default='vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k', help='Name of the timm model to use.')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Path to the model checkpoint (.safetensors or .pth).')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of the ImageNet validation set.')
    parser.add_argument('--rank', type=int, required=True, help='Target rank for the decomposition.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading.')
    args = parser.parse_args()
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading model '{args.model_name}'...")
    pretrained_cfg = timm.create_model(args.model_name).default_cfg
    pretrained_cfg['file'] = args.checkpoint_path
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=dict(file=args.checkpoint_path, custom_load=False),
    ).to(device)


    # --- 2. Create Data Transforms and DataLoader ---
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    
    dataset = ImageNetFlat(root_dir=args.data_root, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- 3. Modify Model ---
    print(f"\nApplying projection to rank {args.rank}...")
    model = mask_model_linear_layers(model, args.rank, device)
    print("\nModel modification complete.")

    # --- 4. Evaluate Modified Model ---
    evaluate(model, dataloader, device)

if __name__ == '__main__':
    main()