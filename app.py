import argparse
import numpy as np
import gradio as gr
import torch
import rembg
import trimesh
from moge.model.v1 import MoGeModel
from utils.geometry import compute_pointmap
import os, shutil
import cv2
from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt
from eval_wrapper.eval import EvalWrapper, eval_scene
from torchvision import transforms

outdir = "/tmp/rayst3r"

# loading all necessary models
print("Loading DINOv2 model")
dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitl14_reg")
dino_model.eval()
dino_model.to("cuda")

print("Loading MoGe model")
device = torch.device("cuda")
# Load the model from huggingface hub (or load from local).
moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

print("Loading RaySt3R model")
rayst3r_checkpoint = hf_hub_download("bartduis/rayst3r", "rayst3r.pth")
rayst3r_model = EvalWrapper(rayst3r_checkpoint)

def depth2uint16(depth):
    return depth * torch.iinfo(torch.uint16).max / 10.0 # threshold is in m, convert to uint16 value

def save_tensor_as_png(tensor: torch.Tensor, path: str, dtype: torch.dtype | None = None):
    if dtype is None:
        dtype = tensor.dtype
    Image.fromarray(tensor.to(dtype).cpu().numpy()).save(path)

def colorize_points_with_turbo_all_dims(points, method='norm',cmap='turbo'):
    """
    Assigns colors to 3D points using the 'turbo' colormap based on a scalar computed from all 3 dimensions.

    Args:
        points (np.ndarray): (N, 3) array of 3D points.
        method (str): Method for reducing 3D point to scalar. Options: 'norm', 'pca'.

    Returns:
        np.ndarray: (N, 3) RGB colors in [0, 1].
    """
    assert points.shape[1] == 3, "Input must be of shape (N, 3)"

    if method == 'norm':
        scalar = np.linalg.norm(points, axis=1)
    elif method == 'pca':
        # Project onto first principal component
        mean = points.mean(axis=0)
        centered = points - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        scalar = centered @ vh[0]  # Project onto first principal axis
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Normalize scalar to [0, 1]
    scalar_min, scalar_max = scalar.min(), scalar.max()
    normalized = (scalar - scalar_min) / (scalar_max - scalar_min + 1e-8)

    # Apply turbo colormap
    cmap = plt.colormaps.get_cmap(cmap)
    colors = cmap(normalized)[:, :3]  # Drop alpha

    return colors

def prep_for_rayst3r(img,depth_dict,mask):
    H, W = img.shape[:2]
    intrinsics = depth_dict["intrinsics"].detach().cpu()
    intrinsics[0] *= W
    intrinsics[1] *= H

    input_dir = os.path.join(outdir, "input")
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)
    # save intrinsics
    torch.save(intrinsics, os.path.join(input_dir, "intrinsics.pt"))

    # save depth
    depth = depth_dict["depth"].cpu()
    depth = depth2uint16(depth)
    save_tensor_as_png(depth, os.path.join(input_dir, "depth.png"),dtype=torch.uint16)
    
    # save mask as bool 
    save_tensor_as_png(torch.from_numpy(mask).bool(), os.path.join(input_dir, "mask.png"),dtype=torch.bool)
    # save image
    save_tensor_as_png(torch.from_numpy(img), os.path.join(input_dir, "rgb.png"))

def rayst3r_to_glb(img,depth_dict,mask,max_total_points=10e6,rotated=False):
    prep_for_rayst3r(img,depth_dict,mask)
    rayst3r_points = eval_scene(rayst3r_model,os.path.join(outdir, "input"),do_filter_all_masks=True,dino_model=dino_model).cpu()

    # subsample points
    n_points = min(max_total_points,rayst3r_points.shape[0])
    rayst3r_points = rayst3r_points[torch.randperm(rayst3r_points.shape[0])[:n_points]].numpy()
    
    rayst3r_points[:,1] = -rayst3r_points[:,1]
    rayst3r_points[:,2] = -rayst3r_points[:,2]
    
    # make all points red
    colors = colorize_points_with_turbo_all_dims(rayst3r_points)

    # load the input glb
    scene = trimesh.Scene()
    pct = trimesh.PointCloud(rayst3r_points, colors=colors, radius=0.01)
    scene.add_geometry(pct)
    
    outfile = os.path.join(outdir, "rayst3r.glb")
    scene.export(outfile)
    return outfile


def input_to_glb(outdir,img,depth_dict,mask,rotated=False):
    H, W = img.shape[:2]
    intrinsics = depth_dict["intrinsics"].cpu().numpy()
    intrinsics[0] *= W
    intrinsics[1] *= H
    
    depth = depth_dict["depth"].cpu().numpy()
    cam2world = np.eye(4)
    points_world = compute_pointmap(depth, cam2world, intrinsics)

    scene = trimesh.Scene()
    pts = np.concatenate([p[m] for p,m in zip(points_world,mask)])
    col = np.concatenate([c[m] for c,m in zip(img,mask)])

    pts = pts.reshape(-1,3)
    pts[:,1] = -pts[:,1]
    pts[:,2] = -pts[:,2]


    pct = trimesh.PointCloud(pts, colors=col.reshape(-1,3))
    scene.add_geometry(pct)
    
    outfile = os.path.join(outdir, "input.glb")
    scene.export(outfile)
    return outfile

def depth_moge(input_img):
    input_img_torch = torch.tensor(input_img / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    output = moge_model.infer(input_img_torch)
    return output 

def mask_rembg(input_img):
    #masked_img = rembg.remove(input_img,)
    output_img = rembg.remove(input_img, alpha_matting=False, post_process_mask=True)

    # Convert to NumPy array
    output_np = np.array(output_img)
    alpha = output_np[..., 3]

    # Step 2: Erode the alpha mask to shrink object slightly
    kernel = np.ones((3, 3), np.uint8)  # Adjust size for aggressiveness
    eroded_alpha = cv2.erode(alpha, kernel, iterations=1)
    # Step 3: Replace alpha channel
    output_np[..., 3] = eroded_alpha
  
    mask = output_np[:,:,-1] >= 128
    rgb = output_np[:,:,:3]
    return mask, rgb 

def process_image(input_img):
    # resize the input image
    rotated = False
    #if input_img.shape[0] > input_img.shape[1]:
        #input_img = cv2.rotate(input_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #rotated = True
    input_img = cv2.resize(input_img, (640, 480))
    mask, rgb = mask_rembg(input_img)
    depth_dict = depth_moge(input_img)

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    input_glb = input_to_glb(outdir,input_img,depth_dict,mask,rotated=rotated)
    
    # visualize the input points in 3D in gradio
    inference_glb = rayst3r_to_glb(input_img,depth_dict,mask,rotated=rotated)

    return input_glb, inference_glb   

demo = gr.Interface(
    process_image,
    gr.Image(),
    [gr.Model3D(label="Input"), gr.Model3D(label="RaySt3R",)]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Gradio demo for RaySt3R"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to share this demo through Gradio."
    )
    args = parser.parse_args()
    demo.launch(share=args.share)
