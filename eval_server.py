from fastapi import FastAPI, Form, File, Response
from eval_wrapper.eval import EvalWrapper,eval_scene
import uvicorn
import open3d as o3d
from eval_wrapper.eval_utils import npy2ply, filter_all_masks
import os 
import torch
from huggingface_hub import hf_hub_download
app = FastAPI()

from torch.utils.cpp_extension import _get_cuda_arch_flags
print("[INFO] Using CUDA arch flags:", _get_cuda_arch_flags())

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

@app.post("/reconstruct")
def predict(response : Response, 
            data_dir : str = Form("example_scene/"), 
            n_pred_views : int = Form(5), # Number of predicted views along each axis in a grid, 5--> 22 views total
            visualize : bool = Form(False), # Spins up a rerun client to visualize predictions and camera posees
            run_octmae : bool = Form(False), # Novel views sampled with the OctMAE parameters (see paper)
            set_conf : float = Form(5), # Sets confidence threshold to N 
            filter_all_masks : bool = Form(False), # Use all masks, point gets rejected if in background for a single mask
            tsdf : bool = Form(False), # Fits TSDF to depth maps
            rr_addr : str = Form("0.0.0.0:"+os.getenv("RERUN_RECORDING","9876")), 
            no_input_mask : bool = Form(False), # Do not use input masks
            no_pred_mask : bool = Form(False), # Do not use predicted masks
            no_filter_input_view : bool = Form(False), # Do not filter input view
            false_positive : float = Form(None), # False positive rate for the predicted masks
            false_negative : float = Form(None) # False negative rate for the predicted masks
            ):
    
    print("Loading checkpoint from Huggingface")
    rayst3r_checkpoint = hf_hub_download("bartduis/rayst3r", "rayst3r.pth")
    
    model = EvalWrapper(rayst3r_checkpoint,distributed=False)
    print("Checkpoint loaded")
    all_points = eval_scene(model, data_dir, visualize=visualize, rr_addr=rr_addr, run_octmae=run_octmae, set_conf=set_conf,
                            no_input_mask=no_input_mask, no_pred_mask=no_pred_mask, no_filter_input_view=no_filter_input_view, false_positive=false_positive,
                            false_negative=false_negative, n_pred_views=n_pred_views,
                            do_filter_all_masks=filter_all_masks, tsdf=tsdf).cpu().numpy()
    all_points = eval_scene(model, data_dir, n_pred_views = n_pred_views).cpu().numpy()
    all_points_save = os.path.join(data_dir,"inference_points.ply")
    o3d_pc = npy2ply(all_points,colors=None,normals=None)
    o3d.io.write_point_cloud(all_points_save, o3d_pc)
    
    print(f"Saved point cloud to {all_points_save}")
    #clear model cache
    torch.cuda.empty_cache()
    return {"message": f"Point cloud saved to {all_points_save}"}

if __name__ == "__main__":
    PORT = os.environ.get("PORT", 6000)
    uvicorn.run(app, host="0.0.0.0", port=PORT)

