import sys
import socket
import os
# Add the current working directory to the Python path
current_dir = os.getcwd()
sys.path.append(current_dir)
from xps.util import *

root_log_dir = "logs"
n_views = 2 
dataset_size = -1

imshape_input = (480,640)
imshape_output = (480,640)
render_size = (480,640)

preload_train = False
data_dirs = ["/YOUR/PATH/TO/rayst3r/dataset"]
dino_features = [4,11,17,23]
datasets = ['gso','octmae']
prefetch_dino = False
normalize_mode = 'median'
#start_from = "checkpoints/gso_conf.pth"
start_from = None

noise_std = 0.005
view_select_mode = "new_zoom"
rendered_views_mode = "always"
dataset_train = f"GenericLoader(size={dataset_size},seed=747,dir={repr(data_dirs)},split='train',datasets={datasets},mode='fast',prefetch_dino={prefetch_dino}," \
+f"dino_features={dino_features},view_select_mode='{view_select_mode}',noise_std={noise_std},rendered_views_mode='{rendered_views_mode}')"
dataset_test = f"GenericLoader(size=1000,seed=787,dir={repr(data_dirs)},split='test',datasets={datasets},mode='fast',prefetch_dino={prefetch_dino}," \
+f"dino_features={dino_features},view_select_mode='{view_select_mode}',noise_std={noise_std},rendered_views_mode='{rendered_views_mode}')"
dataset_just_load = f"GenericLoader(size=1000,seed=787,dir={repr(data_dirs)},split='test',datasets={datasets},mode='fast',prefetch_dino={prefetch_dino}," \
+f"dino_features={dino_features},view_select_mode='{view_select_mode}',noise_std={noise_std},rendered_views_mode='{rendered_views_mode}')"

augmentor = "Augmentor()"

patch_size = 16
save_every = 1

vit="base"
if vit == "debug":
    enc_dim = 128
    dec_dim = 128
    n_heads = 4
    enc_depth = 4 
    dec_depth = 4
    head_n_layers = 1
    head_dim = 128
    lr = 3e-4
    batch_size = 20
    blr = 1.5e-4
elif vit == "debug_2":
    enc_dim = 512
    dec_dim = 512
    n_heads = 4
    enc_depth = 4
    dec_depth = 10
    head_n_layers = 1
    head_dim = 128
    blr = 1.5e-4
    batch_size = 18
elif vit == "small":
    enc_dim = 384
    dec_dim = 384
    n_heads = 6
    enc_depth = 12 
    dec_depth = 12
    head_n_layers = 1
    head_dim = 128
    batch_size = 6
    blr = 1.5e-4
elif vit == "base":
    enc_dim = 768
    dec_dim = 768
    n_heads = 12
    enc_depth = 4 
    dec_depth = 12
    head_n_layers = 1
    head_dim = 128
    batch_size = 10
    blr = 1.5e-4

lambda_classifier = 0.1
for skip_conf_points in [False]:
    skip_conf_mask = True
    model = f"RayQuery(ray_enc=RayEncoder(dim={enc_dim},num_heads={n_heads},depth={enc_depth},img_size={render_size},patch_size={patch_size})," + \
            f"pointmap_enc=PointmapEncoder(dim={enc_dim},num_heads={n_heads},depth={enc_depth},img_size={render_size},patch_size={patch_size})," + \
            f"dino_layers={dino_features}," + \
            f"pts_head_type='dpt_depth'," + \
            f"classifier_head_type='dpt_mask'," + \
            f"decoder_dim={dec_dim},decoder_depth={dec_depth},decoder_num_heads={n_heads},imshape={render_size}," + \
            f"criterion=DepthCompletion(ConfLoss(L21,skip_conf={skip_conf_points}),ConfLoss(ClassifierLoss(BCELoss()),skip_conf={skip_conf_mask}),lambda_classifier={lambda_classifier}),return_all_blocks=True)"

    key = f"conf_points_{skip_conf_points==False}"
    key = gen_key(key)
    logdir = os.path.join(root_log_dir,key)
    resume=logdir
    wandb_run_name=key
    os.makedirs(logdir,exist_ok=True)

    n_epochs = 20
    eval_every = 1
    max_norm = -1
    OMP_NUM_THREADS=16
    warmup_epochs = 1
    
    executable = f"OMP_NUM_THREADS={OMP_NUM_THREADS} torchrun --nnodes 1 --nproc_per_node $(python -c 'import torch; print(torch.cuda.device_count())') --master_port $((RANDOM%500+29000)) main.py"
    #executable = f"python main.py"
    if '--just_load' in sys.argv:
        batch_size = 5
        command = f"{executable} --{dataset_train=} --{dataset_test=} --{dataset_just_load=} --{logdir=} --{resume=} --{model=} --{batch_size=} --{normalize_mode=} --{augmentor=}"
    else:
        command = f"{executable} --{dataset_train=} --{dataset_test=} --{logdir=} --{n_epochs=} --{resume=} --{normalize_mode=} --{augmentor=} --{warmup_epochs=}" 
        command += f" --{model=} --{eval_every=} --{batch_size=} --{save_every=} --{max_norm=}"
        command += f" --{blr=}"
        if start_from is not None:
            command += f" --{start_from=}"
        if not '--no_wandb' in sys.argv:
            command += f" --wandb_project=3dcomplete " + \
                    f"--{wandb_run_name=}"

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if not '--no_wandb' in arg:
                command += f" {arg}"
    print(command)
