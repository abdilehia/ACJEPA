import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from models.AE_2D_Causal import AE_models
from models.ACJEPA import ACJEPA_models
from utils.pose import Pose
from utils.transform import Transform
from utils.motion_process import plot_3d_motion, t2m_kinematic_chain


def main():
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(3470)
    np.random.seed(3470)
    torch.manual_seed(3470)
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    #################################################################################
    #                                    Train Data                                 #
    #################################################################################
    dim_pose = 3
    mean = np.load(f'utils/22x3_mean_std/t2m/22x3_mean.npy')
    std = np.load(f'utils/22x3_mean_std/t2m/22x3_std.npy')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    after_mean = torch.from_numpy(np.load(pjoin('./checkpoints', 't2m', "AE_2D_Causal", 'AE_2D_Causal_Post_Mean.npy'))) # make sure this is either calculated or obtained from given checkpoint files
    after_std = torch.from_numpy(np.load(pjoin('./checkpoints', 't2m', "AE_2D_Causal", 'AE_2D_Causal_Post_Std.npy'))) # make sure this is either calculated or obtained from given checkpoint files
    after_mean = after_mean.to(device)
    after_std = after_std.to(device)

    mean = torch.from_numpy(mean).to(device) # make sure this is either calculated or obtained from given checkpoint files
    std = torch.from_numpy(std).to(device)

    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin('./checkpoints', 't2m', 'ACJEPA_Flow_S_PatchSize22', 'model')

    ae = AE_models['AE_Model'](input_width=dim_pose)
    ae_ckpt = torch.load('./checkpoints/t2m/AE_2D_Causal/model/latest.tar', map_location='cpu')
    ae.load_state_dict(ae_ckpt['ae'])

    ae.eval()
    ae.to(device)
    for param in ae.parameters():
        param.requires_grad = False

    acjepa = ACJEPA_models["ACJEPA-Flow-S-PatchSize22"](input_dim=ae.output_emb_width, cond_mode='text')
    checkpoint = torch.load(pjoin(model_dir, 'latest.tar'), map_location=device)
    missing_keys, unexpected_keys = acjepa.load_state_dict(checkpoint['ema_acjepa'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

    acjepa.eval()
    acjepa.to(device)

    #################################################################################
    #                                  Initialisation                               #
    #################################################################################
    motion_id = '000002'
    motion = np.load(f"datasets/HumanML3D/new_joints/{motion_id}.npy")
    with open(f"datasets/HumanML3D/texts/{motion_id}.txt", "r+") as f:
        texts = f.readlines()
        conds = [texts[0].split("#")[0]]

    text_cfg = 5

    pose = Pose()
    pose.init_pose(motion[0])
    transform = Transform(mean, std, device)

    motion = torch.from_numpy(motion).to(device).unsqueeze(0)
    
    # motion_seq = torch.tensor(pose.t_pose(), device=device).view(1, 22, 1, 3).permute(0, 2, 1, 3).repeat(1, 4, 1, 1)
    motion_seq = motion[:, 0:4, :, :]
    motion_seq = transform.transform(motion_seq)


    with torch.no_grad():
        latent_seq = ae.encode(motion_seq)

    with torch.no_grad():
        cond_vector = acjepa.encode_text(conds)
        if not (text_cfg == 1):
            uncond_vector = torch.zeros_like(cond_vector)
            cond_vector = torch.cat([cond_vector, uncond_vector], dim=0)

        latent_seq = latent_seq.permute(0, 2, 3, 1)
        latent_seq = transform.transform(latent_seq, after_mean, after_std)
        latent_seq = latent_seq.permute(0, 3, 1, 2)
    
        if not (text_cfg == 1):
            latent_seq = latent_seq.repeat(2, 1, 1, 1)

    for i in range(motion_seq.shape[1] // 4, motion.shape[1] // 4):
        with torch.no_grad():
            seq_len = latent_seq.shape[2]
            N_infer_tokens = seq_len * acjepa.patches_per_frame
            infer_causal_mask = torch.tril(torch.ones(N_infer_tokens, N_infer_tokens, device=device, dtype=torch.bool))
            infer_causal_mask = infer_causal_mask.unsqueeze(0).unsqueeze(0)
    
            if not (text_cfg == 1):
                infer_causal_mask = infer_causal_mask.repeat(2, 1, 1, 1)
    
            predicted_sequence = acjepa.forward_with_CFG(
                latent_seq, 
                conds=cond_vector,
                cfg=text_cfg,
                attention_mask=infer_causal_mask)

            latent_seq = torch.cat([latent_seq, predicted_sequence[:, :, -1:, :]], dim=2)
    

    if not (text_cfg == 1):
        cond_pred, _ = latent_seq.chunk(2, dim=0)
        latent_seq = cond_pred

    with torch.no_grad(): 
        latent_seq = latent_seq.permute(0, 2, 3, 1)
        latent_seq = transform.inv_transform(latent_seq, after_mean, after_std)
        latent_seq = latent_seq.permute(0, 3, 1, 2)
        motion_seq = ae.decode(latent_seq)
        motion_seq = transform.inv_transform(motion_seq)

    predicted = motion_seq.detach().cpu().numpy()
    original = motion.detach().cpu().numpy()

    result_dir = "outputs"
    kinematic_chain = t2m_kinematic_chain
    for k, (caption, pred, orig) in enumerate(zip(conds, predicted, original)):
        s_path = pjoin(result_dir, str(k))
        os.makedirs(s_path, exist_ok=True)

        l, j, d = pred.shape
        print("---->Prediction %d: %s %d" % (k, caption, l))
        save_path = pjoin(s_path, "caption_%s_len%d.mp4" % (caption, l))
        plot_3d_motion(save_path, kinematic_chain, pred, title=caption, fps=20)

        l, j, d = orig.shape
        print("---->Original %d: %s %d" % (k, caption, l))
        save_path = pjoin(s_path, "caption_%s_len%d.mp4" % (caption + "_original", l))
        plot_3d_motion(save_path, kinematic_chain, orig, title=caption, fps=20)


if __name__ == "__main__":
    main()