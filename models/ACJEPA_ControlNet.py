import torch
import torch.nn as nn
from models.ACJEPA import ACJEPA
from models.ACJEPA import ACJEPATransBlock, LlamaRMSNorm
from models.ROPE import RopeND
import gc
from utils.train_utils import lengths_to_mask


#################################################################################
#                                  ACJEPA+ControlNet                             #
#################################################################################
class ACJEPA_ControlNet(ACJEPA):
    def __init__(self, input_dim, cond_mode, base_checkpoint, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.2, clip_dim=512,
                 diff_model='Flow', cond_drop_prob=0.1, max_length=49,
                 patch_size=(1, 22), stride_size=(1, 22),
                 clip_version='ViT-B/32', freeze_base=True, need_base=True, **kargs):
        # --------------------------------------------------------------------------
        # ACJEPA
        super().__init__(input_dim, cond_mode, latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                 num_heads=num_heads, dropout=dropout, clip_dim=clip_dim,
                 diff_model=diff_model, cond_drop_prob=cond_drop_prob, max_length=max_length,
                 patch_size=patch_size, stride_size=stride_size,
                 clip_version=clip_version, **kargs)

        # --------------------------------------------------------------------------
        # ControlNet
        self.c_control_embedder = c_control_embedder(3, self.latent_dim, patch_size=self.patch_size,
                                                     stride_size=self.stride_size)
        self.c_x_embedder = nn.Conv2d(self.input_dim, self.latent_dim, kernel_size=self.patch_size,
                                      stride=self.stride_size, bias=True)
        self.c_y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        self.c_rope = RopeND(nd=1, nd_split=[1], max_lens=self.max_lens)
        self.ControlNet = nn.ModuleList([
            ACJEPATransBlock(self.latent_dim, num_heads, mlp_size=ff_size, rope=self.c_rope, qk_norm=True) for _ in
            range(num_layers)
        ])
        self.zero_Linear = nn.ModuleList([
            nn.Linear(self.latent_dim, self.latent_dim) for _ in range(num_layers)
        ])
        self.initialize_weights_control()
        if need_base:
            for key, value in list(base_checkpoint['ema_acjepa'].items()):
                if key.startswith('ACJEPATransformer.'):
                    new_key = key.replace('ACJEPATransformer.', 'ControlNet.')
                    base_checkpoint['ema_acjepa'][new_key] = value.clone()
            missing_keys, unexpected_keys = self.load_state_dict(base_checkpoint['ema_acjepa'], strict=False)
            assert len(unexpected_keys) == 0
            del base_checkpoint
            gc.collect()
            torch.cuda.empty_cache()

        if self.cond_mode == 'text':
            print('ReLoading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        if freeze_base:
            for param in self.x_embedder.parameters():
                param.requires_grad = False
            for param in self.y_embedder.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False
            for param in self.ACJEPATransformer.parameters():
                param.requires_grad = False

    def initialize_weights_control(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ACJEPATransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ControlNet:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.c_control_embedder.zero_linear.weight, 0)
        nn.init.constant_(self.c_control_embedder.zero_linear.bias, 0)

        for block in self.zero_Linear:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)

    def forward_with_control(self, x, conds, attention_mask, cfg1=1.0, cfg2=1.0, control=None, index=None,
                             force_mask=False):
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            half = x[: len(x) // 3]
            x = torch.cat([half, half, half], dim=0)
        # controlnet
        conds = self.mask_cond(conds, force_mask=force_mask)
        c_control = self.c_control_embedder(control * index)
        if self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(c_control.shape[0], device=c_control.device) * self.cond_drop_prob).view(c_control.shape[0], 1, 1)
            c_control = c_control * (1. - mask)
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            c_control = torch.cat([c_control, c_control, torch.zeros_like(c_control)], dim=0)

        c_x = self.c_x_embedder(x).flatten(2).transpose(1, 2)
        c_y = self.c_y_embedder(conds).unsqueeze(1)
        c_x = c_x + c_control
        c_position_ids = self.position_ids_precompute[:, :c_x.shape[1]]
        c_out = []
        for c_block, c_linear in zip(self.ControlNet, self.zero_Linear):
            c_x = c_block(c_x, c_y, attention_mask, position_ids=c_position_ids)
            c_out.append(c_linear(c_x))
        # main branch
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        y = self.y_embedder(conds).unsqueeze(1)
        position_ids = self.position_ids_precompute[:, :x.shape[1]]
        # merging
        for block, c in zip(self.ACJEPATransformer, c_out):
            x = block(x, y, attention_mask, position_ids=position_ids)
            x = x + c
        x = self.final_layer(x, y)
        if not (cfg1 == 1.0 and cfg2 == 1.0):
            cond_eps, uncond_eps1, uncond_eps2 = torch.split(x, len(x) // 3, dim=0)
            half_eps = cond_eps + (cfg1-1) * (cond_eps - uncond_eps1) + (cfg2-1) * (cond_eps - uncond_eps2)
            x = torch.cat([half_eps, half_eps, half_eps], dim=0)
        return x

    def forward_control_loss(self, latents, y, m_lens, original, index, ae, mean_std):
        latents = latents.permute(0, 2, 3, 1)
        b, l, j, d = latents.shape
        device = latents.device
        
        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1).unsqueeze(-1), latents, torch.zeros_like(latents))


        context_latents = latents[:, :-1, :, :]  # Past/Current frames [B, L-1, J, d]
        target_latents  = latents[:, 1:, :, :].clone().detach() # Future frames [B, L-1, J, d]
        
        original = original.clone().detach()
        target_motion = original[:, :, 4:, :] # 

        # Shift the padding masks to match the L-1 sequence length
        context_mask = non_pad_mask[:, :-1]
        target_mask  = non_pad_mask[:, 1:]

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")
        
        pad_mask = context_mask.unsqueeze(-1).repeat(1, 1, self.patches_per_frame).flatten(1)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        N_tokens = (l - 1) * self.patches_per_frame
        causal_mask = torch.tril(torch.ones(N_tokens, N_tokens, device=device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        attention_mask = pad_mask & causal_mask

        random_indices = torch.randint(0, len(index), (b,)).to(device) # Gets b numbers between 0 and len(args.control_joints)
        indexx = torch.tensor(index, device=device)[random_indices] # makes a tensor from args.control_joints which is an array of numbers and samples from it
        mask_seq = torch.zeros((b, 3, l*4, j), device=device)

        for i in range(b): 
            if torch.rand(1).item() < 0.10:
                total_frames = int(m_lens[i] * 4)

                
                max_dropout_size = int(total_frames * 0.20)
                dropout_size = int(torch.randint(0, max_dropout_size + 1, (1,)).item())

                if dropout_size > 0 and dropout_size < total_frames:
                    max_start = total_frames - dropout_size
                    dropout_start = torch.randint(0, max_start + 1, (1,)).item()
                    dropout_end = dropout_start + dropout_size

                    mask_seq[i, :, :dropout_start, indexx[i]] = 1.0
                    mask_seq[i, :, dropout_end:, indexx[i]] = 1.0
                else:
                    mask_seq[i, :, :, indexx[i]] = 1.0
            else:
                mask_seq[i, :, :, indexx[i]] = 1.0

        mask_seq_input = mask_seq[:, :, :-4, :]
        mask_seq_target = mask_seq[:, :, 4:, :]

        context_input = context_latents.permute(0, 3, 1, 2)
        context_input = context_input + torch.randn_like(context_input) * 0.01

        predicted_latents = self.forward_with_control(context_input, conds=cond_vector, attention_mask=attention_mask, control=target_motion, index=mask_seq_target, force_mask=force_mask)
        predicted_latents = predicted_latents.permute(0, 2, 3, 1)

        loss = torch.nn.functional.mse_loss(predicted_latents, target_latents, reduction='none')
        loss = (loss * target_mask.unsqueeze(-1).unsqueeze(-1)).sum() / target_mask.sum()

        after_mean, after_std, train_mean, train_std = mean_std

        predicted_latents = (predicted_latents * after_std) + after_mean
        predicted_latents = predicted_latents.permute(0, 3, 1, 2)
        predicted_motion = ae.decode(predicted_latents)
        predicted_motion = predicted_motion * train_std + train_mean

        target_motion = target_motion.permute(0, 2, 3, 1)
        target_motion = target_motion * train_std + train_mean
        loss_control = torch.nn.functional.mse_loss(predicted_motion, target_motion, reduction='none')
        loss_control = loss_control.permute(0, 3, 1, 2)
        loss_control = (loss_control * mask_seq_target).sum() / mask_seq_target.sum()
        
        return loss, loss_control

#################################################################################
#                                     ACJEPA Zoos                                #
#################################################################################
def acjepa_raw_flow_s_ps22_control(**kwargs):
    layer = 8
    return ACJEPA_ControlNet(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1,
                 patch_size=(1, 22), stride_size=(1, 22), freeze_base=True, **kwargs)


ACJEPA_ControlNet_Models = {
    'ACJEPA-Flow-S-PatchSize22-ControlNet': acjepa_raw_flow_s_ps22_control,
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class c_control_embedder(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features,
            patch_size,
            stride_size,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_features, hidden_features, kernel_size=(4,patch_size[1]), stride=(4,stride_size[1]), bias=True)
        self.norm = LlamaRMSNorm(hidden_features, eps=1e-6)
        self.zero_linear = nn.Linear(hidden_features, hidden_features)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.zero_linear(x)
        return x