import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
from functools import partial
from timm.models.vision_transformer import Attention
from models.ROPE import RopeND
from utils.train_utils import lengths_to_mask

#################################################################################
#                                      ACJEPA                                    #
#################################################################################
class ACJEPA(nn.Module):
    def __init__(self, input_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0, clip_dim=512,
                 diff_model='Flow', cond_drop_prob=0.1, max_length=49,
                 patch_size=(1, 22), stride_size=(1, 22), num_joint=22,
                 clip_version='ViT-B/32', **kargs):
        super(ACJEPA, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
            self.num_actions = kargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        # --------------------------------------------------------------------------
        # ACJEPA
        print('Loading ACJEPA...')
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patches_per_frame = (num_joint - patch_size[1]) // stride_size[1] + 1

        # Patchification
        self.x_embedder = nn.Conv2d(self.input_dim, self.latent_dim, kernel_size=self.patch_size, stride=self.stride_size, bias=True)

        # Positional Encoding
        max_length = max_length * self.patches_per_frame
        self.max_lens = [max_length]
        self.rope = RopeND(nd=1, nd_split=[1], max_lens=self.max_lens)
        self.position_ids_precompute = torch.arange(max_length).unsqueeze(0)

        self.ACJEPATransformer = nn.ModuleList([
            ACJEPATransBlock(self.latent_dim, num_heads, mlp_size=ff_size, rope=self.rope, qk_norm=True) for _ in range(num_layers)
        ])

        if self.cond_mode == 'text':
            self.y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.y_embedder = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.y_embedder = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.final_layer = FinalLayer(self.latent_dim, self.input_dim, patch_size=patch_size, stride_size=stride_size, patches=self.patches_per_frame, joint=num_joint)

        self.initialize_weights()

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in ACJEPA blocks:
        for block in self.ACJEPATransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)
        assert torch.cuda.is_available()
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, conds, attention_mask, force_mask=False, guidance_scale=None, position_offset = 0):

        conds = self.mask_cond(conds, force_mask=force_mask)
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        y = self.y_embedder(conds).unsqueeze(1)

        position_ids = self.position_ids_precompute[:, :x.shape[1]]
        for block in self.ACJEPATransformer:
            x = block(x, y, attention_mask, position_ids=position_ids)
        x = self.final_layer(x, y)

        return x

    def forward_with_CFG(self, x, conds, attention_mask, cfg=1.0):
        if not cfg == 1.0:
            half = x[: len(x) // 2]
            x = torch.cat([half, half], dim=0)
        x = self.forward(x, conds, attention_mask)
        if not cfg == 1.0:
            cond_eps, uncond_eps = torch.split(x, len(x) // 2, dim=0)
            half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
            x = torch.cat([half_eps, half_eps], dim=0)
        return x

    def forward_loss(self, latents, y, m_lens):
        latents = latents.permute(0, 2, 3, 1)
        b, l, j, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1).unsqueeze(-1), latents, torch.zeros_like(latents))

        context_latents = latents[:, :-1, :, :]  # Past/Current frames [B, L-1, J, d]
        target_latents  = latents[:, 1:, :, :].clone().detach() # Future frames [B, L-1, J, d]

        # Shift the padding masks to match the L-1 sequence length
        context_mask = non_pad_mask[:, :-1]  # Used for attention
        target_mask  = non_pad_mask[:, 1:]   # Used for loss calculation

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

        print(pad_mask.shape)

        N_tokens = (l - 1) * self.patches_per_frame
        causal_mask = torch.tril(torch.ones(N_tokens, N_tokens, device=device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        print(causal_mask.shape)

        attention_mask = pad_mask & causal_mask

        context_input = context_latents.permute(0, 3, 1, 2)
        context_input = context_input + torch.randn_like(context_input) * 0.01

        predicted_latents = self.forward(context_input, conds=cond_vector, attention_mask=attention_mask, force_mask=force_mask)
        predicted_latents = predicted_latents.permute(0, 2, 3, 1)

        loss = torch.nn.functional.mse_loss(predicted_latents, target_latents, reduction='none')

        print("Loss shape: ", loss.shape)
        loss = (loss * target_mask.unsqueeze(-1).unsqueeze(-1)).sum() / target_mask.sum()
        return loss

    @torch.no_grad()
    def generate(self,
                 conds,
                 m_lens,
                 cond_scale: int,
                 temperature=1,
                 j=22,
                 ):
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        noise = torch.randn(b, self.input_dim, l, j).to(device)
        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=0)
            noise = torch.cat([noise, noise], dim=0)

        attention_mask = (~padding_mask).unsqueeze(-1).repeat(1,1,self.patches_per_frame).flatten(1).unsqueeze(1).unsqueeze(1)
        model_kwargs = dict(conds=cond_vector, attention_mask=attention_mask, cfg=cond_scale)
        sample_fn = self.forward_with_CFG

        if not cond_scale == 1:
            model_kwargs["attention_mask"] = attention_mask.repeat(2, 1, 1, 1)

        if self.diff_model == "Flow":
            model_fn = self.gen_diffusion.sample_ode()  # default to ode sampling
            sampled_token_latent = model_fn(noise, sample_fn, **model_kwargs)[-1]
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs,
                progress=False,
                temperature=temperature
            )
        if not cond_scale == 1:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
        sampled_token_latent = sampled_token_latent.permute(0,2,3,1)

        latents = torch.where(padding_mask.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(sampled_token_latent), sampled_token_latent)
        return latents.permute(0,3,1,2)
    @torch.no_grad()
    def generate_rollout(self, seed_pose, conds, target_length=60, guidance_scale=1.0):
        """
        seed_pose: [B, d, 1, J] (Your starting frame)
        conds: Text/Action conditions
        """
        device = next(self.parameters()).device
        
        # 1. Encode text condition
        cond_vector = self.encode_text(conds)
        
        # Setup for Classifier-Free Guidance (if guidance_scale > 1.0)
        if guidance_scale > 1.0:
            uncond_vector = torch.zeros_like(cond_vector)
            # We process both conditional and unconditional together
            cond_vector = torch.cat([cond_vector, uncond_vector], dim=0)
            
        current_sequence = seed_pose # Start with length = 1
        
        # 2. The Rollout Loop
        for _ in range(target_length - 1):
            
            # If using CFG, duplicate the current sequence for the batch
            if guidance_scale > 1.0:
                model_input = torch.cat([current_sequence, current_sequence], dim=0)
            else:
                model_input = current_sequence

            seq_len = model_input.shape[2]
            N_infer_tokens = seq_len * self.patches_per_frame
            infer_causal_mask = torch.tril(torch.ones(N_infer_tokens, N_infer_tokens, device=device, dtype=torch.bool))
            infer_causal_mask = infer_causal_mask.unsqueeze(0).unsqueeze(0)
                
            # 3. Predict the next sequence of frames
            predicted_sequence = self.forward(model_input, conds=cond_vector, attention_mask=None, position_offset=0)
            
            # 4. Extract the LAST predicted frame 
            # (This is the new t+1 frame)
            next_frame = predicted_sequence[:, :, -1:, :]
            
            # 5. Apply Classifier-Free Guidance
            if guidance_scale > 1.0:
                cond_pred, uncond_pred = next_frame.chunk(2, dim=0)
                next_frame = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                
            # 6. Append the new frame to our sequence
            current_sequence = torch.cat([current_sequence, next_frame], dim=2) # Concat along sequence length
    
        # Return the full generated sequence
        return current_sequence
#################################################################################
#                                     ACJEPA Zoos                                #
#################################################################################
def acjepa_raw_flow_s_ps22(**kwargs):
    layer = 8
    return ACJEPA(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=2000,
                 patch_size=(1, 22), stride_size=(1, 22), **kwargs)
def acjepa_flow_s_ps22(**kwargs):
    layer = 8
    return ACJEPA(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=2000,
                 patch_size=(1, 22), stride_size=(1, 22), **kwargs)
def acjepa_flow_xl_ps2(**kwargs):
    layer = 20
    return ACJEPA(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=2000,
                 patch_size=(1, 2), stride_size=(1, 2), **kwargs)
def acjepa_mesh_flow_s_ps28(**kwargs):
    layer = 8
    return ACJEPA(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=2000, num_joint=28,
                 patch_size=(1, 28), stride_size=(1, 28), **kwargs)
ACJEPA_models = {
    'ACJEPA-Raw-Flow-S-PatchSize22': acjepa_raw_flow_s_ps22, 'ACJEPA-Flow-S-PatchSize22': acjepa_flow_s_ps22,
    'ACJEPA-Flow-XL-PatchSize2': acjepa_flow_xl_ps2, 'ACJEPA-Mesh-Flow-S-PatchSize28': acjepa_mesh_flow_s_ps28,
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ACJEPAAttention(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        rope=None,
        qk_norm=True,
        **block_kwargs,
    ):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, **block_kwargs)
        self.rope = rope

    def forward(self, x, position_ids=None, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop.p
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class ACJEPATransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, rope=None, qk_norm=True):
        super().__init__()
        self.norm1 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.attn = ACJEPAAttention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=LlamaRMSNorm,
                                        qk_norm=qk_norm, rope=rope)
        self.norm2 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask=None, position_ids=None):
        dtype = x.dtype
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        norm_x1 = self.norm1(x.to(torch.float32)).to(dtype)
        attn_input_x = modulate(norm_x1, shift_msa, scale_msa)
        attn_output_x = self.attn(attn_input_x, attention_mask=attention_mask, position_ids=position_ids)
        x = x + gate_msa * attn_output_x

        norm_x2 = self.norm2(x.to(torch.float32)).to(dtype)
        gate_input_x = modulate(norm_x2, shift_mlp, scale_mlp)
        gate_output_x = self.mlp(gate_input_x)
        x = x + gate_mlp * gate_output_x
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size, patch_size=(1, 22), stride_size=(1,22), patches=1, joint=22):
        super().__init__()
        self.norm_final = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patches = patches
        self.joint=joint
        self.linear = nn.Linear(hidden_size, output_size*patch_size[0]*patch_size[1], bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        norm_x = self.norm_final(x.to(torch.float32)).to(x.dtype)
        x = modulate(norm_x, shift, scale)
        x = self.linear(x)
        x = x.reshape(shape=(x.shape[0], x.shape[1]//self.patches, self.patches, self.patch_size[0], self.patch_size[1], x.shape[-1] // self.patch_size[1]))
        x = torch.einsum('nljpqc->nclpjq', x)
        x = x.reshape(shape=(x.shape[0], x.shape[1], -1, self.joint))
        return x


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)