# Custom nodes for running Anima
# Some of the code was copied and modified from the following two projects:
#   https://github.com/KohakuBlueleaf/HDM-ext
#   https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter

import os
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3Model, Qwen2Tokenizer

# Comfy
import folder_paths
import comfy
import comfy.utils
from comfy import supported_models_base
import comfy.sd1_clip as sd1_clip
from comfy.text_encoders.cosmos import T5XXLTokenizer


@contextmanager
def operation_patch(operations):
    """
    Directly patch torch.nn so we don't need to re-impl the model arch
    Not recommended but easiest
    """
    module_list = set(i for i in dir(torch.nn) if not i.startswith("_"))
    for op in dir(operations):
        if op in module_list:
            setattr(torch.nn, f"org_{op}", getattr(torch.nn, op))
            setattr(torch.nn, op, getattr(operations, op))
    yield
    for op in dir(operations):
        if op in module_list:
            setattr(torch.nn, op, getattr(torch.nn, f"org_{op}"))


class Qwen3_600MTokenizer(sd1_clip.SDTokenizer):
    """
    Tokenizer Class for Qwen3 with TI embedding support
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "qwen3_tokenizer"
        )
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_size=1024,  # We need this for TI embedding
            embedding_key="qwen3",
            tokenizer_class=Qwen2Tokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=512,
            pad_token=151643,
            tokenizer_data=tokenizer_data,
        )


class CosmosQwenTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen3 = Qwen3_600MTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = {}
        out["qwen3"] = self.qwen3.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.qwen3.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class Qwen3_600M_Wrapper(Qwen3Model):
    """
    Qwen3 Model wrapper to support ComfyUI's API
    """

    def forward(
        self,
        input_tokens,
        attention_mask=None,
        embeds=None,
        num_tokens=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
        **kwargs
    ):
        # We use Autocast here to resolve dtype issue easily
        # In theory you may want to manually set dtype instead of autocast for performance
        with torch.autocast(self.device.type, dtype=self.dtype):
            result = (
                super()
                .forward(input_tokens, attention_mask, inputs_embeds=embeds)
                .last_hidden_state
            )
        # we only have last layer output, and we don't have pooled emb as well
        return result, None


def Qwen3_600M(config, dtype, device, operations):
    """
    Qwen3 Model Class, we init model in normal mode + operation patch
    num_layers is needed in ComfyUI's Node
    """
    with torch.inference_mode(False), operation_patch(operations):
        model = (
            # TODO: load from selectable path
            Qwen3_600M_Wrapper.from_pretrained(
                config['model_path'], dtype=dtype, attn_implementation="sdpa"
            )
            .to(device)
        )
        model = model.to(device).eval().requires_grad_(False)
        model.num_layers = model.config.num_hidden_layers
    return model


class Qwen3_600MModel(sd1_clip.SDClipModel):
    def __init__(
        self,
        device="cpu",
        layer="last",
        layer_idx=None,
        dtype=None,
        attention_mask=True,
        textmodel_json_config={},
        model_options={},
    ):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=textmodel_json_config,
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=Qwen3_600M,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            zero_out_masked=attention_mask,
            model_options=model_options,
        )


class QwenWithAdapter(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, textmodel_json_config={}):
        super().__init__(
            device=device,
            dtype=dtype,
            name="qwen3_600m",  # This name will be used to find TE in state_dict
            clip_model=Qwen3_600MModel,
            model_options=model_options,
            textmodel_json_config=textmodel_json_config,
        )
        if textmodel_json_config.get('use_llm_adapter', False):
            self.llm_adapter = LLMAdapter(
                source_dim=1024,
                target_dim=1024,
                model_dim=1024,
                num_layers=6,
                use_self_attn=True,
                device='cuda',  # TODO: figure out how to get this on the right device automatically
                dtype=torch.bfloat16,  # TODO: same here
                operations=getattr(self, self.clip).operations,
            )
        else:
            self.llm_adapter = None

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_qwen3 = token_weight_pairs["qwen3"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]
        result = self.qwen3_600m.encode_token_weights(token_weight_pairs_qwen3)
        if self.llm_adapter is not None:
            hidden_states = result[0].cuda()
            qwen_attn_mask = result[2]['attention_mask'].cuda()
            t5_input_ids = torch.tensor([[t[0] for t in entry] for entry in token_weight_pairs_t5], dtype=torch.int, device=hidden_states.device)
            t5_attn_mask = t5_input_ids.bool().int()  # 0 is pad ID so this works
            with torch.autocast(hidden_states.device.type, dtype=torch.bfloat16):
                hidden_states = self.llm_adapter(hidden_states, t5_input_ids, target_attention_mask=t5_attn_mask, source_attention_mask=qwen_attn_mask)
                hidden_states[~t5_attn_mask.bool()] = 0
            result = (hidden_states, None, {'attention_mask': t5_attn_mask})
        return result

    def load_sd(self, sd):
        if 'llm_adapter.out_proj.weight' in sd:
            new_sd = {k.replace('llm_adapter.', ''): v for k, v in sd.items()}
            return self.llm_adapter.load_state_dict(new_sd)
        else:
            return super().load_sd(sd)


# class Qwen3Loader:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {"model_name": (get_llm_checkpoints(), {"default": get_llm_checkpoints()[0] if get_llm_checkpoints() else None})},
#             "optional": {"device": (["default", "cpu"], {"advanced": True})}
#         }
#     RETURN_TYPES = ("CLIP",)
#     FUNCTION = "load_clip"

#     CATEGORY = "advanced/loaders"

#     def load_clip(self, model_name, type="stable_diffusion", device="default"):
#         model_options = {}
#         if device == "cpu":
#             model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

#         clip_target = supported_models_base.ClipTarget(CosmosQwenTokenizer, QwenWithAdapter)
#         clip_target.params['textmodel_json_config'] = {'model_path': get_llm_checkpoint_path(model_name)}
#         parameters = 600_000_000
#         embedding_directory = folder_paths.get_folder_paths("embeddings")
#         clip = comfy.sd.CLIP(
#             clip_target,
#             embedding_directory=embedding_directory,
#             parameters=parameters,
#             # TODO: make dtype configurable?
#             model_options={'dtype': torch.bfloat16},
#         )

#         return (clip,)

# --------------------------------------------------------------------------------------------------------
# LLM Adapter model definition

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None, operations=None):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        self.k_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        self.v_proj = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)

        self.o_proj = operations.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def init_weights(self):
        torch.nn.init.zeros_(self.o_proj.weight)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False, layer_norm=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            self.self_attn = Attention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim//num_heads,
                device=device,
                dtype=dtype,
                operations=operations,
            )

        self.norm_cross_attn = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim//num_heads,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        self.norm_mlp = operations.LayerNorm(model_dim, device=device, dtype=dtype) if layer_norm else operations.RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            operations.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype)
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)
        self.cross_attn.init_weights()


class LLMAdapter(nn.Module):
    def __init__(
            self,
            source_dim,
            target_dim,
            model_dim,
            num_layers=6,
            num_heads=16,
            use_self_attn=False,
            layer_norm=False,
            device=None,
            dtype=None,
            operations=None,
        ):
        super().__init__()

        self.embed = operations.Embedding(32128, target_dim, device=device, dtype=dtype)
        if model_dim != target_dim:
            self.in_proj = operations.Linear(target_dim, model_dim, device=device, dtype=dtype)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim//num_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn, layer_norm=layer_norm, device=device, dtype=dtype, operations=operations) for _ in range(num_layers)
        ])
        self.out_proj = operations.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = operations.RMSNorm(target_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))

# --------------------------------------------------

def get_adapters_dict():
    """
    Get the dictionary of LLM adapters.
    Keys are the names of the LLM adapters, values are the paths to the LLM adapters.
    """
    adapters_dict = {}
    if "llm_adapters" in folder_paths.folder_names_and_paths:
        adapters_paths, _ = folder_paths.folder_names_and_paths["llm_adapters"]
    else:
        adapters_paths = [os.path.join(folder_paths.models_dir, "llm_adapters")]

    for adapters_path in adapters_paths:
        if os.path.exists(adapters_path):
            for item in os.listdir(adapters_path):
                if item.endswith('.safetensors'):
                    adapters_dict[item] = os.path.join(adapters_path, item)

    return adapters_dict


def get_llm_adapters():
    """
    Get the list of available LLM adapters.
    """
    ret = list(get_adapters_dict().keys())
    ret.sort()
    return ret


def get_llm_adapter_path(adapter_name):
    """
    Get the path to an LLM adapter.
    """
    adapters_dict = get_adapters_dict()

    if adapter_name in adapters_dict:
        return adapters_dict[adapter_name]
    else:
        raise ValueError(f"Adapter {adapter_name} not found")


def get_llm_dict():
    """
    Get the dictionary of LLM checkpoints.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """
    llm_dict = {}
    if "llm" in folder_paths.folder_names_and_paths:
        llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
    elif os.path.exists(os.path.join(folder_paths.models_dir, "llm")):
        llm_paths = [os.path.join(folder_paths.models_dir, "llm")]
    else:
        llm_paths = [os.path.join(folder_paths.models_dir, "LLM")]

    for llm_path in llm_paths:
        if os.path.exists(llm_path):
            for item in os.listdir(llm_path):
                item_path = os.path.join(llm_path, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid model directory (contains config.json or similar)
                    if any(f in os.listdir(item_path) for f in ['config.json', 'model.safetensors', 'pytorch_model.bin']):
                        llm_dict[item] = item_path
                elif item.endswith(('.safetensors', '.bin', '.pt')):
                    llm_dict[item] = item_path

    return llm_dict


def get_llm_checkpoints():
    """
    Get the list of available LLM checkpoints.
    """
    return list(get_llm_dict().keys())


def get_llm_checkpoint_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")


class AnimaCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "llm_name": (get_llm_checkpoints(), ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
            },
            "optional": {
                "llm_adapter_name": (['None'] + get_llm_adapters(), {
                    "default": 'None', "tooltip": "Override LLM adapter weights. Advanced option intended for testing."
                }),
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_checkpoint"

    CATEGORY = "advanced/loaders"

    def load_checkpoint(self, checkpoint_name, llm_name, weight_dtype, llm_adapter_name='None'):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        checkpoint_path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_name)
        full_sd = comfy.utils.load_torch_file(checkpoint_path)

        llm_adapter_sd, dit_sd = {}, {}
        for k, v in full_sd.items():
            if 'llm_adapter' in k:
                llm_adapter_sd[k.replace('net.', '')] = v
            else:
                dit_sd[k] = v

        if llm_adapter_name != 'None':
            llm_adapter_path = get_llm_adapter_path(llm_adapter_name)
            llm_adapter_sd = comfy.utils.load_torch_file(llm_adapter_path)
            llm_adapter_sd = {'llm_adapter.'+k: v for k, v in llm_adapter_sd.items()}

        model = comfy.sd.load_diffusion_model_state_dict(dit_sd, model_options=model_options)

        clip_target = supported_models_base.ClipTarget(CosmosQwenTokenizer, QwenWithAdapter)
        clip_target.params['textmodel_json_config'] = {'model_path': get_llm_checkpoint_path(llm_name), 'use_llm_adapter': True}
        parameters = 600_000_000
        embedding_directory = folder_paths.get_folder_paths("embeddings")
        clip = comfy.sd.CLIP(
            clip_target,
            embedding_directory=embedding_directory,
            parameters=parameters,
            # TODO: make dtype configurable?
            model_options={'dtype': torch.bfloat16},
        )
        clip.load_sd(llm_adapter_sd)

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "AnimaCheckpointLoader": AnimaCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaCheckpointLoader": "Anima Checkpoint Loader",
}
