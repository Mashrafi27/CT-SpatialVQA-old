"""NV-Reason-CT 3D vision-language model implementation."""

import itertools
import warnings

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    Qwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5VisionConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model,
    Qwen3_5VisionPatchMerger,
)

from dynamic_network_architectures.architectures.primus import Primus


class Vision3D(PreTrainedModel):
    """3D vision encoder followed by the feature projection."""

    config_class = Qwen3_5VisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        """Initialize weights and rebuild non-persistent buffers."""
        super()._init_weights(module)

        # Primus rotary pos_embed is absent from state_dict and must be recomputed
        # after materialization to avoid uninitialized values.
        # https://github.com/huggingface/transformers/issues/43644

        if (
            hasattr(module, "_get_pos_embed_values")
            and hasattr(module, "get_embed")
            and getattr(module, "pos_embed", None) is not None
            and getattr(module, "feat_shape", None) is not None
        ):
            pe = module.pos_embed
            if pe.device.type == "meta":
                return
            fresh = module._get_pos_embed_values(
                feat_shape=module.feat_shape,
                device=pe.device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                pe.copy_(fresh.to(dtype=pe.dtype))
            return

        nps = getattr(module, "_non_persistent_buffers_set", None)
        if nps:
            unhandled = [
                n
                for n in nps
                if module._buffers.get(n) is not None
                and module._buffers[n].device.type != "meta"
            ]
            if unhandled:
                warnings.warn(
                    "[vlm3d] non-persistent buffer(s) were not reinitialized "
                    f"after Transformers meta-device loading in "
                    f"{type(module).__name__}: {unhandled}",
                    stacklevel=2,
                )

    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        input_shape=(192, 192, 192),
        patch_embed_size=(8, 8, 8),
    ):
        """Initialize the Primus backbone and Qwen3.5 vision merger."""
        super().__init__(config)

        # The 3D path never spatially merges tokens. The upstream 2D tower
        # retains its configured spatial merge size.
        self.spatial_merge_size = 1

        self.sub_vision = Primus(
            input_channels=1,
            num_classes=1,
            eva_depth=16,
            eva_numheads=12,
            embed_dim=864,
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            use_rot_pos_emb=True,
            use_abs_pos_embed=False,
            drop_path_rate=0.2,
            init_values=0.1,
            scale_attn_inner=True,
            num_register_tokens=0,
        )
        self.sub_vision.up_projection = nn.Identity()  # type: ignore

        primus_embed_dim = self.sub_vision.eva.embed_dim
        merger_cfg = Qwen3_5VisionConfig(
            hidden_size=primus_embed_dim,
            spatial_merge_size=1,
            out_hidden_size=config.out_hidden_size,
        )

        # Project Primus features into the language-model embedding space.
        self.merger = Qwen3_5VisionPatchMerger(merger_cfg)

    def forward(self, x, *args, **kwargs):
        """Encode 3D CT volumes into projected visual tokens."""
        x = self.sub_vision(x)  # [B, 864, T, H, W]
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, H, W, C]
        return self.merger(x.view(-1, x.shape[-1]))


class VLM3D_Model(Qwen3_5Model):
    """Qwen3.5 + 3D ViT."""

    _checkpoint_conversion_mapping = {}

    def __init__(self, config):
        """Initialize Qwen3.5 and attach the Primus 3D vision tower."""
        super().__init__(config)
        self.vision3d = Vision3D(
            config.vision_config,
            input_shape=getattr(config, "vit3d_input_shape", (192, 192, 192)),
            patch_embed_size=getattr(config, "vit3d_patch_embed_size", (8, 8, 8)),
        )
        # The parent initializes before vision3d exists. Run post_init again so
        # its merger and non-persistent Primus rotary buffers are initialized.
        self.post_init()

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        """Route 5D volumes to Primus and ordinary images to upstream Qwen3.5."""
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 5:
            return self._get_volume_features(pixel_values, image_grid_thw, **kwargs)
        return super().get_image_features(
            pixel_values, image_grid_thw=image_grid_thw, **kwargs
        )

    def _get_volume_features(self, pixels, grid_thw, **kwargs):
        """Encode 3D volumes and split flattened patch embeddings per input grid."""
        pixels = pixels.type(self.vision3d.dtype)
        embeds = self.vision3d(pixels, grid_thw=grid_thw)
        embeds = embeds.pooler_output if hasattr(embeds, "pooler_output") else embeds
        split_sizes = grid_thw.prod(-1).tolist()
        return BaseModelOutputWithPooling(pooler_output=torch.split(embeds, split_sizes))

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MRoPE positions using a merge size appropriate to each grid.

        This follows the upstream Qwen3.5 implementation, using merge size 1
        for 3D volume grids and the configured 2D merge size otherwise.
        """
        # Expand video grids per frame because MRoPE timestamps are frame-specific.
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        # Parent would do `spatial_merge_size = self.config.vision_config.spatial_merge_size`
        # here. We pick per-grid below instead.
        stock_sms = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # Modality IDs: text=0, image=1, video=2.
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    # Volumes (T>1) match the unmerged 3D processor grid;
                    # images and individual video frames use the configured size.
                    grid_sms = 1 if grid_thw[0] > 1 else stock_sms
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, grid_sms, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // grid_sms
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas


class VLM3D_ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 conditional generation wrapper with `VLM3D_Model`."""

    _checkpoint_conversion_mapping = {}

    def __init__(self, config):
        """Initialize conditional generation around ``VLM3D_Model``."""
        # Skip the stock conditional-generation constructor so it does not
        # create Qwen3_5Model; install VLM3D_Model below instead.
        super(Qwen3_5ForConditionalGeneration, self).__init__(config)
        self.model = VLM3D_Model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()
