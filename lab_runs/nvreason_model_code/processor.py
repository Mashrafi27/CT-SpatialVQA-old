"""NIfTI CT-volume preprocessing and multimodal inputs for NV-Reason-CT."""

import math
import os
import warnings

import numpy as np
import torch
from scipy import ndimage
from transformers import BaseImageProcessor, BatchFeature
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor, Qwen3VLProcessorKwargs

warnings.filterwarnings("ignore", category=FutureWarning, module=r"monai\.utils\.deprecate_utils")

from monai.transforms import (  # noqa: E402
    CenterSpatialCropd,
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Transposed,
)


class AnatomySpatialCropd(CenterSpatialCropd):
    """Crop an ROI around chest or abdomen bounds.

    Lung bounds are estimated from enclosed-air components after LPS
    reorientation. If reliable bounds cannot be found, cropping falls back to
    a centered x/y crop at the superior z edge.

    Example:
        Crop an LPS-oriented, 2-mm, channel-first 4D CT tensor in HU with shape
        ``(C, H, W, D)``:

        >>> crop = AnatomySpatialCropd(keys=["image"], roi_size=(192, 192, 192), pixdim=(2.0, 2.0, 2.0))
        >>> chest = crop({"image": volume, "anatomy_region": "chest"})["image"]
        >>> abdomen = crop({"image": volume, "anatomy_region": "abdomen"})["image"]
    """

    VALID_REGIONS = {"chest", "abdomen"}

    def __init__(
        self,
        keys,
        roi_size,
        pixdim=(2.0, 2.0, 2.0),
        body_hu=-500.0,
        min_frac=0.10,
        min_lung_volume_ml=250.0,
        component_merge_gap_mm=40.0,
        max_chest_span_mm=500.0,
        abdomen_overlap_above_lung_base_mm=100.0,
        abdomen_length_mm=300.0,
        allow_missing_keys=True,
    ):
        """Initialize crop geometry and anatomy-detection thresholds."""
        super().__init__(keys=keys, roi_size=roi_size, allow_missing_keys=allow_missing_keys)
        self.roi_size = self._normalize_roi_size(roi_size)
        self.pixdim = self._normalize_pixdim(pixdim)
        self.body_hu = float(body_hu)
        self.min_frac = float(min_frac)
        self.min_lung_volume_ml = float(min_lung_volume_ml)
        self.component_merge_gap_mm = float(component_merge_gap_mm)
        self.max_chest_span_mm = float(max_chest_span_mm)
        self.abdomen_overlap_above_lung_base_mm = float(abdomen_overlap_above_lung_base_mm)
        self.abdomen_length_mm = float(abdomen_length_mm)

    @staticmethod
    def _normalize_roi_size(roi_size):
        """Normalize an ROI size to three positive integer dimensions."""
        if isinstance(roi_size, (int, np.integer)):
            out = (int(roi_size),) * 3
        else:
            out = tuple(int(x) for x in roi_size)
        if len(out) != 3 or any(x <= 0 for x in out):
            raise ValueError(f"roi_size must contain three positive integers, got {roi_size!r}")
        return out

    @staticmethod
    def _normalize_pixdim(pixdim):
        """Normalize voxel spacing to three positive floating-point values."""
        if isinstance(pixdim, (int, float, np.integer, np.floating)):
            out = (float(pixdim),) * 3
        else:
            out = tuple(float(x) for x in pixdim[:3])
        if len(out) != 3 or any(x <= 0 for x in out):
            raise ValueError(f"pixdim must contain three positive values, got {pixdim!r}")
        return out

    @classmethod
    def resolve_region(cls, anatomy_region):
        """Validate and return the requested anatomy region."""
        try:
            if anatomy_region in {None, *cls.VALID_REGIONS}:
                return anatomy_region
        except TypeError:
            pass
        raise ValueError("anatomy_region must be None, 'chest', or 'abdomen'")

    @staticmethod
    def _as_numpy(image):
        """Convert a channel-first tensor or array to a 3D NumPy array."""
        if isinstance(image, torch.Tensor):
            data = image.detach().cpu().float().numpy()
        else:
            data = np.asarray(image, dtype=np.float32)
        if data.ndim == 4:
            data = data[0]
        elif data.ndim != 3:
            raise ValueError(
                "AnatomySpatialCropd expects channel-first 4D or spatial "
                f"3D data, got shape {image.shape}"
            )
        return data

    @staticmethod
    def _component_summaries(labels, component_ids, sizes):
        """Summarize component voxel counts and z-axis extents."""
        summaries = []
        for component_id in component_ids:
            zs = np.where(labels == component_id)[2]
            if len(zs) == 0:
                continue
            summaries.append((int(component_id), int(sizes[component_id]), int(zs.min()), int(zs.max())))
        return summaries

    def _select_superior_component_cluster(self, labels, component_ids, sizes, z_spacing_mm):
        """Select a superior cluster of enclosed-air components."""
        summaries = self._component_summaries(labels, component_ids, sizes)
        if not summaries:
            return np.array([], dtype=int)
        summaries.sort(key=lambda item: (item[3], item[1]), reverse=True)

        # Orientationd("LPS") makes larger z indices superior. Start from the
        # superior large enclosed-air component and only merge nearby pieces so
        # abdominal bowel/stomach gas does not pull the crop inferiorly.
        selected_ids = [summaries[0][0]]
        selected_z_min = summaries[0][2]
        selected_z_max = summaries[0][3]
        for component_id, _size, z_min, z_max in summaries[1:]:
            gap_mm = max(0, selected_z_min - z_max - 1) * z_spacing_mm
            if gap_mm > self.component_merge_gap_mm:
                continue
            next_z_min = min(selected_z_min, z_min)
            next_span_mm = (selected_z_max - next_z_min + 1) * z_spacing_mm
            if next_span_mm > self.max_chest_span_mm:
                continue
            selected_ids.append(component_id)
            selected_z_min = next_z_min
            selected_z_max = max(selected_z_max, z_max)
        return np.array(selected_ids, dtype=int)

    def _detect_lung_bounds(self, image):
        """Estimate body and lung z-axis bounds from CT intensities."""
        data = self._as_numpy(image)
        body = data > self.body_hu
        z_has_body = body.any(axis=(0, 1))
        body_zs = np.where(z_has_body)[0]
        if len(body_zs) == 0:
            return None

        air = np.empty(body.shape, dtype=bool)
        for z in range(data.shape[2]):
            body_slice = body[:, :, z]
            filled = ndimage.binary_fill_holes(body_slice)
            air[:, :, z] = filled & ~body_slice

        labels, components = ndimage.label(air)
        if components == 0:
            return {
                "body_z_min": int(body_zs.min()),
                "lung_z_min": None,
                "lung_z_max": None,
            }

        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest_component_voxels = int(sizes.max())
        if largest_component_voxels <= 0:
            return {
                "body_z_min": int(body_zs.min()),
                "lung_z_min": None,
                "lung_z_max": None,
            }

        kept_ids = np.where(sizes >= self.min_frac * largest_component_voxels)[0]
        kept_ids = kept_ids[kept_ids != 0]
        selected_ids = self._select_superior_component_cluster(labels, kept_ids, sizes, self.pixdim[2])
        if len(selected_ids) == 0:
            return {
                "body_z_min": int(body_zs.min()),
                "lung_z_min": None,
                "lung_z_max": None,
            }

        lung = np.isin(labels, selected_ids)
        z_has_lung = lung.any(axis=(0, 1))
        lung_zs = np.where(z_has_lung)[0]
        if len(lung_zs) == 0:
            lung_z_min = None
            lung_z_max = None
        else:
            lung_z_min = int(lung_zs.min())
            lung_z_max = int(lung_zs.max())
            max_chest_slices = max(1, int(math.ceil(self.max_chest_span_mm / self.pixdim[2])))
            capped_z_min = max(lung_z_min, lung_z_max - max_chest_slices + 1)
            if capped_z_min > lung_z_min:
                lung[:, :, :capped_z_min] = False
                z_has_lung = lung.any(axis=(0, 1))
                lung_zs = np.where(z_has_lung)[0]
                lung_z_min = int(lung_zs.min()) if len(lung_zs) else None
                lung_z_max = int(lung_zs.max()) if len(lung_zs) else None

        lung_volume_ml = float(lung.sum()) * self.pixdim[0] * self.pixdim[1] * self.pixdim[2] / 1000.0
        if lung_volume_ml < self.min_lung_volume_ml:
            lung_z_min = None
            lung_z_max = None

        return {
            "body_z_min": int(body_zs.min()),
            "lung_z_min": lung_z_min,
            "lung_z_max": lung_z_max,
        }

    def _region_bounds(self, image, region):
        """Return z-axis bounds for the requested chest or abdomen crop."""
        bounds = self._detect_lung_bounds(image)
        if not bounds or bounds["lung_z_min"] is None or bounds["lung_z_max"] is None:
            return None

        if region == "chest":
            return bounds["lung_z_min"], bounds["lung_z_max"]

        z_size = int(image.shape[-1])
        overlap_slices = int(math.ceil(self.abdomen_overlap_above_lung_base_mm / self.pixdim[2]))
        length_slices = int(math.ceil(self.abdomen_length_mm / self.pixdim[2]))
        z_max = min(z_size - 1, int(bounds["lung_z_min"]) + overlap_slices)
        z_min = max(int(bounds["body_z_min"]), z_max - length_slices + 1)
        if z_min > z_max:
            return None
        return z_min, z_max

    def _axis_start(self, size, roi, center):
        """Compute a clamped crop start index centered on one axis."""
        start = int(round(float(center) - roi / 2.0))
        start = max(0, min(start, size - roi))
        return start

    def _crop_one(self, image, region):
        """Crop one volume to the configured ROI for an anatomy region."""
        spatial_shape = tuple(int(x) for x in image.shape[-3:])
        for dim, roi in zip(spatial_shape, self.roi_size):
            if dim < roi:
                raise ValueError(
                    f"AnatomySpatialCropd input spatial shape {spatial_shape} is smaller than roi_size {self.roi_size}"
                )

        bounds = self._region_bounds(image, region)
        x_start = self._axis_start(spatial_shape[0], self.roi_size[0], (spatial_shape[0] - 1) / 2.0)
        y_start = self._axis_start(spatial_shape[1], self.roi_size[1], (spatial_shape[1] - 1) / 2.0)
        if bounds is None:
            filename = getattr(image, "meta", {}).get("filename_or_obj", "<unknown>")
            warnings.warn(
                f"AnatomySpatialCropd could not determine {region} z bounds for {filename}; "
                "using a centered x/y crop at the superior z edge.",
                RuntimeWarning,
                stacklevel=2,
            )
            z_start = spatial_shape[2] - self.roi_size[2]
        else:
            z_min, z_max = bounds
            z_center = (float(z_min) + float(z_max)) / 2.0
            z_start = self._axis_start(spatial_shape[2], self.roi_size[2], z_center)

        slices = (
            slice(None),
            slice(x_start, x_start + self.roi_size[0]),
            slice(y_start, y_start + self.roi_size[1]),
            slice(z_start, z_start + self.roi_size[2]),
        )
        if len(image.shape) == 3:
            slices = slices[1:]
        return image[slices]

    def __call__(self, data, lazy=None):
        """Apply anatomy-aware cropping to the configured dictionary keys."""
        d = dict(data)
        region = self.resolve_region(d.get("anatomy_region"))
        if region is None:
            return super().__call__(d, lazy=lazy)
        for key in self.key_iterator(d):
            d[key] = self._crop_one(d[key], region)
        return d


class ImageLoader3D(BaseImageProcessor):
    """Load and preprocess NIfTI CT volumes for the 3D vision tower.

    Volumes are reoriented to LPS, resampled to ``pixdim``, padded and cropped
    to ``spatial_size``, and normalized from Hounsfield units by default.
    ``normalize_mode="ct"`` clips values to [-1000, 1000] and scales them to
    [-1, 1]. Use ``0`` or ``"none"`` to leave intensities unchanged for
    custom normalization.

    Example:
        Load the same NIfTI image with chest and abdomen crops:

        >>> loader = ImageLoader3D(pixdim=(2.0, 2.0, 2.0), spatial_size=(192, 192, 192))
        >>> chest = loader.load_image("scan.nii.gz", normalize_mode="ct", anatomy_region="chest")
        >>> abdomen = loader.load_image("scan.nii.gz", normalize_mode="ct", anatomy_region="abdomen")

    Notes:
        If ``anatomy_region`` is not specified, a centered crop is used; it may
        not include the desired anatomy. Set ``normalize_mode`` to ``0`` or
        ``"none"`` to leave intensities unchanged for custom normalization.
    """

    def __init__(
        self,
        pixdim=(2.0, 2.0, 2.0),
        spatial_size=(192, 192, 192),
        final_grid_size=(24, 24, 24),
        merge_size=1,
        normalize_mode="ct",
        **kwargs,
    ):
        """Configure CT resampling, cropping, grid, and normalization."""
        super().__init__(**kwargs)
        self.pixdim = list(pixdim)
        self.spatial_size = list(spatial_size)
        self.final_grid_size = list(final_grid_size)
        self.merge_size = merge_size
        self.normalize_mode = normalize_mode
        self.keys = ["image"]
        self.transforms = None

    @staticmethod
    def _resolve_normalize_mode(normalize_mode):
        """Map normalization aliases to raw-HU mode 0 or scaled mode 1."""
        if normalize_mode is None:
            return 0
        if isinstance(normalize_mode, str):
            mode = normalize_mode.strip().lower()
            if mode in {"", "0", "none", "raw", "off"}:
                return 0
            if mode in {"1", "ct", "ct_hu", "hu"}:
                return 1
        if isinstance(normalize_mode, (int, np.integer)) and not isinstance(normalize_mode, bool):
            mode = int(normalize_mode)
            if mode in {0, 1}:
                return mode
        raise ValueError(
            f"Invalid normalize_mode={normalize_mode!r}; expected 0/raw or 1/ct"
        )

    @classmethod
    def _normalize_modes_for_batch(cls, normalize_mode, batch_size):
        """Expand and validate normalization modes for a volume batch."""
        if isinstance(normalize_mode, (list, tuple)):
            if len(normalize_mode) != batch_size:
                raise ValueError(
                    f"normalize_mode list length ({len(normalize_mode)}) must match "
                    f"number of images ({batch_size})"
                )
            return [cls._resolve_normalize_mode(mode) for mode in normalize_mode]
        return [cls._resolve_normalize_mode(normalize_mode)] * batch_size

    @staticmethod
    def _anatomy_regions_for_batch(anatomy_region, batch_size):
        """Expand and validate anatomy regions for a volume batch."""
        if isinstance(anatomy_region, (list, tuple)):
            if len(anatomy_region) != batch_size:
                raise ValueError(
                    f"anatomy_region list length ({len(anatomy_region)}) must match "
                    f"number of images ({batch_size})"
                )
            return [AnatomySpatialCropd.resolve_region(region) for region in anatomy_region]

        region = AnatomySpatialCropd.resolve_region(anatomy_region)
        return [region] * batch_size

    @staticmethod
    def _normalize_ct_hu(image):
        """Clip HU values to [-1000, 1000] and scale them to [-1, 1]."""
        image = torch.nan_to_num(image.float(), nan=-1000.0, posinf=1000.0, neginf=-1000.0)
        return torch.clamp(image, min=-1000.0, max=1000.0) / 1000.0

    def _apply_normalization(self, image, normalize_mode):
        """Apply the requested CT normalization mode to one volume."""
        mode = self._resolve_normalize_mode(normalize_mode)
        if mode == 0:
            return image
        if mode == 1:
            return self._normalize_ct_hu(image)
        raise AssertionError(f"unreachable normalize_mode={mode}")

    def load_image(self, filename, normalize_mode=None, anatomy_region=None):
        """Load one volume using the configured normalization by default."""
        anatomy_region = AnatomySpatialCropd.resolve_region(anatomy_region)
        if normalize_mode is None:
            normalize_mode = self.normalize_mode
        if self.transforms is None:
            self.transforms = self.get_transforms()
        image = self.transforms({"image": filename, "anatomy_region": anatomy_region})["image"]
        image = image.as_subclass(torch.Tensor).contiguous()
        return self._apply_normalization(image, normalize_mode)

    def get_transforms(self):
        """Build the deterministic MONAI volume-preprocessing pipeline."""
        keys = self.keys
        pixdim = self.pixdim
        spatial_size = self.spatial_size
        transforms = []
        transforms.append(
            LoadImaged(
                keys=keys,
                ensure_channel_first=True,
                dtype=None,
                allow_missing_keys=True,
                image_only=True,
            )
        )
        transforms.append(EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float, allow_missing_keys=True))
        # The crop heuristic assumes LPS orientation, where larger z indices are superior.
        transforms.append(Orientationd(keys=keys, axcodes="LPS"))

        transforms.append(Spacingd(
            keys=keys, pixdim=pixdim, mode=("bilinear"), dtype=torch.float,
            min_pixdim=np.array(pixdim) * 0.9, max_pixdim=np.array(pixdim) * 1.1,
            allow_missing_keys=True,
        ))

        transforms.append(SpatialPadd(keys=keys, spatial_size=spatial_size, value=-1000))
        transforms.append(AnatomySpatialCropd(keys=keys, roi_size=spatial_size, pixdim=pixdim))

        transforms.append(Transposed(keys=keys, indices=(0, 3, 2, 1)))
        transforms.append(ToTensord(keys=keys))
        return Compose(transforms)

    def __call__(self, images, *, normalize_mode=None, anatomy_region=None, **kwargs) -> BatchFeature:
        """Preprocess volume paths into model-ready batch features."""
        return self.preprocess(images, normalize_mode=normalize_mode, anatomy_region=anatomy_region, **kwargs)

    def preprocess(self, images, *, normalize_mode=None, anatomy_region=None, **kwargs) -> BatchFeature:
        """Load a volume batch and return pixel tensors with grid metadata."""
        if self.transforms is None:
            self.transforms = self.get_transforms()

        if normalize_mode is None:
            normalize_mode = self.normalize_mode
        normalize_modes = self._normalize_modes_for_batch(normalize_mode, len(images))
        anatomy_regions = self._anatomy_regions_for_batch(anatomy_region, len(images))

        images_list = []
        image_grid_thws = []
        for filename, image_normalize_mode, image_anatomy_region in zip(images, normalize_modes, anatomy_regions):
            images_list.append(
                self.load_image(
                    filename,
                    normalize_mode=image_normalize_mode,
                    anatomy_region=image_anatomy_region,
                )
            )
            image_grid_thws.append(torch.tensor(self.final_grid_size))

        pixel_values = torch.stack(images_list, dim=0)
        image_grid_thws = torch.stack(image_grid_thws, dim=0)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thws}

    def to_dict(self):
        """Serialize configuration without the runtime MONAI pipeline."""
        # The runtime-only MONAI pipeline is not JSON-serializable and is rebuilt lazily after loading.
        output = super().to_dict()
        output.pop("transforms", None)
        return output

    def save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs):
        """Save configuration after discarding the runtime MONAI pipeline."""
        self.transforms = None
        return super().save_pretrained(save_directory, push_to_hub, **kwargs)


class VLM3D_Processor(Qwen3VLProcessor):
    """Qwen3.5 tokenizer/chat template plus the NV-Reason-CT 3D loader.

    CT volumes must be supplied with `images3d=`. The inherited stock image
    and video processors are retained only for Transformers reconstruction.

    Example:
        Prepare text and a chest CT volume with the public processor interface:

        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("nvidia/NV-Reason-CT", trust_remote_code=True)
        >>> messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this CT."}]}]
        >>> prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=prompt, images3d=["scan.nii.gz"], anatomy_region="chest", return_tensors="pt")
    """

    @classmethod
    def get_attributes(cls):
        """Return subcomponents handled by the inherited processor loader."""
        # image_processor_3d is reconstructed explicitly in from_pretrained.
        return ["image_processor", "tokenizer", "video_processor"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        image_processor_3d=None,
        chat_template=None,
        **kwargs,
    ):
        """Initialize tokenizer, standard processors, and the 3D loader."""
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.image_processor_3d = image_processor_3d

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load the composite processor and reconstruct its 3D loader."""
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor_3d_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "force_download", "local_files_only", "token", "revision")
            if key in kwargs
        }
        processor.image_processor_3d = ImageLoader3D.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_processor_3d",
            **image_processor_3d_kwargs,
        )
        return processor

    def save_pretrained(self, save_directory, **kwargs):
        """Save the composite processor and dedicated 3D loader metadata."""
        saved_files = super().save_pretrained(save_directory, **kwargs)
        if self.image_processor_3d is not None:
            self.image_processor_3d.save_pretrained(
                os.path.join(save_directory, "image_processor_3d")
            )
        return saved_files

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        images3d=None,
        *,
        normalize_mode=None,
        anatomy_region=None,
        **kwargs,
    ):
        """Tokenize text and optionally preprocess NIfTI volumes from ``images3d``.

        Normalization modes and anatomy regions may be scalars or lists aligned
        with the volume batch. Volume inputs add ``pixel_values`` and
        ``image_grid_thw`` to the returned batch.
        """
        if images is not None or videos is not None:
            raise ValueError(
                "NV-Reason-CT accepts NIfTI CT volumes through `images3d=`; "
                "ordinary `images=` and `videos=` inputs are not supported."
            )
        if normalize_mode is not None and images3d is None:
            raise ValueError("`normalize_mode` is only supported with `images3d`")
        if anatomy_region is not None and images3d is None:
            raise ValueError("`anatomy_region` is only supported with `images3d`")

        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images3d is not None:
            images3d_kwargs = output_kwargs["images_kwargs"].copy()
            image_inputs = self.image_processor_3d(
                images=images3d,
                normalize_mode=normalize_mode,
                anatomy_region=anatomy_region,
                **images3d_kwargs,
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            image_merge_size = self.image_processor_3d.merge_size
        else:
            image_inputs = {}
            image_grid_thw = None
            image_merge_size = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if image_grid_thw is not None:
            merge_length = image_merge_size**2
            index = 0
            # Expand each image marker to the visual-token count represented
            # by its grid.
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)
