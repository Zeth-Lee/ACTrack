# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import logging

import numpy as np
import torch
import torch.nn.functional as F
from sam3.model.memory import SimpleMaskEncoder
from sam3.model.sam3_tracker_utils import get_1d_sine_pe, select_closest_cond_frames
from sam3.model.utils.kalman_filter import KalmanFilter
from sam3.sam.mask_decoder import MaskDecoder, MLP
from sam3.sam.prompt_encoder import PromptEncoder
from sam3.sam.transformer import TwoWayTransformer
from sam3.train.data.collator import BatchedDatapoint

try:
    from timm.layers import trunc_normal_
except ModuleNotFoundError:

    from timm.models.layers import trunc_normal_


NO_OBJ_SCORE = -1024.0


class Sam3TrackerBase(torch.nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        maskmem_backbone,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,


        max_cond_frames_in_attn=-1,

        keep_first_cond_frame=False,

        multimask_output_in_sam=False,


        multimask_min_pt_num=1,
        multimask_max_pt_num=1,

        multimask_output_for_tracking=False,


        forward_backbone_per_frame_for_eval=False,


        memory_temporal_stride_for_eval=1,


        offload_output_to_cpu_for_eval=False,


        trim_past_non_cond_mem_for_eval=False,

        non_overlap_masks_for_mem_enc=False,

        max_obj_ptrs_in_encoder=16,

        sam_mask_decoder_extra_args=None,

        compile_all_components=False,

        use_memory_selection=False,

        mf_threshold=0.01,
        use_motion_prior: bool = True,
        motion_prior_weight: float = 0.15,
        motion_prior_stable_frames_threshold: int = 15,
        motion_prior_stable_iou_threshold: float = 0.3,
        motion_prior_kf_score_threshold: float = 0.0,
        min_obj_score_logits: float =0.0,
        use_prototype_memory_selection: bool = False,
        prototype_memory_alpha: float = 0.5,
        prototype_memory_iou_threshold: float = 0.5,
        prototype_memory_obj_score_threshold: float = 0.0,
        prototype_memory_kf_score_threshold: float = 0.0,
        prototype_memory_eff_iou_weight: float = 0.0,
        prototype_memory_kf_weight: float = 0.0,
    ):
        super().__init__()


        self.backbone = backbone
        self.num_feature_levels = 3
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder


        self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)


        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim = transformer.d_model


        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):

            self.mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem


        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)


        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)


        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval


        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking


        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4


        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
        self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
        trunc_normal_(self.no_obj_ptr, std=0.02)
        self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame


        self.use_memory_selection = use_memory_selection
        self.mf_threshold = mf_threshold
        self.use_motion_prior = use_motion_prior
        self.motion_prior_weight = float(motion_prior_weight)
        self.motion_prior_stable_frames_threshold = int(motion_prior_stable_frames_threshold)
        self.motion_prior_stable_iou_threshold = float(motion_prior_stable_iou_threshold)
        self.motion_prior_kf_score_threshold = float(motion_prior_kf_score_threshold)
        self.min_obj_score_logits = float(min_obj_score_logits)
        self.use_prototype_memory_selection = bool(use_prototype_memory_selection)
        self.prototype_memory_alpha = float(prototype_memory_alpha)
        self.prototype_memory_iou_threshold = float(prototype_memory_iou_threshold)
        self.prototype_memory_obj_score_threshold = float(prototype_memory_obj_score_threshold)
        self.prototype_memory_kf_score_threshold = float(prototype_memory_kf_score_threshold)
        self.prototype_memory_eff_iou_weight = float(prototype_memory_eff_iou_weight)
        self.prototype_memory_kf_weight = float(prototype_memory_kf_weight)
        self._motion_kf = KalmanFilter() if self.use_motion_prior else None


        self.compile_all_components = compile_all_components
        if self.compile_all_components:
            self._compile_all_components()

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None, dummy=False):
        if dummy:
            return torch.zeros(len(rel_pos_list), self.mem_dim, device=device)

        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = (
            torch.tensor(rel_pos_list).pin_memory().to(device=device, non_blocking=True)
            / t_diff_max
        )
        tpos_dim = self.hidden_dim
        pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        pos_enc = self.obj_ptr_tpos_proj(pos_enc)

        return pos_enc

    def _build_sam_heads(self):

        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride


        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)


        self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        gt_masks=None,
        output_dict=None,
        frame_idx=None,
    ):

        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size


        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:

            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)


        if mask_inputs is not None:


            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:


            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )


        sparse_embeddings = self._maybe_clone(sparse_embeddings)
        dense_embeddings = self._maybe_clone(dense_embeddings)
        image_pe = self._maybe_clone(self.sam_prompt_encoder.get_dense_pe())
        with torch.profiler.record_function("sam_mask_decoder"):
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,
                high_res_features=high_res_features,
            )


        low_res_multimasks = self._maybe_clone(low_res_multimasks)
        ious = self._maybe_clone(ious)
        sam_output_tokens = self._maybe_clone(sam_output_tokens)
        object_score_logits = self._maybe_clone(object_score_logits)

        if self.training and self.teacher_force_obj_scores_for_mem:


            is_obj_appearing = torch.any(gt_masks.float().flatten(1) > 0, dim=1)
            is_obj_appearing = is_obj_appearing[..., None]
        else:
            is_obj_appearing = object_score_logits > self.min_obj_score_logits


        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )


        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        kf_score = None
        if multimask_output:
            batch_inds = torch.arange(B, device=device)
            if (
                (not self.training)
                and self.use_motion_prior
                and self._motion_kf is not None
                and output_dict is not None
                and frame_idx is not None
                and ious.size(1) > 1
            ):
                state = output_dict.setdefault("motion_prior", {})
                means = state.setdefault("means", [])
                covariances = state.setdefault("covariances", [])
                stable_frames = state.setdefault("stable_frames", [])
                last_frame_idx = state.setdefault("last_frame_idx", [])

                while len(means) < B:
                    means.append(None)
                while len(covariances) < B:
                    covariances.append(None)
                while len(stable_frames) < B:
                    stable_frames.append(0)
                while len(last_frame_idx) < B:
                    last_frame_idx.append(None)

                cand_boxes = self._masks_to_boxes_xyxy(high_res_multimasks > 0.0)
                kf_ious = torch.zeros_like(ious)
                use_weighted = torch.zeros(B, device=device, dtype=torch.bool)

                for b in range(B):
                    if not bool(is_obj_appearing[b].item()):
                        means[b] = None
                        covariances[b] = None
                        stable_frames[b] = 0
                        last_frame_idx[b] = int(frame_idx)
                        continue

                    mean = means[b]
                    cov = covariances[b]
                    if (
                        mean is not None
                        and cov is not None
                        and last_frame_idx[b] != int(frame_idx)
                    ):
                        mean, cov = self._motion_kf.predict(mean, cov)

                    if (
                        mean is not None
                        and cov is not None
                        and int(stable_frames[b]) >= self.motion_prior_stable_frames_threshold
                    ):
                        obs_boxes = cand_boxes[b].tolist()
                        kf_iou_list = self._motion_kf.compute_iou(mean[:4], obs_boxes)
                        kf_ious[b] = torch.tensor(
                            kf_iou_list, device=device, dtype=ious.dtype
                        )
                        use_weighted[b] = True

                    means[b] = mean
                    covariances[b] = cov
                    last_frame_idx[b] = int(frame_idx)

                w = float(self.motion_prior_weight)
                if w > 0:
                    ious_for_select = torch.where(
                        use_weighted[:, None], (1 - w) * ious + w * kf_ious, ious
                    )
                else:
                    ious_for_select = ious

                best_iou_inds = torch.argmax(ious_for_select, dim=-1)

                kf_score_per_obj = torch.zeros(B, device=device, dtype=ious.dtype)
                if use_weighted.any():
                    kf_score_per_obj = kf_ious[batch_inds, best_iou_inds]

                for b in range(B):
                    if not bool(is_obj_appearing[b].item()):
                        continue

                    obs_box = cand_boxes[b, best_iou_inds[b]].tolist()
                    if obs_box[2] <= obs_box[0] or obs_box[3] <= obs_box[1]:
                        means[b] = None
                        covariances[b] = None
                        stable_frames[b] = 0
                        continue

                    measurement = np.asarray(
                        self._motion_kf.xyxy_to_xyah(obs_box), dtype=np.float64
                    )
                    mean = means[b]
                    cov = covariances[b]

                    if mean is None or cov is None or int(stable_frames[b]) == 0:
                        mean, cov = self._motion_kf.initiate(measurement)
                        stable_frames[b] = 1
                    elif int(stable_frames[b]) < self.motion_prior_stable_frames_threshold:
                        if (
                            float(ious[b, best_iou_inds[b]].item())
                            > self.motion_prior_stable_iou_threshold
                        ):
                            mean, cov = self._motion_kf.update(mean, cov, measurement)
                            stable_frames[b] = int(stable_frames[b]) + 1
                        else:
                            mean, cov = None, None
                            stable_frames[b] = 0
                    else:
                        if (
                            float(ious[b, best_iou_inds[b]].item())
                            < self.motion_prior_stable_iou_threshold
                        ):
                            mean, cov = None, None
                            stable_frames[b] = 0
                        else:
                            mean, cov = self._motion_kf.update(mean, cov, measurement)

                    means[b] = mean
                    covariances[b] = cov

                ious = ious_for_select
                kf_score = kf_score_per_obj

            else:
                best_iou_inds = torch.argmax(ious, dim=-1)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks


        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()

        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            kf_score,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):


        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(
                high_res_masks.size(-2) // self.backbone_stride * 4,
                high_res_masks.size(-1) // self.backbone_stride * 4,
            ),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )

        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()

        _, _, _, _, _, obj_ptr, _, _= self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
            gt_masks=mask_inputs,
        )


        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            None,
        )

    def forward(self, input: BatchedDatapoint, is_inference=False):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM3VideoPredictor for inference."
            "See examples/sam3_dense_video_tracking.ipynb for an inference example."
        )

    def forward_image(self, img_batch):


        backbone_out = self.backbone.forward_image(img_batch)["sam2_backbone_out"]


        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = self._maybe_clone(
                backbone_out["backbone_fpn"][i]
            )
            backbone_out["vision_pos_enc"][i] = self._maybe_clone(
                backbone_out["vision_pos_enc"][i]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):

        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):


        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None


        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)


        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def cal_mem_score(self, object_score_logits, iou_score):
        object_score_norm = torch.where(
            object_score_logits > self.min_obj_score_logits,
            object_score_logits.sigmoid() * 2 - 1,
            torch.zeros_like(object_score_logits),
        )
        score_per_frame = (object_score_norm * iou_score).mean()
        return score_per_frame

    def _masks_to_boxes_xyxy(self, masks: torch.Tensor) -> torch.Tensor:
        masks = masks.to(torch.bool)
        B, _, H, W = masks.shape
        x_any = masks.any(dim=-2)
        y_any = masks.any(dim=-1)
        empty = ~masks.flatten(2).any(dim=-1)

        x0 = x_any.float().argmax(dim=-1)
        x1 = (W - 1) - x_any.flip(-1).float().argmax(dim=-1)
        y0 = y_any.float().argmax(dim=-1)
        y1 = (H - 1) - y_any.flip(-1).float().argmax(dim=-1)

        x0 = x0.masked_fill(empty, 0)
        y0 = y0.masked_fill(empty, 0)
        x2 = (x1 + 1).masked_fill(empty, 0)
        y2 = (y1 + 1).masked_fill(empty, 0)

        return torch.stack([x0, y0, x2, y2], dim=-1).to(torch.float32)

    def _compute_fg_prototype(
            self, feat_bchw: torch.Tensor, mask_logits_b1hw: torch.Tensor
    ) -> torch.Tensor:
        mask_logits = F.interpolate(
            mask_logits_b1hw,
            size=feat_bchw.shape[-2:],
            mode="bilinear", 
            align_corners=False,
        )
        mask = torch.sigmoid(mask_logits)
        mask = (mask >= 0.5).to(dtype=feat_bchw.dtype)

        masked_feat = feat_bchw * mask
        feat_h, feat_w = masked_feat.shape[-2:]
        area = (
            F.avg_pool2d(mask, kernel_size=(feat_h, feat_w)) * float(feat_h * feat_w)
            + 0.0005
        )
        pooled = (
            F.avg_pool2d(masked_feat, kernel_size=(feat_h, feat_w))
            * float(feat_h * feat_w)
            / area
        )
        return pooled

    def frame_filter(self, output_dict, track_in_reverse, frame_idx, num_frames, r):
        if (frame_idx == 0 and not track_in_reverse) or (
            frame_idx == num_frames - 1 and track_in_reverse
        ):
            return []

        target_non_cond_frames = max(0, int(self.num_maskmem) - 1)
        max_candidate_frames = min(
            num_frames,
            max(int(self.max_obj_ptrs_in_encoder), 30, int(target_non_cond_frames * 5)),
        )

        if not track_in_reverse:
            start = frame_idx - 1
            end = 0
            step = -r
            must_include = frame_idx - 1
        else:
            start = frame_idx + 1
            end = num_frames
            step = r
            must_include = frame_idx + 1

        valid_indices = []
        for i in range(start, end, step):
            out_i = output_dict["non_cond_frame_outputs"].get(i, None)
            if out_i is None:
                continue
            if (
                self.use_motion_prior
                and self._motion_kf is not None
                and "kf_score" in out_i
            ):
                kf_score = out_i["kf_score"]
                if (
                    kf_score is not None
                    and float(kf_score.item()) <= self.motion_prior_kf_score_threshold
                ):
                    continue

            if self.use_prototype_memory_selection:
                iou_score = out_i.get("iou_score", None)
                obj_score = out_i.get("object_score_logits", None)
                if iou_score is None or obj_score is None:
                    continue

                iou_val = (
                    float(iou_score.mean().item())
                    if hasattr(iou_score, "mean")
                    else float(iou_score)
                )
                obj_val = (
                    float(obj_score.mean().item())
                    if hasattr(obj_score, "mean")
                    else float(obj_score)
                )
                if iou_val <= self.prototype_memory_iou_threshold:
                    continue
                if obj_val <= self.prototype_memory_obj_score_threshold:
                    continue
                kf_score = out_i.get("kf_score", None)
                if kf_score is not None and float(kf_score.item()) <= self.prototype_memory_kf_score_threshold:
                    continue

                valid_indices.insert(0, i)
            else:
                score_per_frame = out_i.get("eff_iou_score", None)
                if score_per_frame is None:
                    continue
                if score_per_frame > self.mf_threshold:
                    valid_indices.insert(0, i)

            if len(valid_indices) >= max_candidate_frames - 1:
                break

        if must_include not in valid_indices:
            valid_indices.append(must_include)

        if not self.use_prototype_memory_selection:
            return valid_indices

        if len(valid_indices) <= target_non_cond_frames:
            return valid_indices

        ref_last = self._get_pro_fg(output_dict, must_include)
        if ref_last is None:
            return valid_indices

        cond_frames = output_dict.get("cond_frame_outputs", {})
        if cond_frames:
            first_cond_idx = min(cond_frames.keys())
            ref_first = self._get_pro_fg(output_dict, first_cond_idx)
        else:
            ref_first = None

        if ref_first is None:
            ref_first = ref_last

        device = ref_last.device
        ref_last = ref_last.to(device)
        ref_first = ref_first.to(device)

        alpha = float(self.prototype_memory_alpha)
        eff_w = float(self.prototype_memory_eff_iou_weight)
        kf_w = float(self.prototype_memory_kf_weight)
        keep_k = max(0, target_non_cond_frames - 1)
        scored = []
        for idx in valid_indices:
            if idx == must_include:
                continue
            pro = self._get_pro_fg(output_dict, idx)
            if pro is None:
                continue
            pro = pro.to(device)
            pro_vec = pro.squeeze(-1).squeeze(-1) if pro.dim() == 4 else pro
            ref_last_vec = (
                ref_last.squeeze(-1).squeeze(-1) if ref_last.dim() == 4 else ref_last
            )
            ref_first_vec = (
                ref_first.squeeze(-1).squeeze(-1) if ref_first.dim() == 4 else ref_first
            )
            sim_last = F.cosine_similarity(pro_vec, ref_last_vec, dim=-1).mean()
            sim_first = F.cosine_similarity(pro_vec, ref_first_vec, dim=-1).mean()
            score = alpha * sim_last + (1.0 - alpha) * sim_first

            if eff_w != 0.0:
                eff = output_dict["non_cond_frame_outputs"][idx].get("eff_iou_score", None)
                if eff is not None:
                    eff_val = float(eff.item()) if hasattr(eff, "item") else float(eff)
                    score = score + eff_w * eff_val

            if kf_w != 0.0:
                kf = output_dict["non_cond_frame_outputs"][idx].get("kf_score", None)
                if kf is not None:
                    kf_val = float(kf.item()) if hasattr(kf, "item") else float(kf)
                    score = score + kf_w * kf_val

            scored.append((float(score.item()), idx))

        if len(scored) < keep_k:
            return valid_indices

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = [idx for _, idx in scored[:keep_k]]
        keep.append(must_include)
        keep.sort(reverse=bool(track_in_reverse))
        if keep[-1] != must_include:
            keep = [x for x in keep if x != must_include] + [must_include]

        return keep

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):

        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device


        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        if not is_init_cond_frame and use_prev_mem_frame:

            to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []


            assert len(output_dict["cond_frame_outputs"]) > 0

            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx,
                cond_outputs,
                self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )
            t_pos_and_prevs = [
                ((frame_idx - t) * tpos_sign_mul, out, True)
                for t, out in selected_cond_outputs.items()
            ]


            r = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )

            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if self.use_memory_selection:
                    if t_rel > len(valid_indices):
                        continue
                    prev_frame_idx = valid_indices[-t_rel]
                else:
                    if t_rel == 1:

                        if not track_in_reverse:

                            prev_frame_idx = frame_idx - t_rel
                        else:

                            prev_frame_idx = frame_idx + t_rel
                    else:

                        if not track_in_reverse:


                            prev_frame_idx = ((frame_idx - 2) // r) * r

                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                        else:


                            prev_frame_idx = -(-(frame_idx + 2) // r) * r

                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * r

                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:


                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out, False))

            for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
                if prev is None:
                    continue


                feats = prev["maskmem_features"].cuda(non_blocking=True)
                seq_len = feats.shape[-2] * feats.shape[-1]
                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_prompt_mask.append(
                    torch.zeros(B, seq_len, device=device, dtype=bool)
                )

                maskmem_enc = prev["maskmem_pos_enc"][-1].cuda()
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                if (
                    is_selected_cond_frame
                    and getattr(self, "cond_frame_spatial_embedding", None) is not None
                ):

                    maskmem_enc = maskmem_enc + self.cond_frame_spatial_embedding


                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                )
                to_cat_prompt_pos_embed.append(maskmem_enc)


            if (
                self.training
                and self.prob_to_dropout_spatial_mem > 0
                and self.rng.random() < self.prob_to_dropout_spatial_mem
            ):
                num_spatial_mem_keep = self.rng.integers(len(to_cat_prompt) + 1)
                keep = self.rng.choice(
                    range(len(to_cat_prompt)), num_spatial_mem_keep, replace=False
                ).tolist()
                to_cat_prompt = [to_cat_prompt[i] for i in keep]
                to_cat_prompt_mask = [to_cat_prompt_mask[i] for i in keep]
                to_cat_prompt_pos_embed = [to_cat_prompt_pos_embed[i] for i in keep]

            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)


            if not self.training:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs
            pos_and_ptrs = [

                (
                    (frame_idx - t) * tpos_sign_mul,
                    out["obj_ptr"],
                    True,
                )
                for t, out in ptr_cond_outputs.items()
            ]


            for t_diff in range(1, max_obj_ptrs_in_encoder):
                if not self.use_memory_selection:
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                else:
                    if -t_diff <= -len(valid_indices):
                        break
                    t = valid_indices[-t_diff]

                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"], False))


            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, is_selected_cond_frame_list = zip(*pos_and_ptrs)

                obj_ptrs = torch.stack(ptrs_list, dim=0)
                if getattr(self, "cond_frame_obj_ptr_embedding", None) is not None:
                    obj_ptrs = (
                        obj_ptrs
                        + self.cond_frame_obj_ptr_embedding
                        * torch.tensor(is_selected_cond_frame_list, device=device)[
                            ..., None, None
                        ].float()
                    )


                obj_pos = self._get_tpos_enc(
                    pos_list,
                    max_abs_pos=max_obj_ptrs_in_encoder,
                    device=device,
                )

                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                if self.mem_dim < C:

                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_mask.append(None)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:

            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem


            to_cat_prompt = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_prompt_mask = [torch.zeros(B, 1, device=device, dtype=bool)]
            to_cat_prompt_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]


        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)
        encoder_out = self.transformer.encoder(
            src=current_vision_feats,
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds,
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        output_dict=None,
        is_init_cond_frame=False,
    ):

        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]

        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:


            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )

        if is_mask_from_pts and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:

            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
            pix_feat = pix_feat.view_as(pix_feat)
            maskmem_out = self.maskmem_backbone(
                pix_feat, mask_for_mem, skip_mask_sigmoid=True
            )
        else:
            maskmem_out = self.maskmem_backbone(image, pix_feat, mask_for_mem)

        maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
        maskmem_pos_enc = [self._maybe_clone(m) for m in maskmem_out["vision_pos_enc"]]


        is_obj_appearing = (object_score_logits > self.min_obj_score_logits).float()
        maskmem_features += (
            1 - is_obj_appearing[..., None, None]
        ) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_pos_enc

    def forward_tracking(self, backbone_out, input, return_dict=False):

        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:


            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)


        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]


        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  
            "non_cond_frame_outputs": {},  
        }
        for stage_id in processing_order:

            img_ids = input.find_inputs[stage_id].img_ids
            if img_feats_already_computed:

                current_image = input.img_batch[img_ids]
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:


                (
                    current_image,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(input.img_batch, img_ids)

            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                image=current_image,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                output_dict=output_dict,
                num_frames=num_frames,
            )

            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict

        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]

        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        image,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,


        run_mem_encoder=True,

        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        base_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if mask_inputs is not None:

            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:

            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )


            if prev_sam_mask_logits is not None:
                assert self.iter_use_prev_mask_pred
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                output_dict=output_dict,
                frame_idx=int(frame_idx),
            )
        (
            _,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            kf_score,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if self.use_prototype_memory_selection:
            current_out["pro_fg"] = self._compute_fg_prototype(base_feat, low_res_masks)
        if self.use_memory_selection:
            current_out["object_score_logits"] = object_score_logits
            iou_score = ious.max(-1)[0]
            current_out["iou_score"] = iou_score
            current_out["eff_iou_score"] = self.cal_mem_score(object_score_logits, iou_score)
            if kf_score is not None:
                current_out["kf_score"] = kf_score.mean()
        if not self.training:


            current_out["object_score_logits"] = object_score_logits


        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                output_dict=output_dict,
                is_init_cond_frame=is_init_cond_frame,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None


        if self.offload_output_to_cpu_for_eval and not self.training:

            trimmed_out = {
                "pred_masks": current_out["pred_masks"].cpu(),
                "pred_masks_high_res": current_out["pred_masks_high_res"].cpu(),

                "obj_ptr": current_out["obj_ptr"],
                "object_score_logits": current_out["object_score_logits"],
            }
            if "pro_fg" in current_out:
                trimmed_out["pro_fg"] = current_out["pro_fg"]
            if run_mem_encoder and self.num_maskmem > 0:
                trimmed_out["maskmem_features"] = maskmem_features.cpu()
                trimmed_out["maskmem_pos_enc"] = [x.cpu() for x in maskmem_pos_enc]
            if self.use_memory_selection:
                trimmed_out["iou_score"] = current_out["iou_score"].cpu()
                trimmed_out["eff_iou_score"] = current_out["eff_iou_score"].cpu()
                if "kf_score" in current_out:
                    trimmed_out["kf_score"] = current_out["kf_score"].cpu()
            current_out = trimmed_out


        def _trim_past_out(past_out, current_out):
            if past_out is None:
                return None
            trimmed = {
                "pred_masks": past_out["pred_masks"],
                "obj_ptr": past_out["obj_ptr"],
                "object_score_logits": past_out["object_score_logits"],
            }
            if "iou_score" in past_out:
                trimmed["iou_score"] = past_out["iou_score"]
            if "eff_iou_score" in past_out:
                trimmed["eff_iou_score"] = past_out["eff_iou_score"]
            if "kf_score" in past_out:
                trimmed["kf_score"] = past_out["kf_score"]
            if "pro_fg" in past_out:
                trimmed["pro_fg"] = past_out["pro_fg"]
            return trimmed

        if self.trim_past_non_cond_mem_for_eval and not self.training:
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx, None)

            if past_out is not None:
                print(past_out.get("eff_iou_score", 0))
                if (
                    self.use_memory_selection
                    and past_out.get("eff_iou_score", 0) < self.mf_threshold
                ) or not self.use_memory_selection:
                    output_dict["non_cond_frame_outputs"][past_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

            if (
                self.use_memory_selection and not self.offload_output_to_cpu_for_eval
            ):
                far_old_frame_idx = frame_idx - 20 * self.max_obj_ptrs_in_encoder
                past_out = output_dict["non_cond_frame_outputs"].get(
                    far_old_frame_idx, None
                )
                if past_out is not None:
                    output_dict["non_cond_frame_outputs"][far_old_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):

        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):

        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device

        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)

        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds


        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _compile_all_components(self):


        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        from sam3.perflib.compile import compile_wrapper

        logging.info("Compiling all components. First time may be very slow.")

        self.maskmem_backbone.forward = compile_wrapper(
            self.maskmem_backbone.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        self.transformer.encoder.forward = compile_wrapper(
            self.transformer.encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,
        )


        self.sam_mask_decoder.forward = compile_wrapper(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    def _maybe_clone(self, x):

        return x.clone() if self.compile_all_components else x


def concat_points(old_point_inputs, new_points, new_labels):

    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}