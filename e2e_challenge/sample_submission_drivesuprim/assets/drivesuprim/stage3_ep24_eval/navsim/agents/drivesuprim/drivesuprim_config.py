from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_config import TransfuserConfig


@dataclass
class InferConfig:
    model: str = "teacher"  # teacher or student
    use_first_stage_traj_in_infer: bool = False
    save_pickle: bool = False


@dataclass
class EgoPerturbConfig:
    n_student_rotation_ensemble: int = 3
    offline_aug_angle_boundary: float = 0
    offline_aug_file: str = '???'


@dataclass
class RefinementConfig:
    use_multi_stage: bool = False
    refinement_approach: str = "transformer_decoder"
    num_refinement_stage: int = 1  # 2
    stage_layers: str = "3"  # "3" or "3+3" ...
    topks: str = "256"  # "256" or "256+64" ...
    use_mid_output: bool = True
    use_separate_stage_heads: bool = True
    use_imi_learning_in_refinement: bool = True
    # Learned positional embedding on the refinement stage's key/value tokens.
    #
    # The first stage adds one (_keyval_embedding); refinement never did. That is
    # inherited from the ORIGINAL DriveSuprim, where the tokens are patches of a
    # stitched perspective panorama and their content already distinguishes them.
    # On a BEV grid it distinguishes much less: cell (12,30) and cell (12,31) of
    # empty road carry near-identical features, so without a position signal the
    # decoder cannot tell which metric cell a token came from except by content.
    #
    # SafeDrive adds a LearnedPositionalEncoding to the BEV map immediately before
    # flattening in EVERY transformer -- proposal, refinement and its drivable
    # head alike (navsim/agents/safedrive/modules/transformer_motion.py:87-89).
    #
    # Shape matches the first stage's embedding: one vector per token, d_model
    # wide, so bev_h*bev_w x tf_d_model (3136 x 256 = 803K params at 56x56).
    # A row/col-factorised form as in SafeDrive would be ~14K instead, but the
    # flat table is what this codebase already uses and trains fine at stage 1.
    #
    # Default OFF: this adds parameters and changes what the refinement decoder
    # sees, so every existing checkpoint and run stays reproducible.
    keyval_pos_embed: bool = False


@dataclass
class DriveSuprimConfig(TransfuserConfig):
    ckpt_path: str = None

    seq_len: int = 2
    # Global multiplier on every planning term (imitation + the eight PDM
    # metrics + the refinement stages). Stage 1 of the two-stage schedule sets
    # this to 0 so only the perception heads are trained; stage 2 restores it to
    # 1 and instead scales the perception weights down. Implemented as a
    # multiply-by-zero rather than a skip so the trajectory parameters keep
    # receiving (zero) gradients and DDP does not report unused parameters.
    planning_loss_weight: float = 1.0
    # Stage 2 of the three-stage schedule: train planning on a fixed perception.
    # Sets requires_grad=False and eval mode on the BEV front-end and both
    # auxiliary heads (see DriveSuprimAgent.PERCEPTION_MODULES), matching EAD's
    # phase-2 freeze_perception. Loss weights do NOT change between stages --
    # EAD controls the schedule with these two switches alone.
    freeze_perception: bool = False
    # LR schedule. 'none' = constant LR (what this repo did before, and what the
    # existing 10-epoch runs used). 'cos' = EAD's WarmupCosLR: linear warmup over
    # scheduler_warm_epoch, then cosine down to scheduler_min_lr across
    # scheduler_epoch epochs. EAD uses cos / 30 epochs / 1 warmup / min_lr 1e-6.
    # scheduler_epoch is the length of the COSINE, not the length of the run.
    # EAD leaves it at 30 for all three phases while running them for 20 / 5 / 30
    # epochs, so phases 1 and 2 stop part-way down the curve on purpose (at 20/30
    # the LR is still 0.25 of peak, at 5/30 it is 0.93). Do not "fix" it to match
    # max_epochs -- see configure_optimizers in drivesuprim_agent.py.
    scheduler_type: str = 'none'        # 'none' | 'cos'
    scheduler_epoch: int = 30
    scheduler_warm_epoch: int = 1
    scheduler_min_lr: float = 1e-6
    # ITERATION-based warmup, as BEVFormer and UniAD both use
    # (warmup='linear', warmup_iters=500, warmup_ratio=1/3): the LR ramps from
    # lr * warmup_ratio to lr over the first N optimizer steps. EAD's warmup is
    # epoch-based, so its default of 1 warmup epoch means no ramp at all -- the
    # first epoch already runs at full LR. 0 disables.
    scheduler_warmup_iters: int = 0
    scheduler_warmup_ratio: float = 1.0 / 3
    # Optimizer. 'adam' is what this repo used; 'adamw' with weight_decay 0.01 is
    # what BEVFormer and UniAD use. Note weight_decay applies to the default
    # group only -- the backbone groups already carry `backbone_wd`.
    optimizer_type: str = 'adam'        # 'adam' | 'adamw'
    weight_decay: float = 0.0
    trajectory_imi_weight: float = 1.0
    # NuRec does not provide a reliable per-frame traffic-light phase.  Keep the
    # prediction head for checkpoint compatibility, but allow its supervision
    # and trajectory-ranking contribution to be disabled for that dataset.
    use_traffic_light_compliance: bool = True
    # Annotated, so it is a real dataclass field: an agent YAML can override it.
    # Without the annotation this was a bare class attribute, hydra could not
    # instantiate the config with one, and the only way to set it was the
    # assignment DriveSuprimAgent used to make -- which overwrote every YAML.
    trajectory_pdm_weight: Dict[str, float] = field(default_factory=lambda: {
        'no_at_fault_collisions': 3.0,
        'drivable_area_compliance': 3.0,
        'time_to_collision_within_bound': 4.0,
        'ego_progress': 2.0,
        'driving_direction_compliance': 1.0,
        'lane_keeping': 2.0,
        'traffic_light_compliance': 3.0,
        'history_comfort': 1.0,
    })

    # Which sub-score heads exist, and how the ranking score is assembled from
    # them.  The label these are trained against decides all three: a head for a
    # term the label does not carry learns noise, and a term the score does not
    # multiply cannot change a ranking.  Defaults reproduce NAVSIM's EPDMS
    # exactly; the NuRec agent overrides them to NC x DAC x GT x EP.
    #
    #   pdm_heads         one sigmoid head per name, plus 'imi'
    #   pdm_score_log     w * log(sigmoid(x)), the multiplicative terms
    #   pdm_score_sum     the weighted sum inside a single log
    #   pdm_score_sum_scale  outer coefficient on that log
    # Legacy checkpoints steer off the four-way driving command rather than the
    # route. NuRec logs a command, but it reads straight on 85% of frames where
    # NAVSIM reads 62%, so it under-signals every manoeuvre a NAVSIM-trained
    # model expects to see. Derive it from the route instead, on frames that
    # have one. Only consulted when use_route is off -- with the route on, the
    # command slots are zeroed anyway.
    derive_command_from_route: bool = True
    # Keep the four command slots populated even with the route on. The default
    # zeroes them (the route replaced the command as the intent signal), which
    # leaves those columns of _status_encoding without a gradient for the whole
    # run. True feeds both: the command as a coarse per-frame intent on the ego
    # status, the route as the geometry each candidate reads for itself.
    use_command_with_route: bool = False

    pdm_heads: Tuple[str, ...] = (
        'no_at_fault_collisions',
        'drivable_area_compliance',
        'time_to_collision_within_bound',
        'ego_progress',
        'driving_direction_compliance',
        'lane_keeping',
        'traffic_light_compliance',
        'history_comfort',
    )
    pdm_score_log: Dict[str, float] = field(default_factory=lambda: {
        'no_at_fault_collisions': 0.5,
        'drivable_area_compliance': 0.5,
        'driving_direction_compliance': 0.3,
    })
    pdm_score_sum: Dict[str, float] = field(default_factory=lambda: {
        'time_to_collision_within_bound': 5.0,
        'ego_progress': 5.0,
        'lane_keeping': 2.0,
        'history_comfort': 1.0,
    })
    pdm_score_sum_scale: float = 6.0
    # An extra head predicting the aggregate score itself, supervised against
    # the label's own `pdm_score` column. Borrowed from SafeDrive
    # (`pdm_score_loss_weight` / `pdm_score_test_weight`): composing the ranking
    # from components alone puts the whole burden on hand-set log coefficients,
    # while this head learns the composition from the label. It enters the
    # ranking as a plain sigmoid bonus rather than a log term -- it is already a
    # score in [0, 1], not a factor that should be able to veto.
    pdm_imi_rank_weight: float = 0.02
    # Refinement-stage override for the imitation weight. None -> refinement
    # reuses pdm_imi_rank_weight, which is what shipped, so nothing moves until
    # the YAML sets it. After the coarse cut the product of the three gates
    # spans ~0.18 while normalised imi spans exactly 1.0, so at equal weight the
    # imitation term decides the refinement argmax on 98.9% of frames.
    pdm_imi_rank_weight_refine: Optional[float] = None
    # Refinement-stage override for the per-term exponents. None -> refinement
    # reuses pdm_rank_product_exponents, which is what shipped. Without this the
    # two selectors are forced to share one exponent set, so an exponent cannot
    # be attributed to the stage it acted on.
    pdm_rank_product_exponents_refine: Optional[Dict[str, float]] = None
    pdm_aggregate_head: bool = False
    pdm_aggregate_loss_weight: float = 1.0
    pdm_aggregate_rank_weight: float = 1.0
    # Rank candidates by the product the label is actually made of --
    # NC x DAC x GT x EP -- instead of the weighted sum of logs above.
    #
    # The two are the same function when every weight is 1, and behave nothing
    # alike anywhere else. Summing w*log s(x) lets a confident term buy back a
    # doubtful one: at the shipped weights a proposal the collision head scores
    # at 0.005 loses 0.5*log(0.005) = -2.6, which the other five terms cover, and
    # measured on plan-only epoch 3 that is 34.2% of the surviving 256 at a
    # standstill -- candidates the simulator scores a hard zero for collision.
    # A product cannot be bought back: one factor near zero takes the whole
    # score with it, which is what the label does.
    #
    # Applies to BOTH stages. `_rank_score` is called once for the coarse
    # 4096 -> 256 cut and again inside RefineTrajHead for the 256 -> 1, so the
    # refinement never sees what the product already rejected.
    pdm_rank_product: bool = False
    # Which heads multiply in. Defaults to the four the NuRec label multiplies;
    # a name with no head is skipped, as in the sum path.
    pdm_rank_product_terms: Tuple[str, ...] = (
        'no_at_fault_collisions',
        'drivable_area_compliance',
        'ego_progress',
    )
    # Per-factor exponent, applied as s(x)**e before the product -- the product's
    # analogue of a weight, since prod s_i**e_i = exp(sum e_i log s_i). Left at
    # 1.0 the ranking is the label's own composition with nothing added.
    pdm_rank_product_exponents: Dict[str, float] = field(default_factory=dict)
    # Divide softmax(imi) by its own maximum before it enters the ranking, so its
    # best candidate always scores 1.0.
    #
    # The four gates are per-candidate: every one of the 4,096 may read 0.99 at
    # once. imi is a distribution over the vocabulary and sums to 1, so what its
    # best candidate gets depends on how peaked the frame is -- measured across
    # 600 val frames the maximum runs 0.16 at the median and 0.99 at worst, and
    # `pdm_imi_rank_weight` multiplies that. At 5.0 the term is worth 0.8 on an
    # uncertain frame and 4.97 on a confident one, against a product that never
    # exceeds 1.0: the same weight buys a nudge in one frame and a veto in the
    # next. Normalising puts imi on the gates' scale, where the weight means the
    # most it can ever add and means it everywhere.
    pdm_rank_imi_normalize: bool = False
    # Targets that are 0 / 0.5 / 1 rather than 0 / 1 and need collapsing before
    # a binary cross-entropy reads them.
    pdm_three_class_terms: Tuple[str, ...] = (
        'no_at_fault_collisions',
        'driving_direction_compliance',
    )

    vocab_size: int = 8192
    vocab_path: str = None
    normalize_vocab_pos: bool = False
    
    num_ego_status: int = 1

    # PAI-Track route as the navigation-intent signal instead of the 4-way
    # driving command. NuRec ships a real route; nuPlan-style splits do not, so
    # this stays off by default. The command's four input slots are kept and fed
    # zeros when it is on, so existing checkpoints still load _status_encoding.
    use_route: bool = False
    route_norm_m: float = 80.0          # route x reaches 80 m, ego velocity ~8 m/s
    route_hidden_dim: int = 64
    # Sinusoid bands on the route waypoint coordinates before the MLP. 0 feeds
    # the raw pair instead -- measured on ep19 weights that puts the 20 keys on
    # one line (PC1 99.3% vs 60.1%) and sends 88.2% of attention argmaxes to an
    # endpoint, so it is an ablation switch rather than a tuning knob.
    route_fourier_bands: int = 6
    # False concatenates the route into the image memory (one softmax over both);
    # True gives it a gated cross-attention of its own. Concat is the default
    # because the cost is real and measured -- +6.8% on the stage-1 decoder,
    # +19.3% on refinement, whose 256 queries make the per-query Q/out
    # projections a large share -- while the dilution it fixes (route holding
    # 20 of 3,156 keys) is so far only argued, not measured. Flip it once the
    # attention mass says the route is being drowned out.
    route_separate_attention: bool = False
    # tanh() of this scales the route cross-attention branch in every decoder
    # layer. 0.0 makes the branch exactly the identity at initialisation, so a
    # checkpoint trained without a route starts where it left off; the gate is a
    # single scalar with an undiluted gradient, which is a far stronger signal
    # than 20 route keys competing in a softmax against 3,136 image ones. Raise
    # it to force the branch open from step one when ablating.
    route_gate_init: float = 0.0
    sigma: float = 0.5
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.1
    )

    # img backbone
    backbone_type: str = 'resnet34'
    vit_ckpt: str = ''
    intern_ckpt: str = ''
    vov_ckpt: str = ''
    swin_ckpt: str = ''
    sptr_ckpt: str = ''

    # --- BEVFormer-M front-end (SafeDrive-style), used when backbone_type == 'bevformer_m' ---
    # ResNet image backbone + FPN neck feeding a BEV encoder (learnable BEV queries do
    # deformable spatial cross-attention onto multi-view image features using camera
    # calibration), then a temporal concat+conv over `bev_seq_len` frames.
    bevformer_img_backbone: str = 'resnet34'   # timm CNN model for per-view feature extraction (kind='cnn')
    bevformer_ckpt: str = ''                    # optional pretrained weights for the whole BEV front-end
    # Per-view image backbone kind. 'cnn' -> multi-scale CNN (resnet, out_indices
    # 2/3/4) + FPN. 'vit' -> a plain ViT (single-scale, stride=patch) turned into
    # the multi-scale pyramid BEVFormer's deformable attention needs via a
    # ViTDet-style Simple Feature Pyramid. Use 'vit' + bevformer_vit_name for ViT-L.
    # 'vits' -> DINOv3 ViT-Small/patch16 ported from EADv1.1's deployed ViT-S
    # setup (its own LearnedRGBResize2 stem + up/native/down pyramid); see
    # vits_dinov3_backbone.py and the bevformer_vits_* fields below.
    bevformer_img_backbone_type: str = 'cnn'    # 'cnn' | 'vit' | 'vits' | 'vov'
    # kind='cnn' only. timm stages are picked by STRIDE, not by index, because the
    # index of a given stride differs per family (ResNet [2,4,8,16,32] vs
    # ConvNeXt [4,8,16,32]).
    bevformer_cnn_out_strides: Tuple[int, ...] = (8, 16, 32)
    bevformer_cnn_pretrained: bool = False      # load timm's pretrained weights
    # Overwrite the timm ImageNet weights with a nuImages DETECTION checkpoint,
    # which is what EAD does for its ResNet-50 (ead_backbone.py:34-39, and
    # ead_config.py:106 has use_nuimg_pretrained=True by DEFAULT -- easy to miss,
    # since its phase yaml only names image_architecture: "resnet50").
    #
    # The checkpoint is mmdetection3d's cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim.
    # Only its `backbone.*` keys are used, prefix stripped, `fc.*` dropped, loaded
    # with strict=False -- the same selection StreamPETR and EAD both make. Applied
    # AFTER timm's pretrained load, so it replaces ImageNet rather than merging.
    #
    # Only meaningful for a ResNet: the key names are torchvision-ResNet ones and
    # will simply all land in `unexpected` on any other family.
    bevformer_cnn_nuimg_ckpt: str = ''          # '' = off; else path to the .pth
    # V2-99 (VoVNetV2-99), the backbone DriveSuprim's drivesuprim_agent_vov uses.
    # Stages 3/4/5 are strides 8/16/32 with 512/768/1024 channels, matching the
    # 3-stage -> FPN(num_outs=4) input the original BEVFormer config expects.
    bevformer_vov_ckpt: str = ''                # dd3d_det_final.pth
    bevformer_vov_out_features: Tuple[str, ...] = ('stage3', 'stage4', 'stage5')
    # Input convention fed to the image backbone. 'none' = ToTensor() output as-is
    # (RGB [0,1]), which is what DriveSuprim does; 'dd3d' = the V2-99 checkpoint's
    # own pixel_mean/pixel_std. See BEVFormerM.__init__ for the measurement.
    bevformer_img_norm: str = 'none'             # 'none' | 'dd3d'
    # Which ViT implementation backs kind='vit':
    #   'timm'   - timm.create_model(bevformer_vit_name); DINOv2 ViT-L is patch-14
    #              + 4 register tokens, which forces a /14 input (252x504).
    #   'dinov2' - DriveSuprim's own DinoVisionTransformer (navsim.agents.utils.vit),
    #              patch-16, no register tokens, initialised from da_vitl16.pth.
    #              Patch-16 divides 256x512 exactly, matching SafeDrive's resolution.
    bevformer_vit_impl: str = 'timm'            # 'timm' | 'dinov2'
    bevformer_vit_name: str = 'vit_large_patch16_384'  # timm ViT (used when impl='timm'); ViT-L, patch16
    bevformer_vit_pretrained: bool = False      # let timm pull pretrained weights (needs network)
    bevformer_vit_ckpt: str = ''                # OR load ViT weights from this local checkpoint
                                                # (required for impl='dinov2': path to da_vitl16.pth)
    bevformer_vit_freeze: bool = False          # freeze the ViT (train only pyramid+BEV); saves memory, stabilizes
    # kind='vits': DINOv3 ViT-Small/patch16, ported from EADv1.1's actual deployed
    # ViT-S setup (SimpleBEVDinoBackbone / SimpleBEVViTFeaturePyramid /
    # LearnedRGBResize2 in ead_backbone_simplebev.py). Reuses bevformer_vit_name /
    # _pretrained / _ckpt / _freeze above (set bevformer_vit_name=
    # 'vit_small_patch16_dinov3'); these four fields are the pieces that do not
    # already exist elsewhere. See vits_dinov3_backbone.py's module docstring for
    # what carries over from EAD verbatim vs. what is re-tuned for 512x256.
    bevformer_vits_ms_up: int = 1                    # upsample (finer) pyramid levels
    bevformer_vits_ms_down: int = 1                  # downsample (coarser) pyramid levels
                                                      # ms_up=1,ms_down=1 -> 3 levels @ strides 8/16/32
    bevformer_vits_learned_resize2: bool = False     # EAD's learned /2 RGB stem; off at 512x256 (see docstring)
    bevformer_vits_resize_hidden_channels: int = 16  # only used if _learned_resize2=True
    bevformer_vits_resize_blocks: int = 1            # only used if _learned_resize2=True
    bevformer_vits_amp: bool = True                  # autocast the ViT forward (EAD: simplebev_vit_amp)
    bevformer_vits_flash_dtype: str = 'bf16'         # 'bf16' | 'fp16' (EAD: simplebev_vit_flash_dtype)
    # Encoder aligned to BEVFormer/UniAD (6 layers, 4 feature levels, embed 256,
    # 4 pillar z-samples). BEV grid kept at 100x100 for the 24GB (RTX 3090) budget
    # (BEVFormer-base uses 200x200, which needs ~48GB-class GPUs).
    bev_h: int = 100                            # BEV query grid height (ego-forward axis)
    bev_w: int = 100                            # BEV query grid width  (ego-lateral axis)
    # Trajectory-head keyval pooling. The first-stage HydraTrajHead cross-attends
    # all 8192 vocab queries to the BEV tokens; at the original 200x200 grid that
    # is 8192x40000 attention (4x the 100x100 cost) and dominates memory. Setting
    # bev_keyval_grid>0 adaptive-pools the BEV feature to a GxG grid JUST for the
    # first-stage keyval (e.g. 50 -> 2500 tokens), while detection/seg/refinement
    # still use the FULL-resolution BEV map. 0 = use the full bev_h x bev_w grid.
    bev_keyval_grid: int = 0
    bev_embed_dims: int = 256                   # BEV query / feature channels (matches tf_d_model)
    bev_num_encoder_layers: int = 6             # BEVFormer/UniAD standard = 6
    # History frames run only the first N encoder layers (weight-shared, no extra
    # parameters); the current frame always runs the full depth. They are fused by
    # a conv afterwards and supply context rather than precise geometry, so the
    # full stack is wasted on them. 0 = full depth for every frame (original).
    bev_history_encoder_layers: int = 0
    # Reuse the previous call's per-frame BEVs instead of re-encoding the history
    # every forward. Inference only, and only valid when frames arrive in order --
    # call BEVFormerM.reset_stream() on any discontinuity.
    bev_streaming: bool = False
    # BEV encoder compute precision. 'fp32' | 'bf16'. fp16 is not an option: the
    # BEV features grow across layers and overflow its 65504 range. bf16 keeps
    # fp32's exponent range so it cannot overflow, trading mantissa bits instead.
    bev_encoder_dtype: str = 'fp32'
    bev_num_feature_levels: int = 4             # FPN output levels (BEVFormer num_outs=4)
    bev_num_points_in_pillar: int = 4           # z-samples per BEV pillar for spatial cross-attn

    # NuRec 로그의 자차 pose 는 (x, y, yaw) 뿐이라 pitch·roll 이 없다.  BEV 격자와
    # gt_boxes 는 그 수평 좌표계에 있고 카메라 외부 파라미터는 차와 함께 기울어진
    # rig 좌표계에 있어서, 그대로 투영하면 이미지에서 f*tan(pitch) 만큼 세로로
    # 밀린다 -- 거리와 무관하게 일정하고, 프레임의 18.4 % 가 10 px 이상이다.
    # 켜면 `assets/nurec/rig_tilt.npz` 의 보정행렬을 곱해 그 자리를 바로잡는다.
    #
    # 기본값 켜짐.  `assets/nurec/rig_tilt.npz` 에 없는 프레임(NAVSIM/nuPlan 로그
    # 전부)은 조회가 비어 보정을 건너뛰므로, NuRec 밖의 동작은 바뀌지 않는다.
    #
    # 주의: 이 플래그 이전에 학습된 체크포인트는 **밀린** 투영에 맞춰져 있다.
    # 그것들을 추론할 때는 학습 때와 같게 두려면 꺼야 한다.
    bev_pitch_correct: bool = True
    # Deformable sampling points per head. Measured on one encoder layer at
    # 128x128 / 3 levels: spatial cross-attn is 57.2% of the layer, BEV
    # self-attn 2.8% and the FFN 1.1% -- so sca_num_points is the only one of
    # the three worth tuning. Defaults are BEVFormer's, which pair 8 SCA points
    # with FOUR feature levels; three levels here already changes the total.
    bev_sca_num_points: int = 8
    bev_ssa_num_points: int = 4
    # Vertical extent the BEV encoder's pillar anchors span, as (z_min, z_max).
    # None -> use point_cloud_range's z, which is what BEVFormer does.
    #
    # Worth separating because point_cloud_range's z serves a second master: the
    # detection head denormalises its sigmoid box z with it, so it has to stay
    # wide enough that no object centre is unrepresentable. The anchors want the
    # opposite -- tight around real object heights, or they project onto sky and
    # road instead of onto the objects they are lifting.
    bev_pillar_z_range: Optional[Tuple[float, float]] = None
    bev_num_cameras: int = 3                    # multi-view cameras fed to BEV encoder (cam_l0/f0/r0)
    bev_img_height: int = 256                   # per-view image height fed to the img backbone
    bev_img_width: int = 512                    # per-view image width  fed to the img backbone
    # Optional SafeDrive-style preprocessing before the resize above. Both are
    # off by default, preserving the direct raw-image squash resize path.
    bev_undistort: bool = False                 # cv2 undistort/remap using cam.distortion first
    bev_crop_top_bottom: int = 0                # pixels cropped from both top and bottom before resize
    bev_seq_len: int = 3                        # frames for temporal fusion (current + past)
    bev_use_temporal_align: bool = True         # warp past-frame BEV into current ego frame via ego pose
    # Gradient checkpointing over the per-frame BEV extraction (image backbone +
    # encoder). Trades ~1.3x compute for a large activation-memory cut. Needed
    # when the rotation ensemble runs 4 student forwards (ori + 3 rotated) that
    # would otherwise OOM 24GB GPUs. No effect on numerics.
    bev_use_grad_checkpoint: bool = False
    # Compute the past frames' BEV without gradients, the way BEVFormer's
    # forward_train does (obtain_history_bev runs under no_grad + eval, and
    # only img[:, -1] reaches extract_feat with a graph). Saves the backward
    # pass over T-1 of the T frames. The cost is that the image backbone and
    # BEV encoder then learn from the current frame only.
    bev_history_no_grad: bool = False
    # metric extent of the BEV grid in ego frame [x_min, y_min, z_min, x_max, y_max, z_max] (meters)
    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        -32.0, -32.0, -3.0, 32.0, 32.0, 5.0
    )

    # --- BEVFormer auxiliary dense heads (detection + BEV map segmentation) ---
    # The BEVFormer front-end produces a real BEV feature grid, so we can attach
    # the same auxiliary supervision TransFuser uses: a BEV semantic-segmentation
    # head (dense map/agent raster) and a DETR-style agent detection head. These
    # add geometric grounding to the BEV features that the trajectory-scoring
    # heads alone do not provide. Only active when backbone_type == 'bevformer_m'.
    # Off by default so existing runs / checkpoints are unaffected.
    use_aux_heads: bool = False
    # BEV segmentation target scope. drivable_only=True -> BINARY seg of the
    # drivable area only (0 = non-drivable, 1 = drivable/road), where "drivable"
    # is the LANE+INTERSECTION polygon (== TransFuser's "road" class, and the
    # region the PDM drivable_area_compliance metric cares about). Set False to
    # fall back to the full multi-class (num_bev_classes) semantic map.
    aux_bev_drivable_only: bool = True

    # Which polygon layers the binary drivable target is rasterised from.
    # Defaults to None, meaning the inherited bev_semantic_classes[1], so NAVSIM
    # agents are untouched.
    #
    # The NuRec agents set LANE + LANE_CONNECTOR instead of the inherited
    # LANE + INTERSECTION. INTERSECTION is one polygon over a whole junction, so
    # on a roundabout it paints the centre island as drivable -- of the frames
    # where the two choices disagree most, every one was a roundabout and the
    # recorded human drove around the island, never across it. LANE_CONNECTOR
    # keeps the through-lanes that actually cross a junction, and the pair is
    # what pdm_scorer._road_edge_compliance reads, so the target and the metric
    # look at the same polygons.
    #
    # Road boundaries are deliberately NOT used. They would recover the ~5% of
    # the recorded human's body positions the lane polygons miss (the annotation
    # sits about a metre off the driven line), but every route to them failed:
    # rasterised on the 1 m grid they eat half a metre either side; flood-filled
    # they escape around any boundary that ends inside the window, on one scene
    # in five; and map_api's proximal query returns none of them at all, which
    # made an empty wall read as "the entire frame is drivable".
    aux_bev_drivable_layers: Optional[List[SemanticMapLayer]] = None

    # Drop the segmentation target on frames where the map does not cover the
    # road the car is on. Measured over 1,000 frames from 250 clips, 0.4% have
    # nothing drivable within six metres ahead of the rear axle -- the ego is
    # driving on tarmac the polygons do not describe. Those are map coverage
    # gaps, not hard examples: supervising them teaches the head that the road
    # under the car is not road.
    #
    # Only the segmentation term is dropped. The frame's trajectory labels are
    # unaffected and stage 2 and 3 still learn from them, so the frame stays in
    # the split.
    aux_bev_seg_require_ego_on_road: bool = False
    aux_bev_seg_ego_band_m: float = 6.0     # ahead of the rear axle
    aux_bev_seg_ego_half_width_m: float = 2.0
    # SafeDrive's loss table lists BEV semantic and drivable-area segmentation as
    # two separate terms (14.0 each). With this on, the head keeps one ConvNeXt-V2
    # trunk and adds a second 1x1 classifier for the binary drivable mask, so both
    # are supervised. Requires aux_bev_drivable_only=False (the main map is then
    # the multi-class one).
    aux_bev_dual_seg: bool = False
    aux_bev_drivable_weight: float = 1.0
    # Drop the box-derived entries (static_objects / vehicles / pedestrians) from
    # the BEV semantic map, leaving only the static map layers. SafeDrive calls
    # this branch "static BEV segmentation maps", and those three classes cover
    # just 2.53% of pixels while the detection head already supervises the same
    # objects far more directly (900 queries, Hungarian-matched boxes).
    aux_bev_seg_static_only: bool = False

    def active_bev_semantic_classes(self):
        """bev_semantic_classes, optionally without the box entries, relabelled
        to a contiguous 1..N so the head's channel count stays tight."""
        items = sorted(self.bev_semantic_classes.items())
        if self.aux_bev_seg_static_only:
            items = [(k, v) for k, v in items if v[0] != "box"]
        return {i + 1: v for i, (_, v) in enumerate(items)}

    @property
    def num_active_bev_classes(self) -> int:
        return len(self.active_bev_semantic_classes()) + 1  # + background
    # Loss weights. RELATIVE RATIO seg:class:box = 10:10:1 is taken verbatim from
    # TransFuser (PAMI'22) / the NAVSIM-official Transfuser baseline in this repo
    # (transfuser_config.py: bev_semantic=10, agent_class=10, agent_box=1) -- the
    # same two heads on the same dataset, so it is the strongest reference. Box is
    # 1/10 of class because our box coords are in METERS (AgentHead tanh*32), so
    # the L1 magnitude is ~30x that of DETR's [0,1]-normalized coords; DETR raises
    # its L1 weight (5) for normalized coords, TransFuser lowers it for metric ones.
    #
    # ABSOLUTE SCALE is set 10x below TransFuser's so the aux block acts as a
    # GROUNDING REGULARIZER (~10-15% of the planning term it is folded into), not a
    # co-equal objective. Measured planning scale on the aug run: train/loss-ori
    # ~= 7.67; converged raw aux ~= (seg CE 0.4, class BCE 0.15, box L1 5m), so the
    # weighted aux below lands ~1.0 (~13% of loss-ori). Unlike TransFuser (planning
    # = single L1*10, so aux is co-equal), DriveSuprim's planning is a large
    # vocab-scoring BCE+imi sum, which would be swamped by co-equal aux.
    # NOTE: aux is folded into loss-ori (the val checkpoint monitor); kept small so
    # selection stays ~planning-driven. Tune per-run via the train.sh env vars, or
    # switch to uncertainty weighting (Kendall CVPR'18) if hand-tuning proves brittle.
    # MEASURED (aug run, realistic GT): planner loss-ori ~= 6.3-9.6; raw aux at
    # init seg_CE 1.89, class_BCE 0.61, box_L1 26.2 (box large: coords in meters).
    # With box=0.1 the init weighted box (2.6) alone is ~30% of the planner and
    # dominates the aux block, so box is set to 0.05 (=> init weighted aux ~3.8,
    # converged ~0.8 ~10-13% of the planner -- a balanced regularizer, not a hot
    # early competitor). This makes the ratio 20:20:1, stronger metric-box
    # down-weighting than TransFuser's 10:10:1 -- justified by the measured
    # metric-scale dominance. Cleanest alternative (not taken here to keep the
    # reused TransFuser matcher): normalize box coords by pc_range (DETR/BEVFormer
    # style) so L1 is O(1) and the 10:10:1 ratio holds directly.
    aux_bev_seg_weight: float = 1.0     # BEV seg cross-entropy    (TransFuser 10, /10)
    # Per-class weights for the BEV segmentation cross-entropy, in EAD's
    # convention: inverse class frequency normalised so the weights sum to the
    # class count (EAD's 7-class vector sums to 7.0). Measured on this drivable
    # target over 388 navtest frames / 12.4M cells -- 60.84% background,
    # 39.16% drivable -> [0.7832, 1.2168]. None disables weighting.
    # Re-measure if bev_h/bev_w/point_cloud_range or the drivable layers change.
    aux_bev_seg_class_weights: Optional[Tuple[float, ...]] = None
    aux_bev_drivable_class_weights: Optional[Tuple[float, ...]] = None
    # Clamp BEV logits to +/- this before the softmax (EAD's bev_logit_clamp).
    # 0 disables; NaN/inf are sanitised either way.
    aux_bev_logit_clamp: float = 0.0
    # Detach the BEV feature before the detection and segmentation heads, so the
    # perception losses train those heads ONLY and never reach the shared trunk.
    #
    # Aimed at stage 3, where perception is unfrozen and the two objectives start
    # competing for the same features. Measured on v1: unfreezing costs mAP
    # 0.4524 -> 0.4139 and seg IoU 0.8475 -> 0.8097 in the first epoch, while
    # every PDM sub-loss improves (DAC -20%, LK -14%, NC -7%). Lowering the
    # backbone lr damps both effects together; this separates them -- the trunk
    # follows the planner alone and the heads re-fit to it.
    #
    # Which is better is an empirical question, and it matters here more than for
    # the original architecture because the feasibility gate reads these heads.
    aux_detach_bev: bool = False
    aux_agent_class_weight: float = 1.0      # detection cls, Hungarian (TransFuser 10, /10)
    aux_agent_box_weight: float = 0.05       # detection box L1 (metric): extra down-weight, see above
    # Detection head implementation. "transfuser" preserves the lightweight
    # NAVSIM baseline head; "bevformer" uses a BEVFormer-style query decoder over
    # the full BEV grid with reference-point xy refinement. Segmentation remains
    # the binary drivable-area head controlled by aux_bev_drivable_only.
    aux_agent_head_type: str = "transfuser"
    aux_agent_tf_layers: int = 3             # transformer-decoder layers in the lightweight head
    aux_bevformer_det_layers: int = 6        # decoder layers in the BEVFormer-style detection head
    aux_bevformer_det_with_box_refine: bool = True
    # Sampling points per head for the detection head's deformable cross-attention
    # (original BEVFormer CustomMSDeformableAttention default).
    aux_bevformer_det_num_points: int = 4
    # Original BEVFormer decoder uses dropout 0.1 on self-attn / cross-attn / FFN
    # (attn_cfgs dropout=0.1, ffn_dropout=0.1).
    aux_bevformer_det_dropout: float = 0.1
    # feedforward_channels=_ffn_dim_=_dim_*2 in the BEVFormer config; DriveSuprim's
    # tf_d_ffn is 1024. Default to BEVFormer's for the BEVFormer-style head.
    aux_bevformer_det_ffn: int = 512
    # Focal loss for the detection classifier, as in the original head
    # (loss_cls=FocalLoss(gamma=2.0, alpha=0.25)). Set False for plain BCE.
    aux_agent_use_focal: bool = True
    aux_agent_focal_gamma: float = 2.0
    aux_agent_focal_alpha: float = 0.25
    # 3D detection, following the original BEVFormer head's code_size=10 layout:
    #   [cx, cy, log(l), log(w), cz, log(h), sin(rot), cos(rot), vx, vy]
    # (see BEVFormer's normalize_bbox; cz sits at index 4, which is why the
    #  decoder refines the reference z from tmp[..., 4:5]). NAVSIM annotations
    # carry all of it: boxes are (N, 7) = x,y,z,l,w,h,heading and velocity_3d is
    # (N, 3), so nothing has to be dropped.
    aux_agent_box_3d: bool = False
    # Log detection AP / centre-distance errors and segmentation IoU next to the
    # losses. The matching loop is python-level, so on large query counts it is
    # worth computing only every N steps.
    aux_log_metrics: bool = True
    # Training-side interval. Measured on the NuRec stage-1 run: computing these
    # every step cost 23x the rest of the step put together, and 66.7 min of a
    # 66.7 min epoch. They are logs, not losses -- nothing in the gradient reads
    # them -- so sampling the training curve every N steps loses nothing.
    aux_metric_every_n_steps: int = 1
    # Validation gets its OWN interval, and it defaults to every batch. Sharing
    # one counter across both phases was a real bug: at 50 it left the epoch's
    # seg IoU / det mAP measured on 2-3 of the 144 val batches (~72 frames of
    # 13,729), which moved 0.02 between epochs on sampling noise alone and made
    # the checkpoint-selection metrics unusable. The losses were unaffected --
    # they are computed on every batch either way -- which is exactly what made
    # it look like a performance drop rather than a measurement change.
    aux_metric_every_n_steps_val: int = 1
    det_metric_thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    # BEVFormer's default code_weights: velocity down-weighted to 0.2.
    aux_agent_code_weights_3d: Tuple[float, ...] = (
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2,
    )
    # Per-dimension weights for the legacy 2D box (x, y, heading, length, width).
    aux_agent_code_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    # How many box dimensions the Hungarian matching cost compares. BEVFormer's
    # HungarianAssigner3D uses bbox_pred[:, :8] -- everything but vx, vy. 0 (or a
    # box layout with fewer dims, e.g. the 5-dim 2D one) falls back to the
    # centre-only cost.
    aux_agent_match_cost_dims: int = 8
    # Hungarian classification cost. True = mmdet's FocalLossCost
    # (pos_cost - neg_cost, the assigner BEVFormer configures); False = the plain
    # -p(class) cost.
    aux_agent_use_focal_match_cost: bool = True
    # Average the positive count across ranks before using it as avg_factor, as
    # BEVFormer's reduce_mean(num_total_pos) / sync_cls_avg_factor=True do. No
    # effect on a single GPU.
    aux_agent_sync_avg_factor: bool = True
    # How the 5 intermediate decoder layers' losses combine. BEVFormer logs each
    # layer separately, so they are summed at full weight; 'mean' scales each to
    # 1/5 and makes the deep-supervision signal ~5x weaker.
    aux_decoder_layer_reduction: str = 'sum'   # 'sum' | 'mean'
    # Detection classes. ("vehicle",) reproduces the original single-class
    # objectness head, which discards ~74% of the labelled boxes. Segmentation
    # supervises only the drivable area, so these boxes are the sole signal for
    # everything else.
    #
    # Measured over EVERY scene, counting only boxes inside point_cloud_range --
    # i.e. what training actually receives, not the raw annotation totals:
    #
    #                    navtrain (47950)        navtest (5044)
    #   generic_object   16.80/frame  42.82%     8.99/frame  33.85%
    #   vehicle          10.11        25.76      8.53        32.15
    #   pedestrian        9.52        24.27      5.80        21.87
    #   traffic_cone      2.17         5.52      2.35         8.84
    #   barrier           0.46         1.18      0.70         2.64
    #   bicycle           0.09         0.24      0.11         0.40
    #   czone_sign        0.08         0.20      0.07         0.25
    #   total            39.24                  26.54
    #
    # Two things this makes visible. The splits are NOT identically distributed:
    # barrier is 2.2x more common in navtest than in navtrain and traffic_cone
    # 1.6x, while generic_object is 9 points rarer -- so dropping the rare
    # classes costs more at evaluation than the training counts suggest. And the
    # imbalance is 214:1 between the most and least common, which is why the
    # bottom three will have poor AP no matter what; for the feasibility gate
    # that is tolerable, since it only needs the box to clear the confidence
    # threshold, not to be classified precisely.
    aux_agent_classes: Tuple[str, ...] = (
        "vehicle", "pedestrian", "bicycle", "traffic_cone",
        "barrier", "czone_sign", "generic_object",
    )
    # Number of object queries in the BEVFormer-style detection head. Kept
    # SEPARATE from num_bounding_boxes (the number of GT slots): DETR-family
    # heads want far more queries than objects, but the Hungarian cost matrix is
    # num_queries x num_gt_slots, and linear_sum_assignment is superlinear.
    # Measured, batch 8 x 6 decoder layers: 900x900 costs 1500 ms/step on CPU,
    # while 900x100 costs 22 ms. Padding the GT side to 900 buys nothing --
    # navtest has at most ~90 boxes inside the BEV.
    # None -> fall back to num_bounding_boxes (original coupled behaviour).
    aux_bevformer_det_num_queries: Optional[int] = None
    # ConvNeXt-V2 blocks in front of the 1x1 BEV segmentation classifier.
    aux_seg_convnext_blocks: int = 2

    # --- Agent motion prediction (OPTIONAL 4s forecasting) -------------------
    # DEFAULT: OFF -> the agent head only does CURRENT-frame detection (boxes at
    # the present timestep). Turn this ON to additionally forecast each detected
    # agent's future trajectory over the next few seconds in the current ego
    # frame (GT linked across future frames via track_tokens; navtrain has 10
    # future frames @ 0.5s = 5s available). Shares the detection decoder +
    # Hungarian matching, adds an L1 waypoint loss.
    aux_predict_agents: bool = False
    aux_pred_num_poses: int = 8              # future waypoints (x0.5s frame interval => 4s horizon)
    aux_pred_weight: float = 0.1             # L1 on future waypoints (metric coords -> down-weighted like box)

    # --- Trajectory-candidate feasibility filter (OPTIONAL) ------------------
    # Uses the aux heads' OWN outputs (drivable-area BEV seg + agent detection)
    # to demote vocab candidates that leave the drivable area or collide with
    # a detected agent, BEFORE the score argmax / refinement top-k. Requires
    # use_aux_heads=True.
    #
    # Applied in TRAINING as well as inference, deliberately. The gate is part of
    # the deployed model, so gating only at inference would train refinement on a
    # candidate distribution it never sees at test time. Feeding it GT boxes /
    # GT drivable instead was considered and rejected: the per-candidate BCE
    # labels come from the offline PDM pickle and are unaffected by which
    # candidates are selected, so a GT gate buys no label quality -- it only
    # strips out the colliding candidates the scorer has to learn to reject, and
    # leaves it untrained on exactly the cases a prediction gate lets through.
    #
    # On by default. The model runs the aux heads even without compute_aux. If every
    # candidate in a sample is masked, that sample falls back to the unfiltered
    # scores. The ego is treated as its footprint RECTANGLE (4 corners) at each of
    # the 40 poses:
    #  - drivable gate matches NAVSIM DAC (pdm_scorer): a pose is off-road if ANY
    #    ego bounding-box corner is off the drivable seg (corner-based, not center).
    #  - collision gate is an oriented-box (OBB-OBB, SAT) overlap between the ego
    #    footprint and each detected agent box.
    # Both boxes are slightly scaled up (feasibility_collision_scale; set 1.0 to
    # match DAC/collision geometry exactly). Corners outside point_cloud_range are
    # ignored; agents are static at their current box unless aux_predict_agents
    # gives per-timestep futures.
    feasibility_enabled: bool = True          # master switch (training + inference)
    feasibility_drivable: bool = True          # drop candidates leaving the drivable area
    feasibility_collision: bool = True         # drop candidates colliding with an agent
    feasibility_drivable_thresh: float = 0.5   # min P(drivable) at an ego corner to keep it
    # Check the drivable gate every Nth pose instead of all 40. The ego is 4.6 m
    # long and a BEV cell is 0.4 m, so consecutive poses overlap heavily and most
    # of the lookups are redundant. Unlike the collision short-circuit this is an
    # APPROXIMATION: a violation entirely between two sampled poses is missed, and
    # the gap grows with speed (20 m/s x 0.2 s = 4 m, comparable to the car).
    feasibility_drivable_pose_stride: int = 1
    # Agents per SAT chunk. Trades peak memory ([B,C,Veff,P_cv,4] floats) against
    # the number of Python iterations; the result is identical either way.
    # Measured on 30 random navtest scenes (34 detections on average, 146 peak):
    # 1 -> 7.6 ms, 16 -> 2.8 ms, 32 -> 1.75 ms, 64 -> 1.60 ms, with peak memory
    # rising only 1834 -> 1950 MB across that whole range -- the memory cost of
    # going wide is small enough that there is no reason to stop short.
    feasibility_agent_chunk: int = 64
    feasibility_agent_conf_thresh: float = 0.3 # detection confidence to count an agent
    feasibility_ego_length: float = 4.6        # m, ego footprint length (OBB collision)
    feasibility_ego_width: float = 1.9         # m, ego footprint width
    # Forward offset from the trajectory pose to the ego bounding-box CENTRE.
    # The gate places the footprint rectangle at the pose, which is only correct
    # when the two coincide. Under AlpaSim they do not: a pose is the rig origin,
    # defined as the mid bottom rear edge of the box, so the box centre sits
    # roughly half a vehicle ahead of it and the rectangle is otherwise evaluated
    # ~1.5 m behind the real car -- missing its front and testing empty road
    # behind it. Only the forward component exists in practice: the rig is on the
    # vehicle centreline (zero lateral offset) and the transform carries no
    # rotation. The driver fills this from the rollout spec's
    # rig_to_bounding_box; 0.0 reproduces the pose-centred behaviour the
    # checkpoint was trained with.
    feasibility_ego_center_dx_m: float = 0.0   # m, forward offset pose -> box centre
    feasibility_collision_scale: float = 1.1   # slight box scale-up (ego + agent) for margin
    # Per-class agent box scale, overriding feasibility_collision_scale for the
    # named classes. Keys must appear in aux_agent_classes; classes not listed
    # fall back to the global scale.
    #
    # Only the vulnerable road users are widened. PDM zeroes NC outright for
    # AGENT_TYPES (vehicle, pedestrian, bicycle) but merely halves it to 0.5 for
    # the static types, and pedestrian and bicycle are the two that also move
    # freely inside the horizon -- measured on navtrain, pedestrian runs 1.06 m/s
    # median and bicycle 1.6 m/s median / 10.6 m/s at p95, so neither stays
    # inside its own footprint. A full-weight metric plus unconstrained motion is
    # what earns the 2.0; vehicles move faster still but are large and
    # kinematically constrained, so 1.1 covers them.
    #
    # The static classes (barrier, czone_sign, traffic_cone, generic_object) sit
    # at the global 1.1 because PDM's own object manager does not propagate them
    # at all: they hold their pose for the full horizon, so there is no motion
    # uncertainty for a margin to cover. Inflating a 1.28 m czone_sign to 2.56 m
    # only cuts passable candidates.
    feasibility_class_scale: dict = field(default_factory=lambda: {
        'vehicle': 1.1, 'pedestrian': 2.0, 'bicycle': 2.0, 'traffic_cone': 1.1,
        'barrier': 1.1, 'czone_sign': 1.1, 'generic_object': 1.1,
    })
    feasibility_collision_margin: float = 0.0  # m, extra half-extent added to every box side
    # Constant-velocity propagation of detected agents along the candidate
    # trajectory, instead of freezing them at their current box for the whole
    # 4 s horizon. PDM's own NC walks every timestep with the agents' real
    # future poses, so a static gate both over-rejects (a car being followed
    # never moves out of the way) and under-rejects (nothing ever cuts in).
    # Only vx, vy are available -- the box has no yaw-rate term -- so this is
    # CV, i.e. CTRV with omega = 0. Beyond feasibility_predict_horizon the
    # collision test is SKIPPED rather than extrapolated: past ~1 s a
    # constant-velocity guess is not worth trusting, and holding the agent
    # at its last pose would reject candidates on a fiction. The drivable
    # gate still applies over the full trajectory.
    feasibility_predict_agents_cv: bool = True
    feasibility_predict_horizon: float = 1.0   # s of propagation; 0 = static

    lr_mult_backbone: float = 1.0
    # Only meaningful for backbone_type='bevformer_m', where `_backbone.image_encoder`
    # bundles the pretrained per-view image backbone with the from-scratch BEV
    # front-end. Set (e.g. 0.1) to give the image backbone its own lower LR when
    # unfreezing it; None keeps the original single-bucket behaviour.
    lr_mult_img_backbone: Optional[float] = None
    backbone_wd: float = 0.0

    n_camera: int = 3  # 1 or 3 or 5

    camera_width: int = 2048
    camera_height: int = 512
    img_vert_anchors: int = camera_height // 32
    img_horz_anchors: int = camera_width // 32

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    training: bool = True

    # Augmentation setting
    only_ori_input: bool = False  # 如果是 True，说明是原来的训练设置
    ego_perturb: EgoPerturbConfig = EgoPerturbConfig()
    ori_vocab_pdm_score_full_path: str = "???"
    # Optional per-token directory of ori vocab PDM scores ({token}.pkl each).
    # When set, the agent lazily loads only the batch's tokens instead of the
    # ~15GB monolithic pickle, which otherwise sits resident in every DDP
    # process (and is copy-on-write duplicated across dataloader workers,
    # causing system-RAM OOM). Leave empty to use ori_vocab_pdm_score_full_path.
    ori_vocab_pdm_score_dir: str = ""
    aug_vocab_pdm_score_dir: str = "???"

    # Self-distillation
    # Self-distillation on/off. The teacher branch exists only to produce soft
    # targets for the student; at 0.0 the teacher forward is skipped outright
    # (SSLMetaArch.forward), which removes half the parameters from the step and
    # makes this a plain supervised run against the labels alone.
    use_soft_teacher: bool = True
    soft_label_traj: str = 'first'  # first or final
    soft_label_imi_diff_thresh: float = 1.0
    soft_label_score_diff_thresh: float = 0.15
    update_buffer_in_ema: bool = False

    refinement: RefinementConfig = RefinementConfig()
    inference: InferConfig = InferConfig()
