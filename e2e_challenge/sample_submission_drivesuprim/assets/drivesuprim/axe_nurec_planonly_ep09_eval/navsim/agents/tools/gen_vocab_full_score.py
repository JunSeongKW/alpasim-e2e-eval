import json
import logging
import lzma
import os
import pickle
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import hydra
import numpy as np
import torch
import numpy.typing as npt
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list
from omegaconf import DictConfig

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.evaluate.pdm_score import (
    human_states_to_simulate,
    pdm_score_full_v2,
    proposal_states_to_simulate,
)
from navsim.common.enums import SceneFrameType
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)
# One file per token holding every component. It serves twice over: as the
# resume cache for this job, and as the directory the trainer lazily reads
# through ``ori_vocab_pdm_score_dir`` -- the layouts are identical, so writing
# it once is enough.
PER_TOKEN_DIRNAME = 'per_token'
trajpdm_root = os.getenv('NAVSIM_TRAJPDM_ROOT')
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
CONFIG_PATH = f"{devkit_root}/navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    vocab_size = cfg.vocab_size
    scene_filter_name = cfg.scene_filter_name
    # NuRec has its own trajectory distribution.  Keep the historical default
    # but allow a generated NuRec vocabulary to be supplied through Hydra.
    traj_path = cfg.get("vocab_path") or f"{devkit_root}/traj_final/test_{vocab_size}_kmeans.npy"
    dir = f'ori/vocab_score_{vocab_size}_{scene_filter_name}'
    build_logger(cfg)
    worker = build_worker(cfg)
    vocab = np.load(traj_path)
    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_loader = SceneLoader(
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    os.makedirs(f'{trajpdm_root}/{dir}', exist_ok=True)
    result_name = cfg.get("result_name") or scene_filter_name
    result_path = f'{trajpdm_root}/{dir}/{result_name}.pkl'
    report_path = f'{trajpdm_root}/{dir}/{result_name}_report.json'
    per_token_dir = f'{trajpdm_root}/{dir}/{PER_TOKEN_DIRNAME}'
    os.makedirs(per_token_dir, exist_ok=True)
    print(f'Results will be written to {result_path}')
    print(f'Per-token component scores: {per_token_dir}')

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "vocab": vocab
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    new_data_points = []
    for data in data_points:
        for token in data['tokens']:
            new_data_points.append({
                "cfg": cfg,
                "dir": dir,
                "log_file": data['log_file'],
                "token": token,
                "vocab": vocab
            })
    requested_tokens = {data['token'] for data in new_data_points}

    score_rows: List[Dict[str, Any]] = map_with_gpu_share(worker, run_pdm_score, new_data_points)
    final: Dict[str, Any] = {}
    failures: List[Dict[str, str]] = []
    for tmp in score_rows:
        if tmp is None:
            continue
        if tmp.get('failed'):
            failures.append({"token": tmp['token'], "reason": str(tmp.get('reason', 'unknown'))})
            continue
        # NuRec training can consume a compact token -> EPDMS array artifact;
        # the worker already reduced the payload when +epdms_only=true.
        final[tmp['token']] = tmp['score']

    accounted = set(final) | {failure['token'] for failure in failures}
    unaccounted = sorted(requested_tokens - accounted)
    report = {
        "result_path": result_path,
        "per_token_dir": per_token_dir,
        "epdms_only": bool(cfg.get("epdms_only", False)),
        "requested_tokens": len(requested_tokens),
        "scored_tokens": len(final),
        "failed_tokens": len(failures),
        "unaccounted_tokens": len(unaccounted),
        "failures": failures[:1000],
        "unaccounted": unaccounted[:1000],
    }
    with open(report_path, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)

    # Write through a sibling temp file: a 400 MB dump interrupted on NFS must
    # not leave a truncated artifact where a good one used to be.
    tmp_result_path = f'{result_path}.tmp'
    with open(tmp_result_path, 'wb') as stream:
        pickle.dump(final, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_result_path, result_path)

    logger.info(
        f"Training reads the components from {per_token_dir} via "
        f"agent.config.ori_vocab_pdm_score_dir; {result_path} is the EPDMS-only artifact."
    )
    if failures or unaccounted:
        logger.error(
            f"{len(final)}/{len(requested_tokens)} tokens scored; {len(failures)} failed, "
            f"{len(unaccounted)} unaccounted. See {report_path}"
        )
    else:
        logger.info(f"All {len(final)} requested tokens scored. Report: {report_path}")


def map_with_gpu_share(
    worker: WorkerPool, fn: Callable[[List[Any]], List[Any]], items: List[Any]
) -> List[Any]:
    """``worker_map``, except every Ray task asks Ray for a slice of a GPU.

    nuplan's ``worker_map`` submits ``Task(fn=fn)`` with ``num_gpus`` unset.
    Ray reads that as zero and hands the worker ``CUDA_VISIBLE_DEVICES=""``, so
    ``torch.cuda.is_available()`` is False inside every worker and the MPC
    quietly falls back to the CPU -- correct answers, roughly seventy times the
    runtime, with nothing in the log to say so. Requesting a fraction makes Ray
    pin each worker to a real device and spread them evenly, which also retires
    the pid-modulo guess in ``resolve_device``.
    """
    if worker.number_of_threads == 0:
        return fn(items)

    chunks = chunk_list(items, worker.number_of_threads)
    share = _gpu_share_per_task(len(chunks))
    if share is None:
        logger.warning("No GPU visible to Ray; the controller will run on the CPU.")
    else:
        logger.info(f"Requesting {share:.4f} GPU per worker across {len(chunks)} workers.")
    scattered = worker.map(Task(fn=fn, num_gpus=share), chunks)
    return [row for rows in scattered for row in rows]


def _gpu_share_per_task(num_tasks: int) -> Union[float, None]:
    """How much of a GPU one worker may claim, or None when there are none.

    Asked of Ray rather than torch: the driver has no reason to open a CUDA
    context, and Ray's own count is what the scheduler will honour anyway.
    """
    try:
        import ray
    except ImportError:
        return None
    if not ray.is_initialized():
        return None
    gpus = float(ray.cluster_resources().get("GPU", 0.0))
    if gpus <= 0.0 or num_tasks <= 0:
        return None
    return min(1.0, gpus / num_tasks)



class _ReplaySimulator:
    """Hands back rollouts computed earlier, in the order they were asked for.

    ``pdm_score_full_v2`` simulates a token's proposals and then, for original
    frames, the human trajectory. Both are produced up front for a whole group
    of tokens in one GPU call; this stands in for the simulator so the scoring
    path stays exactly as it is. The shape check is what catches the scoring
    code changing what it asks for.
    """

    def __init__(self, proposal_sampling, rollouts: List[npt.NDArray[np.float64]]):
        self.proposal_sampling = proposal_sampling
        self._rollouts = list(rollouts)
        self._next = 0

    def simulate_proposals(self, states, initial_ego_state):
        if self._next >= len(self._rollouts):
            raise AssertionError(
                "scoring asked for more rollouts than were batched; the batched "
                "path assumes one proposal call and at most one human call")
        out = self._rollouts[self._next]
        self._next += 1
        expected = states[:, : self.proposal_sampling.num_poses + 1].shape
        if out.shape != expected:
            raise AssertionError(f"batched rollout has shape {out.shape}, scoring wanted {expected}")
        return out

    def exhausted(self) -> bool:
        return self._next == len(self._rollouts)


def _simulate_group(
    simulator, metric_caches: List[MetricCache], vocab_trajectories, sampling
) -> List[List[npt.NDArray[np.float64]]]:
    """Roll out several tokens in one call and split the result back per token.

    A single token's 4096 proposals leave the GPU less than half busy. Stacking
    a group of them into one call is a pure throughput change -- every proposal
    is independent, and ``simulate_proposals_from_states`` gives each row its own
    starting state.

    Many workers share each card, so a group that fits on its own can still not
    fit beside its neighbours. Rather than size the group for the worst case and
    give up the throughput everywhere else, an out-of-memory group is halved and
    retried; a single token that still will not fit is a real failure.
    """
    if not metric_caches:
        return []
    try:
        return _simulate_group_once(simulator, metric_caches, vocab_trajectories, sampling)
    except RuntimeError as error:
        # Torch raises its own OutOfMemoryError when its allocator runs dry, but
        # a plain RuntimeError when the driver refuses an allocation underneath
        # it -- cuSOLVER's workspace does the latter. Both mean the same thing
        # here and both used to kill the whole run.
        if not _is_out_of_memory(error) or len(metric_caches) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(metric_caches) // 2
        logger.warning(
            f"out of memory rolling out {len(metric_caches)} tokens together; "
            f"retrying as {half} + {len(metric_caches) - half}")
        return (_simulate_group(simulator, metric_caches[:half], vocab_trajectories, sampling)
                + _simulate_group(simulator, metric_caches[half:], vocab_trajectories, sampling))


def _is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _simulate_group_once(
    simulator, metric_caches: List[MetricCache], vocab_trajectories, sampling
) -> List[List[npt.NDArray[np.float64]]]:
    blocks: List[npt.NDArray[np.float64]] = []
    starts: List[npt.NDArray[np.float64]] = []
    layout: List[List[int]] = []
    for cache in metric_caches:
        sizes: List[int] = []
        for states in (
            proposal_states_to_simulate(cache, vocab_trajectories, sampling),
            human_states_to_simulate(cache, sampling) if _needs_human(cache) else None,
        ):
            if states is None:
                continue
            blocks.append(states)
            starts.append(np.broadcast_to(
                ego_state_to_state_array(cache.ego_state), (states.shape[0], StateIndex.size())))
            sizes.append(states.shape[0])
        layout.append(sizes)

    try:
        simulated = simulator.simulate_proposals_from_states(
            np.concatenate(blocks, axis=0), np.concatenate(starts, axis=0))
    finally:
        # The result is already on the host. Six workers share each card, so a
        # worker that keeps its rollout workspace through its own scoring phase
        # -- four fifths of the cycle -- denies it to the others for that whole
        # time. Handing it back measured no slower and leaves the headroom that
        # keeps the group size viable.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_token: List[List[npt.NDArray[np.float64]]] = []
    offset = 0
    for sizes in layout:
        rollouts = []
        for size in sizes:
            rollouts.append(simulated[offset:offset + size])
            offset += size
        per_token.append(rollouts)
    assert offset == simulated.shape[0]
    return per_token


# Whether the scorer will actually consume a human rollout. The batching step
# pre-simulates one per ORIGINAL token, but pdm_score_full_v2 only uses it when
# scorer._config.human_penalty_filter is on. With the filter off the extra
# rollout is left unconsumed and _score_one raises "scoring used fewer rollouts
# than were batched". Set once per worker from the instantiated scorer.
_HUMAN_FILTER_ON = True


def _set_human_filter(scorer) -> None:
    global _HUMAN_FILTER_ON
    _HUMAN_FILTER_ON = bool(getattr(scorer._config, "human_penalty_filter", False))


def _needs_human(metric_cache: MetricCache) -> bool:
    """Whether scoring will also roll out the human trajectory for this token."""
    return _HUMAN_FILTER_ON and metric_cache.scene_type == SceneFrameType.ORIGINAL


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    # tokens = [t for a in args for t in a["tokens"]]
    tokens = [a["token"] for a in args]
    cfg: DictConfig = args[0]["cfg"]
    vocab_trajectories = args[0]["vocab"]
    dir = args[0]['dir']
    epdms_only = bool(cfg.get("epdms_only", False))

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    _set_human_filter(scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
    )
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    scene_tokens = set(scene_loader.tokens)
    cached_tokens = set(metric_cache_loader.tokens)
    tokens_to_evaluate = sorted(scene_tokens & cached_tokens)
    # Tokens without a metric cache used to vanish silently, which made the
    # final artifact impossible to reconcile against the index. Report them as
    # explicit failures instead.
    missing_cache = sorted(scene_tokens - cached_tokens)
    if missing_cache:
        logger.warning(
            f"{len(missing_cache)} tokens have no metric cache in thread_id={thread_id}, "
            f"node_id={node_id}; first={missing_cache[0]}"
        )

    pdm_results: List[Dict[str, Any]] = [
        {"token": token, "failed": True, "score": None, "reason": "missing metric cache"}
        for token in missing_cache
    ]

    # One token's proposals do not fill a modern GPU, so tokens are rolled out
    # in groups and scored one by one from the result. The group is only a
    # throughput device: every proposal is independent of the others, and a
    # token scores identically whichever group it lands in.
    group_size = max(1, int(cfg.get("simulate_group_size", 1)))
    done = 0
    for start in range(0, len(tokens_to_evaluate), group_size):
        group = tokens_to_evaluate[start:start + group_size]
        logger.info(
            f"Processing scenarios {done + 1}-{done + len(group)} / {len(tokens_to_evaluate)} "
            f"in thread_id={thread_id}, node_id={node_id}"
        )
        done += len(group)

        pending = [
            token for token in group
            if not os.path.exists(f'{trajpdm_root}/{dir}/{PER_TOKEN_DIRNAME}/{token}.pkl')
        ]
        caches = {token: _load_metric_cache(metric_cache_loader.metric_cache_paths[token])
                  for token in pending}
        rollouts = {}
        if group_size > 1 and caches and hasattr(simulator, "simulate_proposals_from_states"):
            batched = _simulate_group(
                simulator, [caches[t] for t in pending], vocab_trajectories,
                simulator.proposal_sampling)
            rollouts = dict(zip(pending, batched))

        for token in group:
            pdm_results.append(
                score_token(
                    token=token,
                    tmp_cache_path=f'{trajpdm_root}/{dir}/{PER_TOKEN_DIRNAME}/{token}.pkl',
                    epdms_only=epdms_only,
                    compute=lambda token=token: _score_one(
                        cache=caches.get(token) or _load_metric_cache(
                            metric_cache_loader.metric_cache_paths[token]),
                        rollouts=rollouts.get(token),
                        vocab_trajectories=vocab_trajectories,
                        simulator=simulator,
                        scorer=scorer,
                        traffic_agents_policy=traffic_agents_policy,
                    ),
                )
            )
    return pdm_results


def _score_one(cache, rollouts, vocab_trajectories, simulator, scorer, traffic_agents_policy):
    """Score one token, off a pre-computed rollout when the group produced one."""
    active = simulator if rollouts is None else _ReplaySimulator(
        simulator.proposal_sampling, rollouts)
    result = pdm_score_full_v2(
        metric_cache=cache,
        vocab_trajectories=vocab_trajectories,
        future_sampling=simulator.proposal_sampling,
        simulator=active,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )
    if rollouts is not None and not active.exhausted():
        raise AssertionError(
            "scoring used fewer rollouts than were batched for this token; "
            "the human-filter branch no longer matches _needs_human")
    return result


def _load_metric_cache(path: str) -> MetricCache:
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def score_token(
    token: str,
    tmp_cache_path: str,
    epdms_only: bool,
    compute: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Score one token, turning any failure into a row rather than raising.

    A failure must never abandon the rest of the worker's chunk. Returning
    ``None`` from the worker used to drop every remaining token and then crash
    ``worker_map``'s flatten step, losing a multi-day run at the very end.
    """
    score_row: Dict[str, Any] = {"token": token, "failed": False}
    try:
        if os.path.exists(tmp_cache_path):
            # The per-token cache always keeps the full component dict for
            # debugging; reduce it here so an EPDMS-only run does not ship
            # eight redundant arrays per token back to the driver.
            with open(tmp_cache_path, 'rb') as f:
                cached = pickle.load(f)
            score_row['score'] = cached['pdm_score'] if epdms_only else cached
            return score_row

        result = compute()
        # Cache the full components before reducing, so a later run can still
        # inspect them. Atomic, because a run killed mid-dump must not leave a
        # truncated cache that every later resume would load and fail on.
        os.makedirs(os.path.dirname(tmp_cache_path), exist_ok=True)
        part_path = f'{tmp_cache_path}.{os.getpid()}.part'
        with open(part_path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(part_path, tmp_cache_path)
        score_row['score'] = result['pdm_score'] if epdms_only else result
    except Exception:
        logger.warning(f"----------- Agent failed for token {token}:")
        traceback.print_exc()
        score_row['failed'] = True
        score_row['score'] = None
        score_row['reason'] = traceback.format_exc(limit=3)
    return score_row


if __name__ == "__main__":
    main()
