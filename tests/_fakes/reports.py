"""The report dicts the comparison gate is fed.

One shape, overridable per call: `run_context` keys are passed through as
keyword arguments so a caller that depends on a particular context value says
so at the call site.
"""

from __future__ import annotations

_CTX = {"algo": "dqn", "game": "ALE/Pong-v5", "chip": "2", "update_mode": "eager",
        "seed": 0, "total_frames": 3000, "ttnn_version": "0.77.0",
        "ttnn_build": "wheel", "code_fingerprint": "aaaa", "load_average": 1.0}


def perf_report(median_s, *, n=100, mean_factor=1.0, **ctx):
    return {
        "run_context": {**_CTX, **ctx},
        "kinds": {"with_update": {"n": n, "median_s": median_s,
                                  "mean_s": median_s * mean_factor,
                                  "p10_s": median_s, "p90_s": median_s,
                                  "sample_sufficient": n >= 30}},
    }


def throughput_report(*, env_fps, train_tps=1000.0, iters=200, load=1.0, **ctx):
    context = {"algo": "ppo", "game": "ALE/Pong-v5", "chip": "0",
               "update_mode": "eager", "seed": 0, "total_frames": 16384,
               "ttnn_version": "0.77.0", "ttnn_build": "wheel",
               "code_fingerprint": "abc", "load_average": load}
    context.update(ctx)
    return {
        "run_context": context,
        "kinds": {"ppo_update": {"n": iters, "median_s": 0.076, "mean_s": 0.077,
                                 "sample_sufficient": iters >= 30}},
        "throughput": {"env_fps": env_fps, "train_tps": train_tps,
                       "measured_iterations": iters, "env_frames": 16384,
                       "transitions_trained": 65536, "measured_wall_s": 22.0,
                       "transitions_per_frame": 4.0, "shape": "n_envs=8"},
    }
