# Compute-efficiency report

## Steady-state throughput

- Samples/s median: **60,800** (p10 60,739, p90 60,859)
- Seconds/iteration median: **8.623** (p10 8.615, p90 8.632)
- Timed iterations: 45 (iterations 1–5 excluded)

## Synchronized phase profile

- Rollout: 59.3%
- PPO update: 40.4%
- Host/bookkeeping gap: 0.3%
- Warm compilation cache reduced the first timed iteration by 55.3 s

## GPU telemetry

- GPU utilization median/peak: 95.0% / 100.0%
- Peak memory: 61,537 MiB
- Peak memory fraction: 75.5%
- Power median/peak: 333 W / 654 W

## Out-of-band events

- Evaluation Seconds: 23.117 s

## Device trace

- Unavailable on this node: three isolated attempts (a three-iteration default trace, a three-iteration trace without Python call tracing, and a one-iteration device-only trace without HLO protos) each exited with SIGSEGV at the first traced production rollout under JAX/jaxlib 0.10.2 and NVIDIA driver 580.159.03. A minimal H100 matrix-multiply trace succeeded, isolating the failure to profiling this full pmap workload. No valid production XPlane artifact was produced; kernel-level launch gaps therefore cannot be claimed from this run.

## Recommendations

- **not-worthwhile — host dispatch and bookkeeping:** Only 0.3% of synchronized iteration time is outside rollout/update
- **not-worthwhile — single-GPU occupancy:** Sampled GPU utilization is already high
- **measured-safe — persistent JAX compilation cache:** The diagnostic passes reuse the production compilation cache without changing the training program

## Limitations

- One H100 cannot characterize NCCL or multi-device pmap scaling.
- nvidia-smi samples are coarse and cannot identify individual kernel launch gaps.
- No change to model math, sample order, precision, optimizer order, or curriculum was benchmarked.

## Python profile excerpt

```text
Sat Aug  1 23:33:16 2026    /home/dev/generals-d384-smoke-artifacts/phase_profile.pstats

         40112797 function calls (39369559 primitive calls) in 181.480 seconds

   Ordered by: cumulative time
   List reduced from 8850 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   1707/1    0.053    0.000  181.669  181.669 {built-in method builtins.exec}
        1    0.000    0.000  181.669  181.669 <string>:1(<module>)
        1    0.000    0.000  181.669  181.669 <frozen runpy>:201(run_module)
        1    0.000    0.000  178.978  178.978 <frozen runpy>:65(_run_code)
        1    0.000    0.000  178.978  178.978 /home/dev/generals-d384-smoke-src/generals/training/train.py:1(<module>)
        1    0.020    0.020  178.870  178.870 /home/dev/generals-d384-smoke-src/generals/training/train.py:918(main)
        1    0.049    0.049  178.847  178.847 /home/dev/generals-d384-smoke-src/generals/training/train.py:445(train)
       49   83.935    1.713  117.572    2.399 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pmap.py:315(wrapped)
40063/866    0.045    0.000   91.646    0.106 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/traceback_util.py:191(reraise_with_filtered_traceback)
37870/713    0.150    0.000   91.225    0.128 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pjit.py:250(cache_miss)
37870/21037    0.179    0.000   81.795    0.004 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pjit.py:130(_run_python_pjit)
      418    0.011    0.000   72.677    0.174 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pjit.py:1137(_pjit_call_impl_python)
5475/3151    0.043    0.000   70.806    0.022 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/profiler.py:417(wrapper)
151733/46048    0.985    0.000   41.828    0.001 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/core.py:660(bind)
152930/74389    0.288    0.000   37.423    0.001 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/core.py:709(bind_with_trace)
      418    0.006    0.000   36.217    0.087 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:1209(compile)
      418    0.037    0.000   36.210    0.087 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:1686(from_hlo)
      418    0.011    0.000   35.946    0.086 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:1499(_cached_compilation)
      418    0.013    0.000   35.764    0.086 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/compiler.py:387(compile_or_get_cached)
 4141/698    0.008    0.000   31.545    0.045 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/linear_util.py:210(call_wrapped)
        2    0.002    0.001   30.933   15.467 /home/dev/generals-d384-smoke-src/generals/core/env.py:223(reset)
      380    0.005    0.000   27.896    0.073 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/compiler.py:727(_compile_and_write_cache)
      380   27.875    0.073   27.876    0.073 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/compiler.py:292(backend_compile_and_load)
  691/223    0.017    0.000   25.751    0.115 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/api.py:1146(vmap_f)
  691/223    0.008    0.000   25.555    0.115 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/batching.py:336(_batch_outer)
  691/223    0.018    0.000   25.547    0.115 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/batching.py:344(_batch_inner)
  691/223    0.003    0.000   25.452    0.114 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/batching.py:118(flatten_fun_for_vmap)
     4353    0.015    0.000   24.064    0.006 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/core.py:1285(process_primitive)
39719/12722    0.369    0.000   20.180    0.002 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/batching.py:255(process_primitive)
      418   18.052    0.043   18.958    0.045 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:374(__call__)
37870/17541    0.335    0.000   18.859    0.001 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pjit.py:616(_infer_params)
 2712/749    0.158    0.000   18.209    0.024 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pjit.py:466(_trace_for_jit)
 2993/885    0.098    0.000   18.007    0.020 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/partial_eval.py:2268(trace_to_jaxpr)
      418    0.012    0.000   17.360    0.042 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pjit.py:1082(_resolve_and_lower)
      418    0.002    0.000   17.307    0.041 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/pjit.py:1247(_pjit_lower)
      418    0.067    0.000   17.300    0.041 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:954(lower_sharding_computation)
     3639    0.365    0.000   16.743    0.005 /home/dev/generals-d384-smoke-src/.venv/lib/python3.11/site-packages/jax/_src/dispatch.py:81(apply_primitive)
```
