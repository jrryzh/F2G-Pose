import time
from pathlib import Path
import numpy as np
from .io import write_json
from .evaluate import environment,sha256


def benchmark(estimator, inputs, args):
    import torch
    if args.warmup<1 or args.iterations<1:raise ValueError('Warmup and iterations must be positive.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    preprocessing=[]
    for _ in range(args.iterations):
        start=time.perf_counter();obs=estimator.preprocess(*inputs);preprocessing.append((time.perf_counter()-start)*1000)
    tensors=obs.tensors(estimator.device)
    result=dict(environment=environment(estimator.device),batch_size=1,warmup=args.warmup,iterations=args.iterations,
                checkpoint_sha256=sha256(estimator.checkpoint),precision=args.precision,sam_ms=None,
                note='SAM measured separately. Model timings exclude file IO, preprocessing, host/device copies and output rendering.')
    def statistics(xs):
        return dict(mean_ms=float(np.mean(xs)),median_ms=float(np.median(xs)),p95_ms=float(np.percentile(xs,95)),std_ms=float(np.std(xs)))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16,enabled=args.precision!='fp32'):
        for shape in (True,False):
            for _ in range(args.warmup):estimator.model(*tensors,return_shape=shape)
            torch.cuda.synchronize(estimator.device);torch.cuda.reset_peak_memory_stats(estimator.device)
            elapsed=[]
            for _ in range(args.iterations):
                torch.cuda.synchronize(estimator.device);start=time.perf_counter()
                estimator.model(*tensors,return_shape=shape)
                torch.cuda.synchronize(estimator.device);elapsed.append((time.perf_counter()-start)*1000)
            values=statistics(elapsed);values['model_fps']=1000/values['mean_ms']
            values['peak_allocated_bytes']=torch.cuda.max_memory_allocated(estimator.device)
            values['model_plus_preprocess_fps']=1000/(values['mean_ms']+np.mean(preprocessing))
            result['full' if shape else 'pose_only']=values
    result['preprocessing']=statistics(preprocessing)
    write_json(out/'benchmark.json',result)
    print(result)
