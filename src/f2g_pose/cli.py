"""One command for training, evaluation, inference, demo and benchmarking."""
import argparse
import os


def positive_int(value):
    value=int(value)
    if value<1:raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def parser():
    p=argparse.ArgumentParser(prog='f2g-pose')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('train','eval','predict','demo','benchmark'):
        s=sub.add_parser(name)
        s.add_argument('--checkpoint',required=name!='train')
        s.add_argument('--config',required=name=='train')
        s.add_argument('--radio-repository',default=os.getenv('F2G_RADIO_REPOSITORY'))
        s.add_argument('--radio-checkpoint',default=os.getenv('F2G_RADIO_CHECKPOINT'))
        s.add_argument('--device',default='cuda:0')
        s.add_argument('--cache-dir',help='Root for Torch/HF/Gradio caches')
        s.add_argument('--seed',type=int,default=0)
        if name!='demo':s.add_argument('--output',required=True)
        if name in ('predict','benchmark'):
            for key in ('rgb','depth','mask','intrinsics'):s.add_argument('--'+key,required=True)
            s.add_argument('--depth-scale',type=float,required=True,help='Meters per raw depth unit (0.001 for mm, 1 for m)')
        if name=='predict':s.add_argument('--pose-only',action='store_true')
        if name=='benchmark':
            s.add_argument('--warmup',type=int,default=30);s.add_argument('--iterations',type=int,default=200)
        if name in ('eval','benchmark'):
            s.add_argument('--precision',choices=('fp32','bf16','fp16'),default='fp32',help='Model autocast precision; historical validation/evaluation used bf16')
        if name in ('train','eval'):
            s.add_argument('--data-root',required=True)
            s.add_argument('--object-meta')
            s.add_argument('--workers',type=int,default=8)
            s.add_argument('--batch-size',type=positive_int,default=16 if name=='train' else 32)
            s.add_argument('--limit',type=positive_int)
        if name=='eval':
            s.add_argument('--split',choices=('sope','rope'),required=True)
            s.add_argument('--protocol',choices=('historical','full'),required=True)
            s.add_argument('--metric-workers',type=positive_int,default=4)
            s.add_argument('--manifest',help='Reuse an existing candidate JSONL manifest; preserves its order')
            s.add_argument('--metadata-exclusions',help='Metadata exclusions belonging to the reused manifest')
            s.add_argument('--shard-count',type=positive_int,default=1)
            s.add_argument('--shard-index',type=int,default=0)
        if name=='train':
            s.add_argument('--shapes-root',required=True)
            s.add_argument('--epochs',type=positive_int)
            s.add_argument('--max-steps',type=positive_int)
            s.add_argument('--resume')
            s.add_argument('--accumulation',type=positive_int,default=1)
        if name=='demo':
            s.add_argument('--host',default='127.0.0.1');s.add_argument('--port',type=int,default=7860)
            s.add_argument('--sam-checkpoint');s.add_argument('--output-root',default='demo_outputs')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    if args.cache_dir:
        from pathlib import Path
        cache=Path(args.cache_dir).resolve()
        os.environ['TORCH_HOME']=str(cache/'torch')
        os.environ['HF_HOME']=str(cache/'huggingface')
        os.environ['GRADIO_TEMP_DIR']=str(cache/'gradio')
    if args.command=='eval':
        from .evaluate import evaluate
        return evaluate(args)
    if args.command=='train':
        from .train import train
        return train(args)
    from .estimator import PoseEstimator
    estimator=PoseEstimator(args.checkpoint,config=args.config,radio_repository=args.radio_repository,
                            radio_checkpoint=args.radio_checkpoint,device=args.device,seed=args.seed)
    if args.command=='demo':
        from .demo import launch
        return launch(estimator,args)
    from .io import load_inputs,save_prediction
    rgb,depth,K,mask=load_inputs(args.rgb,args.depth,args.mask,args.intrinsics)
    if args.command=='predict':
        result=estimator.predict(rgb,depth,K,mask,args.depth_scale,return_shape=not args.pose_only)
        save_prediction(result,args.output)
    elif args.command=='benchmark':
        from .benchmark import benchmark
        benchmark(estimator,(rgb,depth,K,mask,args.depth_scale),args)

if __name__=='__main__':main()
