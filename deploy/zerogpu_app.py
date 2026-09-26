"""ZeroGPU deployment candidate. No CUDA initialization during module import."""
import os
from types import SimpleNamespace
import spaces
from f2g_pose.demo import build_demo


class DeferredEstimator:
    device='cuda:0'
    def predict(self,*args,**kwargs):
        # Called only by the @spaces.GPU-wrapped Gradio callback.
        from f2g_pose import PoseEstimator
        estimator=PoseEstimator(os.environ['F2G_CHECKPOINT'],
            radio_repository=os.environ['F2G_RADIO_REPOSITORY'],
            radio_checkpoint=os.environ['F2G_RADIO_CHECKPOINT'])
        return estimator.predict(*args,**kwargs)


options=SimpleNamespace(sam_checkpoint=os.environ.get('F2G_SAM_CHECKPOINT'),output_root=os.environ.get('F2G_OUTPUT_ROOT','demo_outputs'))
demo=build_demo(DeferredEstimator(),options,gpu_decorator=spaces.GPU(duration=60))
if __name__=='__main__':demo.queue().launch()
