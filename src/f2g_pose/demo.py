"""Session-local mask preview, optional SAM prompting, and pose visualization."""
import hashlib
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from PIL import Image
from .io import load_depth,jsonable,save_prediction


def image_digest(rgb):
    a=np.asarray(rgb)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


class Segmenter:
    def __init__(self,checkpoint,device,cache_model=True):
        self.checkpoint=checkpoint;self.device=device;self.cache_model=cache_model
        self.predictor=None;self.lock=threading.RLock()
    def predict(self,rgb,points,box=None):
        import torch
        with self.lock,torch.inference_mode():
            from segment_anything import sam_model_registry,SamPredictor
            if self.predictor is None:
                self.predictor=SamPredictor(sam_model_registry['vit_b'](checkpoint=self.checkpoint).to(self.device).eval())
            torch.cuda.synchronize(self.device);start=time.perf_counter()
            self.predictor.set_image(rgb)
            masks,scores,_=self.predictor.predict(
                point_coords=np.asarray(points,dtype=np.float32) if points else None,
                point_labels=np.ones(len(points),dtype=np.int32) if points else None,
                box=np.asarray(box,dtype=np.float32) if box is not None else None,multimask_output=True)
            mask=masks[int(np.argmax(scores))]
            torch.cuda.synchronize(self.device)
            elapsed=(time.perf_counter()-start)*1000
            if not self.cache_model:self.predictor=None
            return mask,elapsed


def overlay(rgb,mask):
    out=rgb.copy();out[mask]=(.55*out[mask]+.45*np.array([35,210,130])).astype(np.uint8)
    return out


def visualization(rgb,K,result):
    import cv2
    import plotly.graph_objects as go
    from itertools import product
    K=np.asarray(K);R=result['rotation'];t=result['translation_m'];size=result['size_m']
    signs=np.array(list(product([-1,1],repeat=3)))
    corners=signs*size/2@R.T+t
    axes=np.vstack([t,t+R[:,0]*size.max()*.6,t+R[:,1]*size.max()*.6,t+R[:,2]*size.max()*.6])
    def project(points):
        uv=points@K.T;return np.rint(uv[:,:2]/uv[:,2:]).astype(int)
    out=rgb.copy();fig=go.Figure()
    if 'points_cam_m' in result:
        points=result['points_cam_m'];fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',marker=dict(size=2,color=points[:,2],colorscale='Viridis'),name='Completed shape'))
    edges=[(a,b) for a in range(8) for b in range(a+1,8) if np.sum(signs[a]!=signs[b])==1]
    for a,b in edges:
        pair=corners[[a,b]]
        fig.add_trace(go.Scatter3d(x=pair[:,0],y=pair[:,1],z=pair[:,2],mode='lines',line=dict(color='orange'),showlegend=False))
        if np.all(pair[:,2]>0):
            uv=project(pair);cv2.line(out,tuple(uv[0]),tuple(uv[1]),(255,180,25),2)
    for i,(name,color,rgb_color) in enumerate([('x','red',(255,60,60)),('y','green',(60,230,60)),('z','blue',(60,120,255))],1):
        pair=axes[[0,i]]
        fig.add_trace(go.Scatter3d(x=pair[:,0],y=pair[:,1],z=pair[:,2],mode='lines',line=dict(color=color,width=7),name=name))
        if np.all(pair[:,2]>0):
            uv=project(pair);cv2.arrowedLine(out,tuple(uv[0]),tuple(uv[1]),rgb_color,3)
    fig.update_layout(scene=dict(aspectmode='data',xaxis_title='Camera X (m)',yaxis_title='Camera Y (m)',zaxis_title='Camera Z (m)'),margin=dict(l=0,r=0,t=20,b=0))
    return out,fig


def confirmed_mask(rgb,uploaded,state):
    if uploaded:
        mask=np.asarray(Image.open(uploaded))
        return mask
    if not state or state.get('digest')!=image_digest(rgb) or not state.get('confirmed'):
        raise ValueError('Upload a binary mask, or preview and confirm a SAM mask for the current RGB image.')
    return state['mask']


def build_demo(estimator,args,gpu_decorator=None):
    import gradio as gr
    # Plotly inspects pandas through sys.modules. Finish this import before
    # Gradio worker threads can encounter a partially initialized module.
    import pandas  # noqa: F401
    segmenter=Segmenter(args.sam_checkpoint,estimator.device,cache_model=gpu_decorator is None) if args.sam_checkpoint else None
    output_root=Path(args.output_root).resolve();output_root.mkdir(parents=True,exist_ok=True)
    with gr.Blocks(title='F2G-Pose',delete_cache=(3600,86400),analytics_enabled=False) as demo:
        gr.Markdown('# F2G-Pose\nUpload aligned RGB and raw depth for **one object**. Use a binary mask or select the target with SAM. Camera coordinates: X right, Y down, Z forward. Translation and dimensions are in meters.')
        state=gr.State({});points=gr.State([])
        with gr.Row():
            rgb=gr.Image(label='RGB (PNG/JPEG); click target for SAM',type='numpy',image_mode='RGB',sources=['upload'])
            preview=gr.Image(label='Mask preview',interactive=False)
        with gr.Row():
            depth=gr.File(label='Raw depth: uint16 PNG or float NPY',file_types=['.png','.npy'])
            mask=gr.File(label='Optional binary mask: 0/1 or 0/255 PNG',file_types=['.png'])
            unit=gr.Dropdown(['Millimeters','Meters','Custom'],value=None,label='Choose raw depth units (required)')
            custom=gr.Number(value=.001,label='Custom: meters per raw unit')
        with gr.Row():
            fx=gr.Number(label='fx',value=600);fy=gr.Number(label='fy',value=600)
            cx=gr.Number(label='cx',value=320);cy=gr.Number(label='cy',value=240)
        Ktext=gr.Textbox(label='Optional 3×3 K (nine numbers, overrides fx/fy/cx/cy)')
        with gr.Row():
            box=gr.Textbox(label='Optional SAM box: x_min y_min x_max y_max')
            segment=gr.Button('Preview SAM mask',interactive=segmenter is not None)
            confirm=gr.Button('Confirm preview mask')
            clear=gr.Button('Clear selection')
        status=gr.Textbox(label='Status',interactive=False)
        shape=gr.Checkbox(value=True,label='Complete shape')
        predict=gr.Button('Estimate pose',variant='primary')
        with gr.Row():
            rendered=gr.Image(label='Pose axes and 3D box',interactive=False)
            plot=gr.Plot(label='Camera-space shape and box')
        pose=gr.JSON(label='T_cam_obj and metric dimensions')
        downloads=gr.Files(label='Download JSON / PLY')
        def reset():return {},[],None,'Selection cleared'
        rgb.upload(reset,outputs=[state,points,preview,status])
        clear.click(reset,outputs=[state,points,preview,status])
        def select_point(existing,evt:gr.SelectData):
            updated=list(existing)+[list(evt.index)]
            return updated,{},f'{len(updated)} foreground point(s); preview SAM mask next.'
        rgb.select(select_point,inputs=[points],outputs=[points,state,status])
        def segment_target(image,pts,box_text):
            if image is None:raise gr.Error('Upload RGB first.')
            bounds=None
            if box_text.strip():
                try:bounds=[float(x) for x in box_text.replace(',',' ').split()]
                except ValueError:raise gr.Error('Box requires four numeric coordinates.')
                if len(bounds)!=4 or not np.isfinite(bounds).all() or not (0<=bounds[0]<bounds[2]<=image.shape[1] and 0<=bounds[1]<bounds[3]<=image.shape[0]):raise gr.Error('Box must be ordered and inside the RGB image.')
            if not pts and bounds is None:raise gr.Error('Click the target or enter a box.')
            selected,ms=segmenter.predict(image,pts,bounds)
            return overlay(image,selected),dict(mask=selected,digest=image_digest(image),confirmed=False,sam_ms=ms),f'SAM took {ms:.1f} ms. Inspect the mask, then confirm it.'
        segment.click(gpu_decorator(segment_target) if gpu_decorator else segment_target,inputs=[rgb,points,box],outputs=[preview,state,status],concurrency_limit=1)
        def confirm_target(image,current):
            if image is None or not current or current.get('digest')!=image_digest(image):raise gr.Error('Preview a SAM mask for this RGB image first.')
            current=dict(current,confirmed=True);return current,'Mask confirmed; ready for pose estimation.'
        confirm.click(confirm_target,inputs=[rgb,state],outputs=[state,status])
        def preview_uploaded(image,mask_path):
            if image is None or not mask_path:return None,{}
            a=np.asarray(Image.open(mask_path))
            if a.shape!=image.shape[:2] or not np.isin(a,[0,1,255]).all():raise gr.Error('Binary mask must match RGB dimensions.')
            return overlay(image,a>0),{}
        mask.change(preview_uploaded,inputs=[rgb,mask],outputs=[preview,state])
        def run(image,depth_path,mask_path,units,custom_scale,a,b,c,d,kmatrix,current,want_shape):
            if image is None or not depth_path:raise gr.Error('Upload aligned RGB and raw depth first.')
            try:
                if units not in ('Millimeters','Meters','Custom'):
                    raise ValueError('Choose the raw depth units before estimating pose.')
                z=load_depth(depth_path)
                selected=confirmed_mask(image,mask_path,current)
                K=np.array([[a,0,c],[0,b,d],[0,0,1]],np.float32)
                if kmatrix.strip():K=np.array([float(v) for v in kmatrix.replace(',',' ').split()]).reshape(3,3)
                scale={'Millimeters':.001,'Meters':1.,'Custom':custom_scale}[units]
                start=time.perf_counter();result=estimator.predict(image,z,K,selected,scale,return_shape=want_shape)
                ms=(time.perf_counter()-start)*1000
                request_id=uuid.uuid4().hex
                target=save_prediction(result,output_root/request_id)
                # Distinct basenames also prevent Gradio's content cache from
                # merging identical downloads from separate requests.
                for file in list(target.iterdir()):
                    file.rename(file.with_name(f'{file.stem}_{request_id}{file.suffix}'))
                rendered,figure=visualization(image,K,result)
                return rendered,figure,jsonable({k:v for k,v in result.items() if k!='points_cam_m'}),[str(p) for p in target.iterdir()],f'Pose + preprocessing: {ms:.1f} ms. SAM timing is separate.'
            except (ValueError,RuntimeError) as exc:raise gr.Error(str(exc))
        predict.click(gpu_decorator(run) if gpu_decorator else run,inputs=[rgb,depth,mask,unit,custom,fx,fy,cx,cy,Ktext,state,shape],outputs=[rendered,plot,pose,downloads,status],concurrency_limit=1)
    return demo


def launch(estimator,args):
    build_demo(estimator,args).queue().launch(server_name=args.host,server_port=args.port,share=False)
