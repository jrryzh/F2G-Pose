import numpy as np
import pytest
from f2g_pose.preprocess import prepare,intrinsics_matrix
from f2g_pose.estimator import PoseEstimator


def sample():
    rgb=np.zeros((80,100,3),np.uint8);rgb[:,:,0]=255
    depth=np.full((80,100),1000,np.uint16)
    mask=np.zeros((80,100),np.uint8);mask[20:60,25:75]=255
    return rgb,depth,[100,100,50,40],mask


def test_units_center_color():
    rgb,depth,K,mask=sample()
    mm=prepare(rgb,depth,K,mask,.001)
    meter=prepare(rgb,depth.astype(np.float32)/1000,K,mask,1.)
    for key in ('points','rgb','rows','columns','center'):
        np.testing.assert_array_equal(getattr(mm,key),getattr(meter,key))
    assert np.all(mm.rgb[0]==1) and np.all(mm.rgb[1:]==0)
    np.testing.assert_allclose(np.median(mm.points,0),0,atol=1e-7)
    assert abs(mm.center[2]-1)<1e-7


@pytest.mark.parametrize('case',['empty','few','mismatch','bad_K','bad_scale','nonbinary','bad_rgb'])
def test_invalid_inputs(case):
    rgb,depth,K,mask=sample();scale=.001
    if case=='empty':mask[:]=0
    if case=='few':depth[:]=0;depth[30,30]=1000
    if case=='mismatch':depth=depth[:-1]
    if case=='bad_K':K=[0,100,50,40]
    if case=='bad_scale':scale=0
    if case=='nonbinary':mask[30,30]=2
    if case=='bad_rgb':rgb=rgb.astype(float)
    with pytest.raises(ValueError):prepare(rgb,depth,K,mask,scale)


def test_translation_and_shape_use_observed_center():
    import torch
    center=np.array([1,2,3],np.float32)
    ret=(None,torch.zeros(1,8,3),torch.tensor([[1.,0,0,0,1,0]]),torch.tensor([[.1,.2,.3]]),torch.ones(1,3))
    result=PoseEstimator.decode(ret,center)
    np.testing.assert_allclose(result['translation_m'],center+[.1,.2,.3],rtol=1e-6)
    np.testing.assert_array_equal(result['points_cam_m'],np.tile(center,(8,1)))
    np.testing.assert_array_equal(result['rotation'],np.eye(3))
