import numpy as np
import pytest
from f2g_pose.demo import image_digest,confirmed_mask
from f2g_pose.io import save_prediction


def test_sam_confirmation_is_bound_to_current_image():
    a=np.zeros((80,100,3),np.uint8);b=a.copy();b[0,0]=1
    state={'digest':image_digest(a),'mask':np.ones((80,100),bool),'confirmed':False}
    with pytest.raises(ValueError):confirmed_mask(a,None,state)
    state['confirmed']=True
    assert confirmed_mask(a,None,state).all()
    with pytest.raises(ValueError):confirmed_mask(b,None,state)
    with pytest.raises(ValueError):confirmed_mask(a,None,{})


def test_download_directories_never_overwrite(tmp_path):
    import uuid
    one=save_prediction({'translation_m':[1,2,3]},tmp_path/uuid.uuid4().hex)
    two=save_prediction({'translation_m':[4,5,6]},tmp_path/uuid.uuid4().hex)
    assert one!=two and (one/'pose.json').read_text()!=(two/'pose.json').read_text()
    with pytest.raises(FileExistsError):save_prediction({},one)
