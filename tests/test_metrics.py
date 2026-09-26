import numpy as np
from f2g_pose.metrics import summarize,IOU_RANGES,POSE_RANGES


def test_metrics_match_cutoop_threshold_boundaries_and_macro_average():
    from cutoop.eval_utils import DetectMatch
    from cutoop.rotation import SymLabel
    rng=np.random.default_rng(5)
    labels=np.array([0]*3+[1]*14)
    iou=np.r_[.2875,.5025,.75125,rng.random(14)]
    deg=np.r_[.025,.075,.05,rng.random(14)*20]
    cm=np.r_[.01,.025,.05,rng.random(14)*10]
    T=np.tile(np.eye(4),(len(labels),1,1));size=np.ones((len(labels),3))
    dm=DetectMatch(gt_affine=T,gt_size=size,gt_sym_labels=[SymLabel(False,'none','none','none')]*len(labels),gt_class_labels=labels,pred_affine=T,pred_size=size)
    reference=dm.metrics(criterion=(iou,deg,cm),iou_auc_ranges=IOU_RANGES,pose_auc_ranges=POSE_RANGES)
    actual=summarize(labels,iou,deg,cm)
    for label,m in actual['class_metrics'].items():
        r=reference.class_metrics[int(label)]
        np.testing.assert_allclose(m['iou_auc'],[v.auc for v in r.iou_auc],rtol=0,atol=1e-14)
        np.testing.assert_allclose(m['pose_auc'],[v.auc for v in r.pose_auc],rtol=0,atol=1e-14)
    np.testing.assert_allclose(actual['class_means']['iou_auc'],[v.auc for v in reference.class_means.iou_auc],atol=1e-14)
    np.testing.assert_allclose(actual['class_means']['pose_auc'],[v.auc for v in reference.class_means.pose_auc],atol=1e-14)
    assert actual['class_means']['deg_mean']!=np.mean(deg)
