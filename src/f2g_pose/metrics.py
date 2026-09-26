"""Exact midpoint-grid AUC/VUS used by cutoop 0.1.0, vectorized over samples."""
import numpy as np

IOU_RANGES = [(0.25, 1., .075), (.5, 1., .005), (.75, 1., .0025)]
POSE_RANGES = [((0.,5.,.05),(0.,2.,.02)),((0.,5.,.05),(0.,5.,.05)),
               ((0.,10.,.1),(0.,2.,.02)),((0.,10.,.1),(0.,5.,.05))]


def grid_fraction(values, range_, greater):
    lo, hi, step = range_
    grid = np.arange(lo, hi, step) + step/2
    # Strict comparisons match cutoop: IoU > threshold; error < threshold.
    counts = np.searchsorted(grid, values, side='left') if greater else len(grid)-np.searchsorted(grid, values, side='right')
    return counts * step / (hi-lo)


def summarize(labels, iou, deg, cm):
    labels,iou,deg,cm=map(np.asarray,(labels,iou,deg,cm))
    if not len(labels):raise ValueError('No evaluated samples.')
    per={}
    for label in np.unique(labels):
        sel=labels==label;i,d,t=iou[sel],deg[sel],cm[sel]
        per[str(int(label))]=dict(count=int(sel.sum()),iou_mean=float(i.mean()),
            iou_acc=[float((i>x).mean()) for x in (.25,.5,.75)],
            iou_auc=[float(grid_fraction(i,r,True).mean()) for r in IOU_RANGES],
            deg_mean=float(d.mean()),sht_mean=float(t.mean()),
            pose_acc=[float(((d<a)&(t<b)).mean()) for a,b in ((5,2),(5,5),(10,2),(10,5))],
            pose_auc=[float((grid_fraction(d,a,False)*grid_fraction(t,b,False)).mean()) for a,b in POSE_RANGES])
    overall={k:np.mean([v[k] for v in per.values()],axis=0).tolist() for k in next(iter(per.values())) if k!='count'}
    return dict(class_means=overall,class_metrics=per,occurred_classes=len(per),sample_count=len(labels),
                averaging='unweighted mean over occurred classes',rotation_unit='degree',translation_error_unit='cm',
                iou_auc_ranges=IOU_RANGES,pose_auc_ranges=POSE_RANGES)


def compute_criterion(payload):
    import contextlib,io
    from cutoop.eval_utils import DetectMatch
    from cutoop.rotation import SymLabel
    payload=dict(payload)
    payload['gt_sym_labels']=[SymLabel(**s) for s in payload['gt_sym_labels']]
    with contextlib.redirect_stderr(io.StringIO()):
        dm=DetectMatch(**payload).calibrate_rotation(silent=True)
        return dm.criterion()
