import pytest
np=pytest.importorskip('numpy')
from hotc_tracker.representations import robust_band_rgb
from hotc_tracker.tracking.box_utils import mask_to_xywh
from hotc_tracker.evaluation import ope_metrics
def test_synthetic_integration_test():
 cube=np.zeros((20,20,16),np.float32);cube[5:9,7:12]=5
 rgb=robust_band_rgb(cube,(1,8,14)); box=mask_to_xywh(rgb[...,0]>0); result=ope_metrics([box],[box]); assert result['success_auc']==1
