import pytest
from hotc_tracker.data.sequences import parse_hotc_id
def test_final_underscore_is_frame():
 x=parse_hotc_id('nir-my_scene_name_12');assert (x.sensor,x.sequence,x.frame)==('nir','my_scene_name',12)
def test_bad_id():
 with pytest.raises(ValueError): parse_hotc_id('nir-a_nope')
