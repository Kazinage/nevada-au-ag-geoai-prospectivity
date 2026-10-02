import numpy as np
from geoai_nevada.aoa import AreaOfApplicability
from geoai_nevada.evaluation import topk_capture
from geoai_nevada.spatial import exclusion_mask

def test_core_components():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    aoa = AreaOfApplicability(0.95).fit(X)
    result = aoa.predict(X[:10])
    assert result.inside.dtype == bool
    assert 0 < result.threshold
    mask = exclusion_mask(np.array([[0.,0.],[10.,0.]]), np.array([[0.,0.]]), 5.)
    assert mask.tolist() == [False, True]
    capture = topk_capture(np.array([1,0,1,0]), np.array([.9,.8,.7,.1]), fractions=(.5,))
    assert capture[.5] == .5
