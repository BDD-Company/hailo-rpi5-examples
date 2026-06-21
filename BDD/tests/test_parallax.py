import math
from guidance.parallax import closest_point_two_rays, triangulate_parallax


def _unit(v):
    m = math.sqrt(sum(c * c for c in v))
    return tuple(c / m for c in v)


def test_closest_point_recovers_intersection():
    # target at (50,5,0); two cameras at origin and (0,10,0) looking at it
    tgt = (50.0, 5.0, 0.0)
    q1 = (0.0, 0.0, 0.0); d1 = _unit((tgt[0]-q1[0], tgt[1]-q1[1], tgt[2]-q1[2]))
    q2 = (0.0, 10.0, 0.0); d2 = _unit((tgt[0]-q2[0], tgt[1]-q2[1], tgt[2]-q2[2]))
    res = closest_point_two_rays(q1, d1, q2, d2)
    assert res is not None
    pt, sin2, miss = res
    assert miss < 1e-6
    assert all(abs(pt[i] - tgt[i]) < 1e-4 for i in range(3))


def test_parallel_rays_return_none():
    assert closest_point_two_rays((0.,0.,0.), (1.,0.,0.), (0.,5.,0.), (1.,0.,0.)) is None


def test_behind_camera_returns_none():
    # both rays point away from the would-be intersection
    assert closest_point_two_rays((0.,0.,0.), (-1.,0.,0.), (0.,10.,0.), (-1.,0.,0.)) is None


def test_triangulate_compensates_target_motion():
    # target moves +east at 8 m/s; x(t)=x0+v*t with x0 at t_now=0
    v = (0.0, 8.0, 0.0); x0 = (50.0, 0.0, 0.0)
    p_now = (0.0, 0.0, 0.0); t_now = 0.0
    p_old = (0.0, 12.0, 0.0); t_old = -0.4   # target was at x0 + v*(-0.4) = (50,-3.2,0)
    tgt_old = (x0[0]+v[0]*t_old, x0[1]+v[1]*t_old, x0[2]+v[2]*t_old)
    los_now = _unit((x0[0]-p_now[0], x0[1]-p_now[1], x0[2]-p_now[2]))
    los_old = _unit((tgt_old[0]-p_old[0], tgt_old[1]-p_old[1], tgt_old[2]-p_old[2]))
    res = triangulate_parallax(p_now, los_now, t_now, p_old, los_old, t_old, v,
                               min_sin2=0.02, min_baseline_m=1.0, max_miss_m=8.0)
    assert res is not None
    pt, prange = res
    assert all(abs(pt[i] - x0[i]) < 0.2 for i in range(3))      # recovers x(t_now)=x0
    assert abs(prange - 50.0) < 0.5


def test_triangulate_rejects_small_baseline():
    v = (0.0, 0.0, 0.0)
    p_now = (0.0, 0.0, 0.0); p_old = (0.0, 0.1, 0.0)            # 0.1 m baseline < min
    los = _unit((50.0, 0.0, 0.0))
    assert triangulate_parallax(p_now, los, 0.0, p_old, los, -0.1, v,
                                min_sin2=0.02, min_baseline_m=1.0, max_miss_m=8.0) is None
