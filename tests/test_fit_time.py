"""The reader's fit counts real time (reader/fit.py): the content grid, the change costs, the shake measure."""
import numpy as np
import torch

from video2rig.reader.fit import _change_costs, content_grid, shake


def cadence_times(n, kind):
    """clip.cadence's times for n frames."""
    i = np.arange(n)
    t = {'24': i / 24.0, '30-5': (i + i // 4) / 30.0, '25-25': (i + i // 24) / 25.0}[kind]
    return np.round(t, 6)


def test_a_24fps_clip_is_its_own_grid():
    t = cadence_times(121, '24')
    grid, at = content_grid(t)
    assert np.array_equal(grid, t) and np.array_equal(at, np.arange(121))


def test_dropped_content_frames_get_their_place_on_the_grid():
    for kind, n, rate in (('30-5', 120, 30), ('25-25', 120, 25)):
        t = cadence_times(n, kind)
        grid, at = content_grid(t)
        assert np.array_equal(grid[at], t)
        assert np.allclose(np.diff(grid), 1 / rate, atol=2e-6)
        assert len(grid) > n


def test_times_on_no_grid_stay_the_videos_frames():
    t = np.array([0.0, .04, .1, .13, .2])
    grid, at = content_grid(t)
    assert np.array_equal(grid, t) and np.array_equal(at, np.arange(5))


def motion(t):
    """A keypoint and a joint moving at a steady speed: X (N, 1, 3), joints (N, 1), turn (N, 3)."""
    t = torch.as_tensor(t, dtype=torch.float64)
    X = torch.stack([t, 2 * t, 0 * t], -1)[:, None]
    return X, t[:, None] * 3.0, torch.zeros(len(t), 3, dtype=torch.float64)


def test_an_even_grid_costs_the_plain_differences():
    X, j, tu = motion(np.linspace(0, 1, 30) ** 2)
    acc, vel = _change_costs(X, j, tu, None)
    acc1, vel1 = _change_costs(X, j, tu, torch.ones(29, dtype=torch.float64))
    assert torch.isclose(acc, acc1) and torch.isclose(vel, vel1)
    assert torch.isclose(acc, ((X[2:] - 2 * X[1:-1] + X[:-2]) ** 2).sum())


def test_a_steady_move_has_no_acceleration_on_uneven_steps():
    t = cadence_times(40, '30-5')
    X, j, tu = motion(t)
    steps = torch.as_tensor(np.diff(t) / np.median(np.diff(t)))
    acc, _ = _change_costs(X, j, tu, steps)
    plain, _ = _change_costs(X, j, tu, None)
    assert acc.item() < 1e-12 < plain.item()          # counted in frames, the dropped frames looked like jumps


def test_shake_is_zero_for_a_smooth_move_and_grows_with_noise():
    t = np.linspace(0, 1, 50)
    X = np.stack([t ** 2, t, 0 * t], -1)[:, None]
    assert shake(X, 100.0) < 1e-9
    noisy = X + np.random.default_rng(0).normal(0, .01, X.shape)
    assert shake(noisy, 100.0) > .5
