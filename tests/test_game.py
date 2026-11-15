"""The reader fit's helpers for a game (reader/fit.py): standing feet, travel from them, the loop's window, settings."""
import numpy as np
import pytest

from video2rig.reader.fit import SETTINGS, find_loop, settings_for, standing, travel

UP = np.array([0.0, 0.0, 1.0])
HEIGHT = 2.0


def treadmill(n=48, rate=24.0, speed=.25 * HEIGHT):
    """Two feet walking on the spot, as a video model draws it, one step a second: each foot stands for two thirds of
    it, sliding back at `speed` (slow enough to count as standing), then swings forward through the air twice as
    fast. (feet (n, 2, 3), times (n,))."""
    t = np.arange(n) / rate
    feet = np.zeros((n, 2, 3))
    stand, reach = 2 / 3, speed * 2 / 3
    for f, phase in enumerate((0.0, .5)):
        c = (t + phase) % 1.0
        on = c < stand
        swing = (c - stand) / (1 - stand)                             # 0..1 through the air
        feet[:, f, 0] = np.where(on, reach / 2 - speed * c, -reach / 2 + reach * swing)
        feet[:, f, 2] = np.where(on, 0.0, .1 * HEIGHT * np.sin(np.pi * swing))
    return feet, t


def test_a_foot_on_the_floor_that_barely_moves_stands():
    t = np.arange(30) / 24.0
    feet = np.zeros((30, 1, 3))
    feet[:, 0, 0] = .001 * np.arange(30)
    feet[15:, 0, 2] = .2 * HEIGHT                                     # lifted from frame 15
    on = standing(feet, t, UP, np.zeros(1), HEIGHT)
    assert on[:14, 0].all() and not on[16:, 0].any()


def test_a_walk_on_the_spot_travels_by_what_its_standing_feet_slide_back():
    feet, t = treadmill()
    on = standing(feet, t, UP, np.zeros(2), HEIGHT)
    moved = travel(feet, on, UP)
    assert moved[-1, 0] > .2 * HEIGHT * (t[-1] - t[0])                # forward, about the slide's speed
    assert abs(moved[-1, 1]) < 1e-9 and abs(moved[-1, 2]) < 1e-9      # along the floor only
    world = feet + moved[:, None]
    both = on[1:] * on[:-1]
    slid = np.linalg.norm(np.diff(world[..., :2], axis=0), axis=-1)[both > 0]
    assert slid.max() < 1e-9                                          # a standing foot stays put


def test_a_repeating_motion_loops_over_whole_periods():
    t = np.arange(120) / 24.0
    X = np.stack([np.sin(2 * np.pi * .8 * t), np.cos(2 * np.pi * 1.6 * t), .1 * np.sin(2 * np.pi * .8 * t)], -1)
    X = np.stack([X, 2 * X], 1) + np.array([.3, 0, 0]) * t[:, None, None]        # two points, travelling along x
    i, j = find_loop(X, t, UP, shortest=.5)
    period = (t[j] - t[i]) / 1.25
    assert abs(period - round(period)) < .05 and round(period) >= 1


def test_reader_fit_settings_come_from_creature_json_and_set_wins():
    s = settings_for({'reader_fit': {'w_contact': 1.0, 'loop': True}}, {'w_contact': 2.0})
    assert s['w_contact'] == 2.0 and s['loop'] is True and s['w_acc'] == SETTINGS['w_acc']
    with pytest.raises(SystemExit):
        settings_for({'reader_fit': {'w_contactt': 1.0}})


def test_the_game_helpers_are_off_by_default():
    assert SETTINGS['w_contact'] == 0 and SETTINGS['loop'] is False and SETTINGS['travel'] is False


def test_a_closed_loop_ends_where_it_starts_and_keeps_its_travel():
    from video2rig.reader.fit import close_loop
    t = np.arange(81)[:, None] / 24.0
    P = np.zeros((81, 6))
    P[:, :3] = np.array([.5, 0, .02]) * t                                      # travels along x, rises a little
    P[:, 3:] = np.sin(2 * np.pi * .6 * t) + .05 * t                            # angles: about a period, drifting
    i, j = 30, 70
    L, faded = close_loop(P, i, j, UP)
    assert len(L) == j - i + 1 and np.allclose(L[0], P[i])                      # it starts as the clip there
    assert np.allclose(L[-1, 3:], L[0, 3:]) and faded > 0                       # and ends where it starts
    assert np.isclose(L[-1, 0] - L[0, 0], P[j, 0] - P[i, 0]) and np.isclose(L[-1, 2], L[0, 2])  # travel stays
    assert np.allclose(L[:j - i - 10], P[i:j - 10])                             # untouched before the fade
    into = L[-1] - L[-2]                                                        # it arrives moving as the clip
    gap = np.abs(P[j, 3:] - P[i, 3:]).max()
    assert np.abs(into[3:] - (P[i, 3:] - P[i - 1, 3:])).max() < .05 * gap
