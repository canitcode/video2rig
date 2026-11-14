"""Bone names: which bones the tool takes for knees and thighs (creature.py)."""
import pytest

from video2rig.creature import RANGES, THIGH, is_knee

THIGHS = ['LeftUpLeg', 'mixamorig:RightUpLeg', 'thigh.L', 'Thigh_R', 'upper_leg.L', 'UpperLeg', 'leftUpperLeg',
          'J_Bip_L_UpperLeg', 'upperleg_r', 'upper-leg.L']
KNEES = ['LeftLeg', 'mixamorig:RightLeg', 'shin.L', 'Calf_R', 'lower_leg.L', 'LowerLeg', 'leftLowerLeg', 'knee.R',
         'Leg_L', 'J_Bip_R_LowerLeg']


@pytest.mark.parametrize('name', THIGHS)
def test_a_thigh_is_no_knee(name):
    assert THIGH.search(name)
    assert not is_knee(name)


@pytest.mark.parametrize('name', KNEES)
def test_a_knee_is_a_knee(name):
    assert is_knee(name)
    assert not THIGH.search(name)


@pytest.mark.parametrize('name', THIGHS)
def test_a_thigh_gets_the_thighs_range(name):
    assert next(d for rx, d in RANGES if rx.search(name)) == (130, 45, 60)
