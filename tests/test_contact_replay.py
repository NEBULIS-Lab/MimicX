"""Numerical contact auditing without a simulation dependency."""
import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module("mimicx.visualization.contact_replay")


def ballistic():
    dt = .01
    velocity = np.zeros((4, 3)); velocity[:, 2] = np.arange(4) * -9.81 * dt
    position = np.zeros((4, 3)); position[:, 2] = 2 + np.r_[0, np.cumsum(velocity[1:, 2] * dt)]
    return dict(time=np.arange(4) * dt, position=position, velocity=velocity,
                quaternion_wxyz=np.tile([1., 0, 0, 0], (4, 1)),
                constraint_force=np.zeros((3, 3)), contact_count=np.zeros(3, int), contacts=[])


def test_ballistic_audit_accepts_euler_dynamics():
    result = api().audit_replay(ballistic(), mass=.1)
    assert result["unexplained_velocity_change_substeps"] == 0
    assert result["force_balance_residual_max_m_s"] < 1e-10


@pytest.mark.parametrize("tamper", ["velocity", "position", "clock", "quaternion", "force_without_contacts", "nan"])
def test_audit_rejects_teleports_floating_bounces_and_bad_states(tamper):
    out = ballistic()
    if tamper == "velocity":
        out["velocity"][2, 2] += 1
    elif tamper == "position":
        out["position"][2, 0] += .1
    elif tamper == "clock":
        out["time"][2] = out["time"][1]
    elif tamper == "quaternion":
        out["quaternion_wxyz"][1] = 0
    elif tamper == "force_without_contacts":
        out["constraint_force"][0, 2] = 1
    else:
        out["position"][0, 0] = np.nan
    with pytest.raises(ValueError):
        api().audit_replay(out, mass=.1)


def test_contact_groups_sum_normal_and_world_impulses():
    contacts = [dict(time=t, geom="floor", normal_force_n=f, force_world_n=[0, 0, f])
                for t, f in [(0., 10), (.01, 20), (.2, 30)]]
    groups = api().group_contacts(contacts, dt=.01)
    assert len(groups) == 2
    assert groups[0]["normal_impulse_ns"] == pytest.approx(.3)
    np.testing.assert_allclose(groups[0]["impulse_world_ns"], [0, 0, .3])


def test_ballistic_initial_velocity_is_configurable_and_not_a_trajectory_override():
    initial = api().ballistic_initial([0, 0, 1], [1, 0, 2], flight_time=.5)
    np.testing.assert_allclose(initial, [0, 0, 1, 2, 0, 4.4525])
    with pytest.raises(ValueError):
        api().ballistic_initial([0, 0, 1], [1, 0, 2], flight_time=0)
