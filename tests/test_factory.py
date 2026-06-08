"""Tests for the PipedreamBackend coupled-subset support and couple_from_inp.

The PipedreamBackend test needs pipedream; the couple_from_inp test additionally
needs ANUGA. Both skip cleanly when those aren't installed.
"""
import numpy as np
import pytest

from anuga_drainage import read_inp, inp_to_pipedream, PipedreamBackend

_INP = """\
[JUNCTIONS]
;;Name  Elev  MaxD  InitD  Sur  Apond
J1      0.0   1.0   0      0    0
J2      0.0   1.0   0      0    0

[OUTFALLS]
O1      0.0   FREE         NO

[CONDUITS]
C1      J1    J2    8.0    0.013   0   0   0   0
C2      J2    O1    5.0    0.013   0   0   0   0

[XSECTIONS]
C1      CIRCULAR   0.5   0   0   0   1
C2      CIRCULAR   0.5   0   0   0   1

[COORDINATES]
J1      15.0   5.0
J2       8.0   5.0
O1       3.0   5.0
"""


@pytest.fixture
def inp_path(tmp_path):
    p = tmp_path / "net.inp"
    p.write_text(_INP)
    return str(p)


def test_pipedream_backend_couples_subset_and_holds_bc(inp_path):
    SuperLink = pytest.importorskip("pipedream_solver.hydraulics").SuperLink
    inp = read_inp(inp_path)
    sj, sl = inp_to_pipedream(inp)
    s = SuperLink(sl, sj, internal_links=4)
    njunc = len(inp.junctions)                     # 2 junctions, 1 outfall
    be = PipedreamBackend(s, coupled_indices=range(njunc), H_bc=s._z_inv_j.copy())

    assert len(be.get_heads()) == njunc            # only the junctions are coupled
    be.step(np.array([0.05, 0.0]), dt=1.0)         # inject at J1 only
    assert float(s.H_j[-1]) == pytest.approx(float(s._z_inv_j[-1]))  # outfall held at its invert
    assert s.H_j[0] > s._z_inv_j[0]                # J1 filled
    assert list(be.coupling_inflow_volumes()) == pytest.approx([0.05, 0.0])


def test_pipedream_backend_default_is_all_coupled_no_bc(inp_path):
    # Backwards compatibility: no coupled_indices/H_bc -> couple every
    # superjunction and never pass H_bc (matches the hand-built examples).
    SuperLink = pytest.importorskip("pipedream_solver.hydraulics").SuperLink
    sj, sl = inp_to_pipedream(read_inp(inp_path))
    s = SuperLink(sl, sj, internal_links=4)
    be = PipedreamBackend(s)
    assert len(be.get_heads()) == len(s.H_j)
    assert be.H_bc is None


def test_couple_from_inp_pipedream(inp_path):
    anuga = pytest.importorskip("anuga")
    pytest.importorskip("pipedream_solver.hydraulics")
    from anuga_drainage import couple_from_inp

    domain = anuga.rectangular_cross_domain(20, 10, len1=20.0, len2=10.0)
    domain.set_quantity("elevation", 0.0)
    domain.set_quantity("stage", 0.3)
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})

    c = couple_from_inp(domain, inp_path, backend="pipedream",
                        manhole_area=0.5, time_average=1.0, internal_links=4)

    # junctions are coupled, the outfall is not
    assert list(c.inlets) == ["J1", "J2"]
    assert len(c.coupler.inlets) == 2
    assert c.backend.H_bc is not None              # outfall is a free-drainage boundary

    last = None
    for t in domain.evolve(yieldstep=1.0, finaltime=2.0):
        last = c.coupler.step(1.0)
    assert last.Q_in.shape == (2,)
    assert np.isfinite(last.Q_in).all()
