# Experiments

The paper's primary routes are indexed in [the main README](../README.md):
[Atom-DH](../routes/atom_dh/README.md), [Atom-CL](../routes/atom_cl/README.md), and
[Atom-WAM](../routes/atom_wam/README.md). Supporting experiments are organized
by technical role:

- [`atom_cl`](atom_cl/): staged ego-to-robot training.
- [`auxiliary_vla`](auxiliary_vla/): baseline, knowledge insulation, memory,
  and diverse context conditioning; not additional Atom-0 paper routes.

The auxiliary VLA recipes currently form a cumulative ladder. They must not be reported
as independent controlled ablations unless initialization, batch size, data,
training steps, and evaluation are matched.
