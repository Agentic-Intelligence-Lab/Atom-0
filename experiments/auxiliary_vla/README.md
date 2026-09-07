# Auxiliary VLA experiment ladder

These retained knowledge-insulation, memory and context-conditioning recipes
are auxiliary experiments, not Atom-DH, Atom-CL or Atom-WAM paper results.

| Recipe | Training config | Enabled components |
| --- | --- | --- |
| [Baseline](baseline) | `pi05_no_ki_libero` | None |
| [KI](ki) | `pi05_ki_libero` | KI |
| [MEM](mem) | `pi05_mem_libero` | KI + short-term MEM |
| [DCC](dcc) | `pi05_dcc_libero` | KI + short-term MEM + DCC |

Long-term MEM is trained separately with `pi0_hl_rmbench`.

These are the configurations currently present in
`src/openpi/training/config.py`. The recipes are cumulative, not four matched
independent methods.
