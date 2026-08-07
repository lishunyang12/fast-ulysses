# Provenance notes

Two commits on this branch have messages that do not fully describe their contents. Rather than
rewrite the history, the gaps are recorded here.

## d609c0d "Converge on the copy engines, and put the addressing in a host-testable plan"

The message describes two changes: deleting the SM/TMA transports and the autotune (mine), and
adding the host-side plan (a subagent's, reviewed by a second). The commit also contains, from
the same subagent run and not mentioned in the message:

- **Strided subgroup support.** `nvshmem_team_split_strided` gets the real stride derived from
  the PE list instead of a hardcoded `1`, the contiguous-only check is gone, and `comm.py` no
  longer raises `NotImplementedError` for a subgroup. This is what makes tp2 x sp4 work --
  `{0,2,4,6}` and `{1,3,5,7}`, verified at 8 ranks.
- **Four adversarial workers**, ported from the sibling implementation:
  `a2a_window_race.py`, `a2a_cudagraph.py`, `a2a_overlapping_barriers.py`,
  `a2a_ce_flag_ordering.py`, with their registrations in `tests/test_multigpu.py`.

How it happened: the subagents' worktree isolation did not take, they wrote into the main
checkout, and a `git add -A` swept the lot into a commit whose message had been written for the
deletion alone. The message was corrected once afterwards to cover the plan layer; the subgroup
work and the workers were still missing from it, and are recorded here instead.

## Attribution generally

Where a commit message says a change was made by a subagent, it was; where it does not, it was
made directly. Reviews described in messages were performed by a separate agent from the one
that wrote the code, and their findings are summarised in the message they belong to.
