# Remote training host for campaigns

**Built on 2026-10-12 on branch `fastmri-pretraining`, uncommitted; no campaign has run on it
yet.** A campaign with a `training_host` block runs its calibration, trainings and comparisons on
a rented machine while the agents, the queue, the review and every record stay local; a campaign
without it trains locally as before. How it works:
[training host](../../../autoresearch/runtime.md#training-host).

## Checked

- The launcher, from inside the local sandbox, against the rented RTX 5090 with a trivial command:
  the environment built from the lock, the campaign root copied, the command run with the host's
  GPU visible and offline model loading set, the log streamed and the output folder brought back.
- `validate` accepts the first hosted campaign,
  [fastMRI joint training](../../../autoresearch/campaigns/fastmri-joint/INDEX.md).

## Left

- A first campaign through it: calibration, one training and one comparison have not run on the
  host. The comparison loads both runs' checkpoints, which is why it runs there.
- Cancelling a run relies on the heartbeat guard; it has not been exercised.
- A checkpoint comes back only by hand, when a run is submitted or merged.
- The first hosted reference was trained before the paths were mirrored, so its recorded paths
  are declared as exceptions in that campaign.
