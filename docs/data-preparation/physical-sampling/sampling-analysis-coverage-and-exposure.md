# Realized coverage and acquisition exposure

## Realized coverage

The following are selected-triplet fractions across all accepted groups at
validation density. Distinct raw slices count source files used per group,
including different acquisitions at coincident geometry; they do not count
unique anatomical locations or measure actual decoder calls.

| Sampling | Beyond ±30 mm | Beyond ±40 mm | Mean distinct raw slices/type | 95th percentile placement error |
|---|---:|---:|---:|---:|
| Current | 40.59% | 22.14% | 28.53 | No shared physical target |
| Balanced Gaussian, sigma 20 mm | 15.03% | 0.05% | 21.61 | 2.01 mm |
| Balanced Gaussian, sigma 30 mm | 22.71% | 10.05% | 24.75 | 1.96 mm |
| Balanced Gaussian, sigma 40 mm | 29.85% | 13.59% | 26.11 | 1.92 mm |
| Balanced Gaussian, sigma 60 mm | 35.60% | 17.51% | 27.14 | 1.97 mm |
| Balanced Gaussian, sigma 80 mm | 37.68% | 19.24% | 27.46 | 1.95 mm |
| Balanced Gaussian, sigma 120 mm | 39.05% | 20.65% | 27.71 | 1.91 mm |

At training density, the current sampler puts 39.13% of selected centers beyond
±30 mm, compared with 23.25% at sigma 30 and 37.40% at sigma 80. Distinct source
slices average 15.75, 14.57 and 14.97 per type respectively. The same narrowing
appears in the training-partition-only strata, so it is not a consequence of the
small held-out subset.

The earlier suggestion that sigma 30 would provide broad arms was unsupported
by the actual baseline distribution. The non-overlap constraint and high
validation budget already force the current sampler to cover much of each stack.
The shared center can also differ from acquisition midpoints. Replacing those
rules together with a moderately concentrated Gaussian narrows exposure despite
its nonzero tails. Sigma 80 or 120 approaches current breadth but still does not
increase peripheral selection in this sample. At these widths the Gaussian is
fairly flat over a typical acquisition; width should reflect the intended
anatomical coverage, not an assumed benefit of the Gaussian name.

Deterministic quantiles also deserve care: sigma 20 places only 0.05% of selected
validation centers beyond ±40 mm, even though its continuous truncated Gaussian
has nonzero tail probability. A finite set of midpoint quantiles does not visit
every nonzero-probability region. At sigma 80, realized training and validation
tail exposure is much closer.

The current validation sampler supplies 5,659 triplets; all prototype variants
supply 6,248, an increase of 10.4% under the same nominal budgets. The current
sampler falls below the nominal budget in 231 of 595 groups because its triplets
cannot overlap. Candidate budgets therefore do not imply equal actual CNN work.
No runtime, VRAM or pixel-decoding cost was measured, and fewer distinct files
does not imply fewer decoder calls in the present preparation implementation.

At sigma 80 the largest assignment error is 3.17 mm in validation and 5.58 mm
over the training draws. Three of 99,968 training triplets exceed 5 mm; none
does in validation. Sigma 20 has a 12.19-mm worst validation displacement when
distinct middle planes must be assigned to tightly concentrated targets. A
fixed-center uniqueness constraint can therefore distort a narrow distribution.

## Acquisition exposure and eligibility

For depth t covered by m(t) acquisitions, the intended source share is the
Gaussian probability at t divided equally among those m(t) sources. Integrating
over depth gives each acquisition's entitlement. This depends on its available
coverage, not its number of slices. The numerical reference uses 2,048 fixed
Gaussian quantiles per group.

At sigma 30, mean absolute deviations from these intended acquisition shares,
across the 197 acquisitions in repeated groups, are:

| Method | Validation deviation | Training deviation over 32 draws |
|---|---:|---:|
| Current | 3.30 percentage points | 4.53 percentage points |
| Nearest across acquisitions | 23.91 percentage points | 18.90 percentage points |
| Choose supplier first | 1.10 percentage points | 2.95 percentage points |

The current method has a different target distribution, so its row is contextual
rather than an isolated source-selection comparison. The nearest and balanced
rows use identical physical targets. Nearest selection can favor denser grids
and systematically resolve coincident geometry to the same source. Supplier
selection corrects both effects in this cohort. Dense acquisitions, defined
here by median spacing below 2 mm, average 10.16 percentage points above their
intended share with nearest validation selection, versus 0.53 points below it
with supplier-first selection.

This improvement costs little placement accuracy: mean validation error changes
from 0.906 mm to 0.920 mm at sigma 30. The balanced method's small remaining
deviation comes from finite target and source counts; its training result also
contains Monte Carlo variation. This is not proof of strict invariance under
every possible duplicate acquisition or missing-data pattern.

The broad sigma-80 variant retains this behavior: mean source-share deviation
is 1.06 percentage points in validation and 2.76 in the training draws, measured
against its own sigma-80 coverage entitlements. Dense acquisitions average
0.22 points below their validation entitlement and 0.05 points above their
training entitlement.

Fixed absolute eligibility thresholds are unsuitable as the main source rule.
At sigma 30, a 1-mm maximum perpendicular distance to the nearest middle plane
leaves an average 41.93% of target probability without any eligible acquisition.
The corresponding figures are 3.78% at 2 mm and 0.052% at 3 mm. Such cutoffs can
also favor dense acquisitions whenever they remain eligible and coarse ones
do not. The prototype instead uses coverage and lets each selected source
represent the target with its locally available slices. Placement error is
measured after assignment.

The existing sample has regular within-series spacing. This does not establish
what to do with abnormal gaps or missing slices outside the cohort. The
implementation plan must define valid continuous runs and failure handling;
the above eligibility result is not permission to bridge arbitrarily large
gaps.
