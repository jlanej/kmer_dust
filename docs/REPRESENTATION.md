# What the sketch can and cannot see

Findings from a representation study run on `results/chr21` (24 assemblies,
105,007 bins, k=31, `scaled=200`). Every number below was computed directly
against this repository's own outputs and is reproducible from
`matrix/rows.parquet`, `annotate/annotations.parquet`, `sketch/*.sketch.parquet`
and re-sketched positional streams.

The short version: **two design choices in `sketch` and `matrix` are correct for
euchromatin and inverted for satellite**, and satellite is what this project
exists to study.

---

## 1. The representation discards position and multiplicity

Two lines do it.

`hashing.py:201`

```python
out_bin[count] = (i - k + 1) // bin_size
```

The k-mer's coordinate is computed and divided away on the same line.

`sketch.py:311-317`, justified in the module docstring — *"A bin is a set of
hashes, not a multiset ... so duplicates within a bin are collapsed before the
shard is written."*

Everything downstream is a faithful treatment of an object from which order and
copy number have already been removed. No amount of work after `sketch` restores
them.

**Position is free to keep.** `bin_index == position // bin_size`, so emitting
the position instead breaks nothing downstream, and both are 4-byte integers.
Calling `hashing.sketch_contig(..., bin_size=1)` already returns exact positions
today — the existing numba kernel needs no change.

**Multiplicity is not free, but it is cheap, and the cost is wildly asymmetric.**
Instances discarded by the set rule, measured across all 24 assemblies
(5,176,884 retained instances → 4,900,111 written, 5.3 % overall):

| feature | instances discarded |
| --- | --- |
| `asat_hor_active` | **73.6 %** |
| `hsat1a` | 51.8 % |
| `bsat` | 21.4 % |
| `hsat3` | 13.9 % |
| `rdna` | 4.3 % |
| `segdup` | 2.4 % |
| `gene` / `line` / `sine` / `ltr` | 0.2–0.5 % |

The rule looks harmless genome-wide and deletes three quarters of the active
alpha-satellite HOR array.

---

## 2. Satellite arrays are vocabulary-starved, not vocabulary-saturated

Distinct 31-mer hashes in a 200 kb window, median over haplotypes:

| class | haplotypes | distinct / 200 kb | already elsewhere in the **same** haplotype's array | **panel** adds |
| --- | --- | --- | --- | --- |
| `asat_hor_active` | 22 | **27** | 63.2 % | 23.3 % |
| `hsat1a` | 22 | 154 | 51.6 % | 41.7 % |
| `asat_mon` | 24 | 265 | 21.3 % | 77.6 % |
| `line` | 24 | 376 | 5.6 % | 94.1 % |
| `segdup` | 24 | 1,024 | 0.9 % | 98.9 % |

A live HOR array carries **27 distinct words per 200 kb**; a segmental
duplication carries 1,024 — a **38× vocabulary collapse**, with a clean monotone
gradient in between.

This single measurement explains most of the pipeline's behaviour in these
regions: why a single hash reaches multiplicity 148 in `asat_hor_active` and ~2
in `gene`; why cross-haplotype correspondence goes ambiguous at the centromere;
and why adding more haplotypes helps least exactly where the biology is hardest.

*Caution when reproducing:* do **not** use `gene` as the unique-sequence
control. It is a reference-only track, so a "panel" built from other haplotypes'
`gene` bins is empty by construction and the comparison returns 0.0 %. Use
`line` or `segdup`. This is the same asymmetric-track trap recorded in
[`RESULTS.md`](RESULTS.md).

---

## 3. Inside arrays, shareability rises with copy number — IDF is backwards there

For the 21 chr21 `asat_hor_active` arrays ≥ 600 kb, over all 420 ordered array
pairs: the probability that a hash present in array A is also present in array B,
as a function of that hash's bin-occupancy **within A**.

| bin-occupancy in A | n | P(shared with B) |
| --- | --- | --- |
| 1 | 30,860 | **0.179** |
| 2 | 9,580 | 0.229 |
| 3–4 | 6,500 | 0.351 |
| 5–8 | 4,560 | 0.446 |
| 9–16 | 3,480 | 0.546 |
| 17–32 | 1,780 | 0.647 |
| 33–64 | 620 | 0.708 |
| ≥65 | 3,420 | **0.915** |

Monotone, **5.1×**. A singleton hash inside a satellite array is ~80 % noise; a
high-copy hash is ~92 % reproducible in an independently assembled haplotype.

The same effect as landmark stability — median of the top-/bottom-N transferring
to another array:

| anchor rule | N=4 | N=8 | N=16 |
| --- | --- | --- | --- |
| top-N by multiplicity | **4/4** | **8/8** | 14/16 |
| bottom-N by hash value (min-hash) | 2/4 | 4/8 | 8/16 |

**Consequence.** `matrix.weighting: idf` computes `log(n_rows / df)`, which
upweights rare k-mers and downweights common ones. In euchromatin that is right —
rare is specific. Inside satellite arrays it systematically upweights the *least*
reproducible features and downweights the *most* reproducible ones, and it does
so after `sketch` has already deleted the multiplicity that would let you tell
the difference.

The README's argument for `select.max_sample_prevalence: 1.0` — that a k-mer in
every *sample* is not a k-mer in every *bin* — is correct and is not what this
contradicts. The problem is one level down: within a bin and within an array,
local copy number is a positive predictor of cross-haplotype reproducibility, and
the current pipeline cannot represent it, let alone weight by it.

---

## 4. What order buys, where it buys it

Two order-aware representations were prototyped on positional streams
(`(pos, hash)` in contig order, which the stored shards destroy by sorting on
`(bin_idx, hash)`).

**Paired landmarks** — hash each landmark together with the next few and the rank
gap between them, the audio-fingerprinting construction. Mean token multiplicity
within CHM13 chr21:

| feature | single → paired | gain |
| --- | --- | --- |
| `hsat3` | 18.6 → 1.7 | 10.7× |
| `hsat1a` | 225.7 → 31.1 | 7.3× |
| `bsat` | 23.8 → 4.1 | 5.7× |
| `asat_hor` | 38.3 → 8.5 | 4.5× |
| `gene` | 2.2 → 1.1 | 2.1× |

Cross-assembly correspondence (shared tokens vote for an offset; a vote is
collinear if it agrees with its local modal offset within ±10 kb over a 100 kb
window), CHM13 vs three haplotypes: **88.8 / 92.3 / 90.1 % collinear on 2.3–2.5×
more votes**, against 78.6 / 80.2 / 79.2 % for single hashes.

*Honest caveat:* with only ~27 distinct hashes per 200 kb of HOR array, pairing
has at most ~27² composite tokens available there, and the observed
multiplicity drop is consistent with that combinatorial ceiling rather than with
true disambiguation. The collinearity gain is carried mainly by the q-arm.

**Pair correlation** — histogram the distances between *same-hash* pairs, the
crystallographic Patterson construction. This recovers the repeat period with no
monomer library, no period prior and no alignment:

* CHM13's active array returns a fundamental of **1,868 bp** with a clean
  harmonic comb at 1×–5×; 1,868 / 171 = **10.9 monomers**, i.e. the 11-monomer
  HOR of chr21.
* The HOR discriminator is absolute: **G(171 bp) = 0.0** against
  **G(1,868 bp) = 163.0**. Monomers within an HOR unit are too divergent for a
  31-mer to survive; monomers between copies are not.
* Euchromatin control: peak strength **1.0** against 1,853.8 in satellite.
* **22 of 24 haplotypes independently return 1,868 bp.** `NA19909_hap2` returns
  1,700 bp — a clean **10-monomer** variant. `NA21102_hap2` is a heterogeneous
  array carrying several orders at once.

`rdna` fails (incoherent harmonics, and only 1.2× on pairing) because its ~45 kb
unit exceeds the 40 kb lag cap used here.

An interactive explorer for both is at
[`paired_landmark_explorer.html`](paired_landmark_explorer.html) — open it in a
browser; it is self-contained. The scripts that produced every number in this
document are in [`scripts/prototypes/`](../scripts/prototypes/).

---

## 5. What follows

Ordered by value over effort. None of this has been implemented.

1. **Emit the position.** One line in `hashing.py`, zero extra bytes, and it is
   the precondition for every order-aware method. Nothing downstream changes.
2. **Keep the multiset, or at least a per-(bin, hash) count.** Costs ~5 % of rows
   genome-wide; recovers 73.6 % of the discarded signal in live HOR arrays.
3. **Weight by local copy number inside repetitive bins**, rather than by global
   IDF everywhere. §3 gives the calibration curve directly. The simplest honest
   version is a per-bin ablation: IDF vs occupancy-aware, scored by cross-
   haplotype cluster agreement.
4. **Report the vocabulary-density track** (§2) alongside any clustering. A
   cluster drawn from 27 distinct words is not the same kind of object as one
   drawn from 1,024, and the current report does not distinguish them.
5. Only then consider a new representation. Seven cross-field designs were
   generated and hostile-reviewed in this study — imports from audio
   fingerprinting, time-series motif discovery, deduplication, computer-vision
   retrieval, crystallography, network blockmodelling and grammar compression.
   **All seven were killed or sent back for major revision**, and the recurring
   reason was that each was manufacturing features to route around the deficit in
   §2 rather than addressing it. Fix the sampling and weighting first.
