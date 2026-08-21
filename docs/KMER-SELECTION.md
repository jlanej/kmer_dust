# How k-mers are chosen

Everything downstream is a consequence of this stage, so it is worth being
precise about it. There are two independent sampling decisions and three
filters, and they do different jobs.

---

## 1. Sketching: which k-mers exist at all

For every 10 kb bin, `sketch` walks each canonical 31-mer and keeps it only if

```
splitmix64(canonical_2bit_code)  ≤  2⁶⁴ / scaled
```

**Canonical** means the smaller of the k-mer's 2-bit code and its reverse
complement's, so a bin sketches identically whichever strand it was assembled
on. `k` must be odd (a palindrome would otherwise be its own reverse complement
and the choice would be ambiguous) and ≤31 (so the code fits in 64 bits).

`splitmix64` is a **bijection** on the 64-bit integers. That is the whole trick:
because it is a bijection, thresholding its output keeps a uniform random
1-in-`scaled` sample of k-mer space — and, crucially, *the same* sample in every
bin of every assembly, decided with no coordination and no shared state. Two
bins that were never compared to each other still have intersectable sketches.
This is the FracMinHash / "scaled MinHash" construction that sourmash uses.

The alternative — a fixed list of interesting k-mers — would require deciding
that list up front, from a reference, which is exactly the reference dependence
the whole project is trying to avoid.

**What `scaled` buys you.** Expected sketch size per bin is `bin_size / scaled`:

| `scaled` | hashes per 10 kb bin | per 3.1 Gb haplotype |
| --- | --- | --- |
| 20 | ~500 | ~155 M |
| 200 | ~50 | ~15 M |
| 1000 | ~10 | ~3.1 M |

Observed on the acrocentric run at `scaled=200`: 61.6 M hashes over 1,303,160
bins, a median of **49 per bin** against a predicted 50.

A k-mer that occurs many times inside one bin is collapsed to a single entry.
Alpha-satellite bins contain the same 31-mer hundreds of times, and letting that
inflate the document frequency would make every HOR array look like a universal
k-mer.

---

## 2. Selection: which k-mers become matrix columns

`select` streams every shard's hashes, partitions them into `n_buckets` files by
the **high bits** of the hash — uniform output means balanced buckets, and a
given hash always lands in exactly one, so buckets can be counted independently
and concatenated — then counts, per hash, the number of distinct samples,
distinct assemblies, and bins. Peak memory is one bucket, so `n_buckets` is the
knob that keeps a 464-haplotype run inside a node.

Three filters then apply, in order.

### `min_bins` (default 2)

Drops k-mers seen in a single bin genome-wide. These carry no similarity
information by construction — a column with one non-zero cannot make two rows
alike — and they are the bulk of sequencing and assembly error.

### Sample prevalence (default: ≥10 %, ≤100 %)

Counted over **distinct samples, not haplotypes**. The two haplotypes of one
donor share most of their vocabulary, so counting haplotypes would let one
person's private variation look like population-level signal.

The **floor** is the filter that matters: a k-mer in one sample is private
variation or an assembly artefact, and keeping it makes that haplotype its own
cluster.

The **ceiling defaults to 1.0**, i.e. off, and this is deliberate and worth
stating because the opposite is the intuitive choice. A k-mer shared by every
*sample* is not a k-mer shared by every *bin*. An HSat2 31-mer is present in all
232 donors and in ~0.1 % of bins — it is one of the most discriminative columns
in the matrix, and a 0.95 ceiling would delete exactly the repeat-family
vocabulary the clustering exists to find. Bin-level ubiquity is a different
quantity and is handled by IDF weighting in `matrix`, which down-weights a
column smoothly by `log(n_rows / df)` rather than discarding it.

### `max_features`

A second FracMinHash, over the survivors: keep `h` iff
`splitmix64(h ⊕ seed) ≤ threshold`, with one global threshold chosen so the
expected yield is `max_features`. Because the decision depends on nothing but
the hash, it is order-independent, streamable, and identical on a re-run or
under a different bucket count — a property a "take the first N" or "take the
most frequent N" rule would not have.

> **This is the one that will bite you.** `max_features` is an *absolute* cap,
> but the eligible vocabulary grows with how much sequence is in the run. A
> config tuned on 61 k bins and reused on 1.3 M bins kept the same 200 k
> features — **11 %** of the eligible 1.77 M instead of most of them — and the
> matrix thinned from ~44 non-zeros per row to **5.4**. At that density most
> pairs of bins share no feature at all, cosine similarity is mostly exact
> zeros, and the embedding is dominated by ties. Every stage still reported
> success. `matrix` now warns when it happens; `max_features: 0` keeps
> everything that passed the filters.

---

## 3. What it looks like on real data

The acrocentric run — 33 assemblies, chr13/14/15/21/22, `k=31`, `scaled=200`:

```
61,639,507 (bin, k-mer) rows from 33 shards
 1,887,654 distinct k-mers
 1,765,879 after min_bins ≥ 2          (-6.5 %: singletons)
 1,765,879 after prevalence            (unchanged: ceiling is 1.0)
 1,765,879 selected                    (max_features: 0)
```

giving a 1,303,159 × 1,765,879 matrix with 61.5 M non-zeros — **47.2 per row**,
0.0027 % dense, 474 MiB.

For comparison, the 14-assembly chr21 run: 200,268 features over 61,048 bins at
43.9 non-zeros per row. **Non-zeros per row is the number to watch**; healthy
runs sit in the 40s, and it is what `matrix` warns about below 10.

---

## 4. Then what happens to them

The matrix is binary presence, IDF-weighted and L2 row-normalised. That
combination is not arbitrary: a truncated SVD of an IDF-weighted, L2-normalised
presence matrix is exactly **latent semantic analysis**, and "which k-mer
vocabulary does this bin use" is precisely the question LSA answers. It is also
why the SVD is *not* mean-centred — centring would destroy the sparsity that
makes a 1.3 M × 1.77 M factorisation take 19 seconds.

See [CONFIG.md](CONFIG.md) for every knob and [RESULTS.md](RESULTS.md) for what
came out.
