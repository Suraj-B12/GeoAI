# Bengaluru 76-Photo Re-classification Comparison

Compared at: 2026-05-07T17:57:51.523483Z
Rows compared: 76

## Status distribution

| Status | Baseline | Fine-tuned + new prompts |
|---|---|---|
| classified | 29 | 32 |
| expert_review | 47 | 45 |

## Status transitions (baseline → fine-tuned)

| Transition | Count |
|---|---|
| `expert_review -> expert_review` | 40 |
| `classified -> classified` | 24 |
| `expert_review -> classified` | 7 |
| `classified -> expert_review` | 5 |

## Stage 1 label transitions

| Transition | Count |
|---|---|
| `Distress -> Distress` | 43 |
| `Normal -> Normal` | 26 |
| `Normal -> Distress` | 5 |
| `Distress -> Normal` | 2 |

## Expert-review flag flips

- **Resolved** (was flagged, now isn't): 7
- **Newly flagged** (wasn't flagged, now is): 6
- **Unchanged** (same flag value either way): 63

## Confidence deltas (avg, only over rows with both runs producing a value)

- Stage 1 confidence delta avg: `0.014`
- Stage 2 confidence delta avg: `0.011`

## Type changes (22 rows)

| ID | Address | Baseline types | Fine-tuned types |
|---|---|---|---|
| `13e9fba4` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Raveling, Weathering/Oxidation | Pothole (D40) |
| `163ff37f` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Pothole (D40), Raveling | Pothole (D40) |
| `1c0ce951` | WHR8+95V, Basavanagudi, Bengaluru, Karna | _(none)_ | Raveling |
| `46341bbd` | 336, 25th Cross Rd, Siddanna Layout, Ban | Block Crack (D43), Raveling | Pothole (D40) |
| `53a627b7` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | _(none)_ | Raveling |
| `557305d8` | 771, Lal Bahadur Shastri Nagar, Bengalur | Longitudinal Crack (D00), Raveling | Raveling |
| `739f9e77` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Polishing, Raveling | Pothole (D40) |
| `7e5c8914` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Longitudinal Crack (D00), Raveling | Pothole (D40) |
| `86502906` | 70, Kanakapura Main Rd, Navaratan Garden | _(none)_ | Pothole (D40) |
| `86803b76` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Longitudinal Crack (D00), Pothole (D40) | Pothole (D40) |
| `894b1f6c` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Raveling, Weathering/Oxidation | Pothole (D40) |
| `9891fe4c` | WHR8+95V, Basavanagudi, Bengaluru, Karna | Pothole (D40), Raveling | Pothole (D40) |
| `a6bc5616` | 336, 25th Cross Rd, Siddanna Layout, Ban | _(none)_ | Raveling |
| `b8425ccd` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Pothole (D40) | _(none)_ |
| `be596cf5` | WHR8+95V, Basavanagudi, Bengaluru, Karna | Polishing, Raveling | Raveling |
| `c015de56` | Kempegowda International Airport Bengalu | _(none)_ | Longitudinal Crack (D00) |
| `c1d158df` | 70, Kanakapura Main Rd, Navaratan Garden | Edge Crack (D08) | _(none)_ |
| `cf151564` | WHR8+95V, Basavanagudi, Bengaluru, Karna | Raveling, Weathering/Oxidation | Raveling |
| `d02a877a` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Longitudinal Crack (D00), Raveling | Pothole (D40) |
| `dadebfbf` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Raveling, Weathering/Oxidation | Pothole (D40) |
| `e6977719` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Block Crack (D43), Longitudinal Crack (D00) | Pothole (D40) |
| `eedae0bb` | 23, 7th Cross Rd, CK Nagar, NR Colony, B | Raveling, Weathering/Oxidation | Pothole (D40) |

## Severity changes (15 rows)

| ID | Baseline | Fine-tuned |
|---|---|---|
| `13e9fba4` | Medium | High |
| `1c0ce951` | None | Low |
| `46341bbd` | Medium | High |
| `53a627b7` | None | Low |
| `739f9e77` | Low | High |
| `7e5c8914` | Medium | High |
| `86502906` | None | High |
| `894b1f6c` | Medium | High |
| `a6bc5616` | None | Low |
| `b8425ccd` | High | None |
| `c015de56` | None | Low |
| `c1d158df` | Low | None |
| `dadebfbf` | Medium | High |
| `e6977719` | Medium | High |
| `eedae0bb` | Medium | High |
