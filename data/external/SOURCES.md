# Public external validation data

The internal laboratory archive is confidential and is not distributed. Everything
in this directory is an extract from the public external sources listed below.
The extracts are the adapter inputs used in this study. The original full-rate
records must be downloaded from their providers under the current provider terms,
and the original datasets should be cited separately from this code.

## 1. Karlsruhe fine sand

Provider: Torsten Wichtmann and Theodoros Triantafyllidis, Karlsruhe fine sand
database, accompanying their 2016 Acta Geotechnica publications.

- [Isotropically consolidated undrained cyclic tests, TCUI](https://www.torsten-wichtmann.de/KFSDB-TCUI.htm)
- [Anisotropically consolidated undrained cyclic tests, TCUA](https://www.torsten-wichtmann.de/KFSDB-TCUA.htm)
- [Drained monotonic triaxial tests, TMD](https://www.torsten-wichtmann.de/KFSDB-TMD.htm)

Open each series page and select **Data of all tests of this series**, or the
individual **Test data** links. Retain the `*-test-data-vs-time.dat` histories and
the initial-state/loading metadata. TMD is needed by the fixed stiffness mapping
used for KFS and LEAP; it is not an additional cyclic validation cohort.

Included adapter files:

```text
kfs/kfs_meta.csv
kfs/kfs_tmd_summary.csv
kfs/kfs_events.csv
kfs/kfs_cycles_part1.txt
kfs/kfs_cycles_part2.txt
```

The cycle files contain two lines per test: a header with the test name, row count
and integer checksum, followed by semicolon-separated rows of cycle number, peak
and end pressure ratio, and maximum/minimum axial strain. Non-cycle quantities
are stored multiplied by 1000. The reader verifies counts and checksums.
The TCUA10 duplicate and negative-initial-stress-ratio extension tests are excluded
by the fixed adapter. At the ten-cycle landmark the retained counts are TCUI
20 tests/20 events and TCUA 14 tests/1 event.

## 2. LEAP-2017 GWU laboratory tests

El Ghoraiby, Park and Manzari (2018), **LEAP-2017 GWU Laboratory Tests**,
DesignSafe project **PRJ-1783**, [dataset DOI](https://doi.org/10.17603/DS2210X).

Open the published project in the [DesignSafe Data Depot](https://www.designsafe-ci.org/data/browser/),
then download **Cyclic Stress-Controlled Triaxial Tests** and its test metadata.
Use the Ottawa F65 element tests; the centrifuge experiments and strain-controlled
LEAP campaigns are different tasks. DesignSafe may request a free account for
the download workflow.

Included adapter files:

```text
leap2017_gwu/leap2017_ctx_cycles.txt
leap2017_gwu/leap2017_nocyc_published.csv
```

The cycle file has a header/checksum and one data line per test. The rows contain
cycle number, peak pressure ratio and maximum/minimum axial strain, with the
non-cycle columns multiplied by 1000. Header fields retain consolidation stress
and measured cyclic amplitude. The eligible cohort has 18 tests and 17 events.

## 3. Adapazari cyclic laboratory collection

Sancio, Bray, Riemer and Sahin, **Cyclic Laboratory Test Data of Soil Samples from
Adapazari, Turkiye**, DesignSafe project **PRJ-4583**,
[dataset DOI](https://doi.org/10.17603/ds2-24hv-kf48).

Download the published cyclic-test spreadsheets, including their initial-state,
grain-size and loading metadata. The NGL infrastructure also links the collection:
[Next Generation Liquefaction](https://nextgenerationliquefaction.org/).
The eligible cohort used here contains 54 tests and 46 events; 53 tests have a
future pressure query point after cycle 10.

## 4. Christchurch cyclic laboratory collection

Christchurch laboratory collection associated with Beyzaei's research, available
through NGL and DesignSafe project **PRJ-6435**,
[dataset DOI](https://doi.org/10.17603/ds2-cpfr-c782).

Download the cyclic laboratory records and specimen metadata from the published
collection. Use the laboratory element tests rather than field case-history tables.
The eligible cohort contains 21 tests and 15 events; 20 tests have a future pressure
query point after cycle 10.

The two NGL collections share the following adapter input:

```text
ngl/ngl_cyclic_payload.txt
```

`#ADA` and `#CHC` delimit the collections. Each test has a pipe-separated metadata
line followed by semicolon-separated integer rows: cycle, corrected peak pressure
ratio multiplied by 1000, and double-amplitude axial strain multiplied by 10.
The `nr` and `cs` fields verify row count and checksum. Original-resolution event
cycles are retained in header fields, rather than inferred from the sparse tail.

## Processing conventions

Cycle boundaries use upward crossings of cyclic deviator stress relative to its
initial static level, with hysteresis at ±0.3 of the cyclic amplitude. KFS retains
cycles 1–30 at full resolution and a sparse later tail; NGL retains cycles 1–12
and a sparse tail. The final cycle is retained. Event labels are calculated at
the original cycle resolution before tail subsampling.

KFS and LEAP use pressure rise normalized by initial confinement. NGL uses the
cycle maximum of `[Delta_u - (Delta_q/3 + Delta_sigma_cell)] / sigma_3c_effective`.
The internal pressure target retains its cyclic pressure component, so the
three targets are reported by source and not pooled. Event and quality-filter details are in
`external_validation/extval/kfs.py`, `leap.py`, `ngl.py` and `cohort.py`.

Notebook 4.1 reads the included extracts and verifies their checksums. The adapters
expect these processed files; conversion from the providers' original spreadsheets
is a separate preparation step. Keep downloaded full-rate records outside Git,
together with the provider's metadata and terms.
