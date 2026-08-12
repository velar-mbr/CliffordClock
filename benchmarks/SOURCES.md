# Dataset benchmark sources

**TLDR:** this document is the citation, checksum,
license, and access-attempt log for every external source used in
`benchmarks/RESULTS.md`/`benchmarks/MAPPING.md`: URLs/DOIs, retrieval
dates, SHA-256 checksums (independently verified where the source
publishes its own), redistribution terms, and (for the one source this
project could not obtain) the full access-attempt log. See
`benchmarks/RESULTS.md`'s executive summary for what these sources were
used to conclude.

Fetch authorized by the project owner, 2026-08-10, for the first two
sources below, plus four additional sources authorized the same day as
follow-ups: the Supplemental Material of the published version of the
JILA paper (PRL 133, 023401, section 3), NPL Rydberg electrometry
(arXiv:1706.01944, section 4), the USTC Sr1 evaluation (Metrologia 63,
025002 (2026), section 5), and Li J et al 2024's USTC Sr1 previous
evaluation (Metrologia 61, 015006, section 6: **access blocked, no
content obtained**, see that section). No other external source was
fetched for this benchmark pass. Retrieval date for every
successfully-fetched file below: **2026-08-10**.

## 1. arXiv:2403.10664: JILA 1D Sr-87 optical lattice clock

**A. Aeppli, K. Kim, W. Warfield, M.S. Safronova, J. Ye, "A clock with
8x10^-19 systematic uncertainty", arXiv:2403.10664v2 [physics.atom-ph]
(2024-06-08).**

- Abstract page: <https://arxiv.org/abs/2403.10664>
- PDF: <https://arxiv.org/pdf/2403.10664v2>
- TeX source bundle: <https://arxiv.org/e-print/2403.10664v2>
- Version fetched: **v2** (2024-06-08, "Dated: June 11, 2024" per the
  paper's own dateline). v1 (2024-03-15) was not used.

**Ancillary/supplementary data files: none.** The arXiv abstract/format
pages for this submission have no "Ancillary files" section; the e-print
bundle contains only the TeX sources, bibliography, and figure PDFs/PNGs
needed to typeset the paper (`main.tex`, `supplementary.tex`, `main.bbl`,
`supplementary.bbl`, `bibliography.bib`, and per-figure PDF/PNG files):
no machine-readable data table. This was confirmed by listing the e-print
tarball's contents directly, not inferred.

**Files fetched (checksums of the actual bytes retrieved 2026-08-10):**

| File | SHA-256 | Size | Committed to repo? |
|---|---|---|---|
| `arxiv_2403.10664v2.pdf` | `f222fe4de23aa4a5f7f3daa13c428767299a217a2849f2f2f9472f56dfef4871` | 11,101,549 B | **No** (see License below) |
| `arxiv_2403.10664v2_source.tar.gz` | `f8c88d5046b086dac5bf588f03f52c094ef2eac453dd26399668524f8f684d95` | 10,788,809 B | **No** (see License below) |

**License / redistribution.** The submission's rights page
(<https://arxiv.org/abs/2403.10664>) links
`http://arxiv.org/licenses/nonexclusive-distrib/1.0/`: arXiv's default
"perpetual, non-exclusive license to distribute" granted *to arXiv*, not a
copyleft/CC license granting third-party redistribution rights. Copyright
remains with the authors. Per this benchmark's binding instruction ("If a license
forbids redistribution, keep a fetch script instead of committing the raw
file"), **the PDF and TeX source are not committed to this repository.**
`benchmarks/fetch_data.py` re-downloads and checksum-verifies both files
on demand. The small set of numeric values transcribed from the paper's
Table I into `benchmarks/fixtures/jila_2403_10664_table1.csv` (facts/data,
not the paper's copyrighted expression) is used under the same normal
scholarly-citation practice already established in this repo (e.g.
`cliffordclock.ensemble.species.SR87`'s literature-cited `Δα` value); every
transcribed number carries an exact table/section/page citation, see
`benchmarks/MAPPING.md`.

**What the paper actually contains (relevant to this benchmark):** Table I, "Fractional
frequency shifts and uncertainties for the JILA 1D Sr optical lattice
clock": 9 systematic-shift line items + total (BBR, lattice light,
second-order Zeeman, density, first-order Zeeman, background gas, DC
Stark, tunneling, minor shifts). No ancillary machine-readable data of any
kind. See `benchmarks/MAPPING.md` for the full row-by-row scope
classification and citations, and `benchmarks/RESULTS.md` for what
comparison is (and is not) possible against this table.

**WP20 addendum (2026-08-11): extended use, no new fetch.** WP20's BBR
arithmetic-reproduction benchmark case
(`benchmarks/run_bbr_jila_arithmetic_reproduction.py`,
`benchmarks/RESULTS.md`'s "Arithmetic-reproduction case: JILA BBR row"
section) uses two more numbers from this *same* already-fetched,
checksummed source: Table I's "BBR" row (`-48417.2(73)×10⁻¹⁹`, already
transcribed into `benchmarks/fixtures/jila_2403_10664_table1.csv` for
WP10, above: restated as `benchmarks/loaders.JILA_BBR_PUBLISHED_SHIFT`)
and the main-text in-vacuum-RTD operating-temperature statement
(`T = 20.132(4) °C = 293.282(4) K`, `benchmarks/loaders.JILA_BBR_TEMPERATURE_K`).
Per the binding instruction to extend this logged source rather than
duplicate it, this WP20 pass did not re-fetch or re-read the PDF itself;
the temperature value was independently verified against Aeppli et
al.'s primary text in this project's internal review sweep (2026-08-11,
prior session). No new checksum, license question, or access attempt
applies: both values fall entirely within the already-fetched/logged
`arxiv_2403.10664v2.pdf` (sha256 above) and the already-established
transcription-under-scholarly-citation practice this section documents
for Table I.

## 2. data.nist.gov DOI 10.18434/M32206

**T. Nakamura et al., "Data for 'Coherent Optical Clock Down-Conversion for
Microwave Frequencies with 10-18 Instability'", data.nist.gov, DOI:
10.18434/M32206 (published 2020-04-09), referencing arXiv:2003.02923.**

- DOI: <https://doi.org/10.18434/M32206> (redirects to the landing page below)
- Landing page: <https://data.nist.gov/od/id/mds2-2206>
- Machine-readable record (NIST Research Materials/Metadata, RMM API):
  <https://data.nist.gov/rmm/records?@id=ark:/88434/mds2-2206>
- Contact: Takuma Nakamura (takuma.nakamura@nist.gov), NIST
- Description (verbatim from the record): "This data was used for main
  results for the paper entitled 'Coherent Optical Clock Down-Conversion
  for Microwave Frequencies with 10-18 Instability'. We could calculate
  relative phase fluctuation and Allan deviation for both Yb optical
  clocks and 10 GHz microwaves. Uncertainty of our down-conversion system
  was also calculated from this."

**Files fetched (checksums of the actual bytes retrieved 2026-08-10;
these match the SHA-256 values NIST itself publishes alongside each file,
verified byte-for-byte):**

| File | SHA-256 (verified against NIST-published `.sha256`) | Size | Rows | Committed to repo? |
|---|---|---|---|---|
| `Yb_Clock_phase(rad) vs time.csv` | `c00f2c5c03c3ef0cf346de9917b66800eccd4147cc86af88feae9d02725baad9` | 2,268,460 B | 44,002 | **Excerpt only** (first 20 rows, `benchmarks/fixtures/nist_m32206_yb_clock_phase_excerpt.csv`) |
| `10GHz_phase(mrad) vs time.csv` | `9d715e33e8440dc7b84833ee51a48884e7e7091cc632f482ab2812fc466793be` | 2,268,417 B | 44,002 | **Excerpt only** (first 20 rows, `benchmarks/fixtures/nist_m32206_10ghz_phase_excerpt.csv`) |

Column format (no header row in the source files; confirmed by inspecting
the raw bytes): two whitespace-separated float64 columns, `<sample index>
<phase>`. Per the record's own per-file descriptions: `Yb_Clock_phase(rad)
vs time.csv` = "Time, Phase(rad)"; `10GHz_phase(mrad) vs time.csv` =
"first column is time, second is relative phase (mrad)". Both files are
44,002 rows.

**License / redistribution.** `accessLevel: "public"`, license
`https://www.nist.gov/open/license`: "works of NIST employees are not
subject to copyright protection in the United States" (17 U.S.C. § 105);
freely redistributable with attribution to NIST. Redistribution is **not**
the reason these files are excerpted rather than committed in full:
see the next paragraph.

**Why only a 20-row excerpt is committed (not a license issue).** Unlike
the JILA source, nothing here forbids committing the full ~2.2 MB files.
They are excerpted instead because (see `benchmarks/MAPPING.md` and
`benchmarks/RESULTS.md` for the full reasoning) this dataset is a
phase/Allan-deviation instability record for a specific optical-to-
microwave frequency-division scheme, a fundamentally different
measurement category from the systematic-shift/field-gradient physics
this engine computes (CONVENTIONS.md E14b DC Stark, E21 second-order
Doppler), so it never enters any residual computation in
`benchmarks/run_benchmarks.py`. Committing 4.4 MB of numeric data that is
demonstrated-parseable (the loader is real and tested,
`tests/test_benchmarks_loaders.py`) but structurally never used in a
comparison would be repo bloat without benefit; the 20-row excerpts are
enough to exercise and test the parser thoroughly.
`benchmarks/fetch_data.py` re-downloads and checksum-verifies the full
files on demand for anyone who wants to inspect them directly (e.g. to
independently confirm this benchmark's "not comparable" classification).

## 3. Supplemental Material: PRL 133, 023401 (2024) (follow-up, authorized 2026-08-10)

**A. Aeppli, K. Kim, W. Warfield, M.S. Safronova, J. Ye, "Clock with
8x10^-19 Systematic Uncertainty," Phys. Rev. Lett. 133, 023401, published
2024-07-10. DOI: 10.1103/PhysRevLett.133.023401.**

The project owner separately authorized (2026-08-10) fetching this
source's Supplemental Material specifically to check for a residual
stray-field magnitude, quadrant-electrode geometry/voltages, or a stated
`Δα` value that would let the DC-Stark row (source 1) become an
independent forward-comparable case. **Result: the file could not be
retrieved, and a documented cross-check shows it would not have added
anything usable even if it had been.**

**Access attempts (all logged, none bypassed a paywall or used
credentials, prohibited regardless of any authorization):**

| Attempt | URL | Result |
|---|---|---|
| Article abstract page | `https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.023401` | Loads (after a Cloudflare bot-check the browser tool passed automatically); page text explicitly reads "Supplemental Material (Subscription Required)" and "Authorization Required -- We need you to provide your credentials before accessing this content." |
| Supplemental-material anchor/link | `https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.023401#supplemental` | In-page anchor only (`<a name="supplemental">`), not a distinct downloadable file; the one labeled link that reads "Supplemental Material (Subscription Required)" resolves to the same paywalled article-PDF endpoint, confirmed by inspecting the page's DOM directly (`<a href="https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.133.023401">`), not a separate SM file URL. |
| CHORUS public-access accepted manuscript | `https://link.aps.org/accepted/10.1103/PhysRevLett.133.023401` | A legitimately public route (CHORUS is a funder public-access mechanism, no login prompt shown), but the endpoint serves a file download rather than a page; the download could not be captured through any available fetch mechanism (`curl`/`WebFetch`: HTTP 403 from Cloudflare bot-detection even with a browser-like User-Agent; the interactive browser tool triggered a native OS save dialog it cannot read the bytes of, and explicitly refuses a retry of that URL). No file bytes and no checksum were obtained. |
| ResearchGate mirror | `https://www.researchgate.net/publication/382168376` | HTTP 403 (scraper-blocked). |

None of these were pursued further (no credential entry, no institutional
login, no CAPTCHA-solving, no account creation): per this session's
binding safety rules, a paywall is a stop, not an obstacle to route
around, regardless of the owner's fetch authorization.

**Cross-check performed instead (using the already-fetched, checksummed
arXiv v2 e-print, source 1 above):** the arXiv e-print tarball
(`arxiv_2403.10664v2_source.tar.gz`, sha256 above) contains **separate**
`main.tex`/`main.bbl` and `supplementary.tex`/`supplementary.bbl` files,
compiled together into the one PDF already fetched and text-extracted.
This is strong, directly-inspectable evidence that the arXiv v2 PDF
already *is* the main article concatenated with the same Supplemental
Material file submitted to PRL (same LaTeX sources, same authors, same
submission), not an independent, possibly-differing document. Confirming
evidence found by re-reading the extracted text end to end (not assumed):

- The main-text bibliography's reference **[23]** reads verbatim: *"See
  Supplemental Material at [URL Will Be Inserted by Publisher], which
  includes [38-44], for more information"*: this is the paper's own
  citation marker for the journal's Supplemental Material.
- Every place `[23]` is actually cited in the main text body (4
  occurrences: the temperature-sensor uncertainty "4.1 mK [23]", the
  3D1-lifetime "six separate data sets [23]", the BBR "νdyn =
  -153.06(33) mHz at 300 K [23]", and "background gas is dominated by
  hydrogen molecules [23]") corresponds **exactly** to a section that
  exists later in this same PDF, under the compiled document's own
  supplement (roman-numeral sections **I** [3D1/3P1 lifetime uncertainty,
  with subsections "Magnetic field and the tensor shift contribution" /
  "Lifetime Uncertainty Budget"], **II** "Temperature Measurement," **III**
  "Dynamic BBR Shift," **IV** "First Order Zeeman Shift," **V**
  "Background Gas Shift," confirmed complete: the document ends with
  this section's own separate numbered bibliography ([1]-[14]), not a
  truncation).
- **The DC Stark paragraph in the main text cites no such reference at
  all**: its only citation is `[29]`, a generic literature citation for
  "stray electric fields can shift the clock transition frequency"
  (a different, earlier paper on the general effect, not this paper's own
  supplemental data), and the paragraph is fully self-contained,
  stating only the resulting shift (`-9.8 +/- 0.7 x 10^-20`), never a
  field magnitude, exactly as already documented in section 1 above.

**Conclusion:** the Supplemental Material has exactly five sections (I-V
above), none of them about DC Stark, and the DC-Stark discussion's total
absence of any supplemental-material citation is itself evidence (not
merely an absence of evidence) that the officially separate PRL
Supplemental Material (which this session could not access directly)
does not contain additional DC-Stark quantitative data either. See
`benchmarks/MAPPING.md` for how this affects (does not change) this
benchmark's outcome.

**Checksums:** none recorded for this source: no file bytes were
successfully retrieved from `journals.aps.org` or `link.aps.org` in this
session (see the access-attempts table above). This is disclosed
explicitly rather than fabricated.

## 4. arXiv:1706.01944: NPL Rydberg electrometry (follow-up, authorized 2026-08-10)

**W. Bowden, R. Hobson, P. Huillery, P. Gill, M.P.A. Jones, I.R. Hill,
"Rydberg Electrometry for Optical Lattice Clocks," arXiv:1706.01944v1
[physics.atom-ph] (2017-06-06). Published as Phys. Rev. A 96, 023419 (2017).**

- Abstract page: <https://arxiv.org/abs/1706.01944>
- PDF: <https://arxiv.org/pdf/1706.01944v1>
- TeX source bundle: <https://arxiv.org/e-print/1706.01944v1>
- Version fetched: **v1** (the only version: this submission has no v2).

**Files fetched (checksums of the actual bytes retrieved 2026-08-10):**

| File | SHA-256 | Size |
|---|---|---|
| `arxiv_1706.01944v1.pdf` | `f8b24d5a3cd8a7b2d1a2bd078b866526229883c919649519fd122fa7d118ef51` | 1,111,299 B |
| `arxiv_1706.01944v1_source.tar.gz` | `bc0b435aa91320e4a1b3f866e31740f5914d441380b1ca42abbcdb61346d4f55` | 477,641 B |

**License / redistribution.** Same rights page pattern as arXiv:2403.10664
(section 1 above): `http://arxiv.org/licenses/nonexclusive-distrib/1.0/`:
arXiv's default non-exclusive submission license, not a grant of
third-party redistribution. Per the same binding instruction as section 1,
**the PDF and TeX source are not committed to this repository**;
`benchmarks/fetch_data.py` re-downloads and checksum-verifies both on
demand. The two numeric values transcribed into
`benchmarks/loaders.py`'s `NPL_RESIDUAL_FIELD_V_PER_M`/
`NPL_PUBLISHED_SHIFT` are used under the same scholarly-citation practice
as every other transcribed value in this project, each with an exact
section/page citation (`benchmarks/MAPPING.md`).

**What the paper actually contains (relevant to this benchmark),
independently verified from the fetched, `pdftotext`-extracted full text
(not trusted from an internal research dossier that first flagged this
source, which explicitly called for independent re-verification):**

- Section IV ("Rydberg electrometry using EIT"), the paragraph beginning
  "Next the external field was switched off...": *"A fit to the
  resulting splitting revealed a residual electric field of 1.52
  +0.62(stat)/-0.22(stat) +0.05(sys)/-0.03(sys) V m⁻¹ most likely due to
  patch potential on the surrounding chamber."*: a stray field
  measured via Rydberg-state EIT spectroscopy, **independent of the
  clock transition itself**.
- Same section, next paragraph: *"Translating this electric field and
  corresponding uncertainty to the DC Stark shift of the ¹S₀-³P₀ clock
  transition results in a fractional frequency shift of -1.6 (+0.4/-1.6)
  x 10⁻²⁰."*
- Reference [3] (the paper's only Δα-relevant citation, cited in the
  introduction's sanity-check example "an electric field of 570 V/m
  yields a DC Stark shift of 1 Hz [3], or 2 × 10⁻¹⁵ in fractional
  units"): **T. Middelmann, S. Falke, C. Lisdat, U. Sterr, Phys. Rev.
  Lett. 109, 263004 (2012)**: the exact same paper this project's
  `cliffordclock.ensemble.species.SR87` registry entry cites for `Δα`.
  Independently cross-checked: running this project's own pipeline at
  570 V/m (`Sr87`, `coupling.type: stark_dc`) predicts a shift of
  `-2.3297e-15` fractional, i.e. `-0.99997 Hz`: rounds to the paper's
  own quoted "1 Hz", confirming the registry's `Δα` and the paper's are
  the same source used consistently.

No ancillary/supplementary machine-readable data files exist for this
arXiv submission (confirmed by listing the e-print tarball: only
`main.tex`/`main.bbl`, figure PDFs/EPS files, no ancillary data table):
the two values above are the paper's only quantitative DC-Stark-relevant
content, transcribed by hand with exact citations, per the same normal
practice as `benchmarks/fixtures/jila_2403_10664_table1.csv`.

## 5. Metrologia 63, 025002 (2026): USTC Sr1 evaluation (follow-up, authorized 2026-08-10)

**Zhi-Peng Jia et al., "Improved systematic evaluation of a strontium
optical clock with uncertainty below 1×10⁻¹⁸," Metrologia 63, 025002
(2026). DOI: 10.1088/1681-7575/ae449e. Published 2026-03-05. The published
version of arXiv:2509.13991 (not separately fetched: the owner-provided
published PDF supersedes it for this benchmark).**

- Journal page: <https://doi.org/10.1088/1681-7575/ae449e>
- Owner-provided file (not fetched by this session): local path
  `/Users/mrud/Downloads/Jia_2026_Metrologia_63_025002.pdf`

**Provenance (owner-provided file, retrieved/verified 2026-08-10):**

| File | SHA-256 | Size |
|---|---|---|
| `Jia_2026_Metrologia_63_025002.pdf` | `9cfc227150ad7129dc86091999a98f96e4c91164653e7ceb5c38512c39ed6d0d` | 3,811,417 B |

**License.** Confirmed directly from the PDF itself (page 1 sidebar,
"OPEN ACCESS" banner, and the copyright block): *"Original content from
this work may be used under the terms of the Creative Commons Attribution
4.0 licence. Any further distribution of this work must maintain
attribution to the author(s) and the title of the work, journal citation
and DOI."*: **CC BY 4.0**, confirmed, not assumed from the dossier.
This permits committing extracted excerpts freely with attribution (per
the benchmark protocol): see
`benchmarks/fixtures/ustc_metrologia_63_025002_sec3_5_table3_excerpt.txt`,
a verbatim excerpt of Section 3.5 and Table 3 with full attribution. The
owner-provided PDF itself (3.8 MB) is not committed to this repository
(large, and the excerpt already contains everything this benchmark uses from it);
its SHA-256 above is the provenance record.

**What the paper actually contains (relevant to this benchmark, own independent
read of the owner-provided PDF, not trusted from the dossier):**

- Section 3.5 "Other minor systematic shifts," subsection "Residual DC
  Stark shift" (printed page 9 of the PDF): *"this shift originates from
  static electric fields due to charge accumulation on viewports [60]. As
  reported in our previous work [30], the y-component of the field
  caused a shift of 1.4(5.2) × 10⁻²¹. Given ΔνDC ∝ E²stat and Estat ∝
  r⁻² dependence, the shorter y-axis viewport distance (142 mm vs 237 mm
  for x,z) yield an 8× larger shift along y. Electrostatic shielding
  further suppresses x,z-axis fields by a factor of 3 according to FE
  simulations. Combining these, we constrain the total shift to
  0.0(0.1) × 10⁻¹⁹."*
- Table 3 "USTC Sr1 uncertainty budget" (printed page 10): row "DC Stark"
  = `0` (`<0.1`), units `1e-19`, consistent with the section 3.5 text.
- Reference [30] in the bibliography (printed page 11): **"Li J et al
  2024 Metrologia 61 015006."** Per the section 3.5 text quoted above,
  *this* reference is the paper that "characterized" an applied external
  field (the source of the 1.4(5.2)×10⁻²¹ y-component prior value):
  i.e. it is the actual field-magnitude source underlying USTC's
  DC-Stark constraint, one citation away. **Not authorized, not fetched,
  flagged as the next candidate** (`benchmarks/MAPPING.md`).

**No independent field magnitude is published in *this* paper**: the
DC-Stark constraint is derived from a prior shift measurement (not a
field) combined with geometric/shielding-factor scaling arguments, the
same structural gap as the JILA arXiv:2403.10664 DC-Stark row (section 1
above). See `benchmarks/MAPPING.md` for the full reasoning.

## 6. Metrologia 61, 015006 (2024): Li J et al, USTC Sr1's previous evaluation (follow-up, authorized 2026-08-10; ACCESS BLOCKED)

**J. Li, X.-Y. Cui, Z.-P. Jia, D.-Q. Kong, H.-W. Yu, X.-Q. Zhu, X.-Y. Liu,
D.-Z. Wang, X. Zhang, X.-Y. Huang, et al., "A strontium lattice clock
with both stability and uncertainty below 5×10⁻¹⁸," Metrologia 61,
015006 (2024). DOI: 10.1088/1681-7575/ad1a4c. Published 2024-01-12.**

This is the paper the USTC 2026 evaluation (section 5 above) cites as
reference [30] for its DC-Stark y-component prior value
(`1.4(5.2)×10⁻²¹`) and, per that paper's own description, the one that
"characterized" the applied external field underlying it, flagged in
section 5/`benchmarks/MAPPING.md` as the next authorization candidate.
Fetching it was authorized for this benchmark, 2026-08-10.

**Outcome: no file could be retrieved by any route attempted; no
checksum recorded; no content extracted or examined.** This is reported
precisely, not glossed into a classification this session has no basis
for.

**Access attempts (all logged, none bypassed a paywall or used
credentials, prohibited regardless of authorization):**

| Attempt | URL | Result |
|---|---|---|
| DOI resolution | `https://doi.org/10.1088/1681-7575/ad1a4c` | 302 redirect to the IOPscience landing page below. |
| IOPscience landing page (interactive browser, full page render) | `https://iopscience.iop.org/article/10.1088/1681-7575/ad1a4c` | Loads; explicitly **not** marked open access (no "OPEN ACCESS" banner, unlike section 5's 2026 paper): copyright line reads "© 2024 BIPM & IOP Publishing Ltd. All rights, including for text and data mining, AI training, and similar technologies, are reserved." The "Access this article" panel reads: "The computer you are using is not registered by an institution with a subscription to this article," offering only Login / Purchase (Article Galaxy, CCC RightFind) / Rent (DeepDyve): no free-access or accepted-manuscript route of any kind (DOM inspected directly for a hidden PDF/accepted-manuscript/CHORUS-style link, as found for the JILA PRL paper, section 3 above; none exists on this page; the only "open access"-labeled links found are generic footer links to IOP's general publishing-policy page, not specific to this article). |
| arXiv mirror search | 4 independent `WebSearch` queries (title text, author + subject-matter keywords, `site:arxiv.org "Metrologia 61" "015006"`, USTC institutional-repository keywords) | No arXiv preprint or USTC-hosted copy found for this specific paper: every hit was either the IOPscience page itself, an unrelated paper, or the same paywalled ResearchGate listing below. (Contrast with section 4's NPL paper and section 5's USTC 2026 paper, both of which do have accessible full text.) |
| ResearchGate listing | `https://www.researchgate.net/publication/377137903_A_strontium_lattice_clock_with_both_stability_and_uncertainty_below_510-18` | HTTP 403 (scraper-blocked, same failure mode as the JILA-follow-up's ResearchGate attempt, `benchmarks/SOURCES.md` section 3's historical note). |

**No credential entry, institutional login, purchase, rental, or paywall
bypass of any kind was attempted**, per this project's binding safety
rules: unlike the PRL 133,023401 Supplemental Material (section 3
above) and the USTC 2026 paper (section 5, owner-provided), there is no
free/legitimate route to this paper's content available to this session.
**Per the benchmark protocol ("if gated everywhere, report
precisely and stop"), no further access attempt was made.** The owner
may supply the PDF directly (as done for section 5's USTC 2026 paper) if
this source is to be examined in a future pass of this benchmark.

**Consequence for classification (see `benchmarks/MAPPING.md`):** with
zero content examined, this session has no basis to classify this
paper's DC-Stark characterization as "budget-only," "reproducibility-
grade," or "blind-prediction-grade": any such classification would be
a guess dressed as a finding. The correct status is **"not
accessed: classification not possible without a copy,"** reported as
its own outcome, not silently folded into "not applicable" (which this
project uses only for content that *was* examined and found not to map).

## 7. quant-ph/0701215v1: Roos et al., the two-ion quadrupole-shift benchmark (owner-supplied, 2026-08-11)

**C. F. Roos, M. Chwalla, K. Kim, M. Riebe, R. Blatt, "Designer atoms
for quantum metrology," published as Nature 443, 316 (2006), preprint
quant-ph/0701215v1.** (An earlier draft of this section carried the
byline of a different Roos et al. paper; corrected against the
owner-supplied PDF's own title page in review.)

- arXiv abstract page: <https://arxiv.org/abs/quant-ph/0701215>
- Owner-supplied file (not fetched by this session): local path
  `/Users/mrud/Downloads/0701215v1.pdf`

**Provenance: owner-supplied primary text, recorded as such; no fetch
checksum table here.** Unlike sections 1-6 above, this session did not
itself retrieve this file (no `curl`/`WebFetch` request was made for it),
so there is no fetch log or byte-checksum for this session to accurately
report; performing one anyway (hashing a file this session never
downloaded) would misrepresent the provenance as a fresh, independently
verified fetch when it is not. The full primary-text read and extraction
of every number this benchmark uses (Eq. 1, the Fig. 4a slope/offset, the
two-ion entangled-state Theta = (5/12)*h*a relation) was already
performed and independently reviewed against the primary text
(Roos et al., Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1/Fig. 4a)
in this project's internal review (2026-08-11, "PRIMARY
VERIFIED"): this benchmark extends that already-logged extraction
rather than re-reading the PDF a second time, the same "extend, do not
duplicate" pattern `benchmarks/loaders.py`'s `JILA_BBR_TEMPERATURE_K`
addendum uses for the already-fetched arXiv:2403.10664 source (section 1
above).

**What this benchmark consumes from it** (both values restated as
`benchmarks/loaders.PublishedBand` entries, `ROOS_MEASURED_SLOPE_HZ_MM2_PER_V`/
`ROOS_FIT_OFFSET_HZ`; full derivation:
`benchmarks/run_roos_quadrupole_slope.py`):

- **Fig. 4a linear-fit slope**, p.9: "a = 2.975(2) Hz mm^2/V" of the
  two-ion entangled state Psi_1's measured quadrupole-shift parity-
  oscillation frequency against the applied, mechanically (omega_z)
  calibrated axial gradient dE_z/dz = 12-48 V/mm^2 at beta=0.
- **Fig. 4a linear-fit offset**, same page: "Delta_0/(2*pi) = -2.4(1)
  Hz" at zero applied gradient (-2.9 Hz attributed to second-order Zeeman
  at the 2.9 G bias, remainder to a residual stray quadrupole field),
  does not enter the slope comparison, see the script's module docstring.
- **Eq. 1** (p.6): the primary-text quadrupole-level-shift formula
  already adopted verbatim as `docs/CONVENTIONS.md` E34's leading form
  (§14, G8 sign-off gate edit 1a), re-derived independently for this
  benchmark's two-ion extension, not re-transcribed.
- **The two-ion entangled state** Psi_1 = (|-5/2>|+3/2> +
  |-1/2>|-1/2>)/sqrt(2) and the stated 24/5 two-ion enhancement (relative
  to a single |-5/2> ion) and gradient-doubling mechanism (p.5-6: "the
  presence of a second ion doubles the electric field gradient at the
  location of the other ion"), both used as structural pins on this
  benchmark's own derivation (see the script's module docstring), not
  taken on faith.
- **Ca+:D5/2 Theta values**, already registered in
  `cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS["Ca+:D5/2"]`
  (`theta_au=1.83`, `theory_theta_au=1.917`), not re-transcribed here,
  consumed directly from the registry by the benchmark script.

**License/committing note.** The PDF itself (214,638 bytes) is not
committed to this repository, for the same repo-hygiene reasoning as
every other non-CC-BY source in this document (sections 1, 4 above); no
excerpt is needed since every number this benchmark uses is already
transcribed, with page citations, against the primary text (Roos et al.,
Nature 443, 316 (2006), quant-ph/0701215v1, Eq. 1/Fig. 4a)
and restated with citations in `benchmarks/loaders.py`.

## Neither source's raw/committed content is part of the installed package

`benchmarks/` (this whole directory) is not part of the `cliffordclock`
wheel: `pyproject.toml`'s `[tool.setuptools.packages.find] where = ["src"]`
only discovers packages under `src/`, and `benchmarks/` lives at the repo
top level alongside `src/`, `tests/`, `examples/`, the same mechanism
that already excludes `examples/`, `notebooks/`, and `tests/`. No
packaging-config change was needed or made for this benchmark (verified by
reading `pyproject.toml`, not assumed).

## 8. arXiv:2109.12238 / Nature 602, 420 (2022): Bothwell et al. mm-scale
   gravitational redshift

**T. Bothwell, C. J. Kennedy, A. Aeppli, D. Kedar, J. M. Robinson,
E. Oelker, A. Staron, J. Ye, "Resolving the gravitational redshift
across a millimetre-scale atomic sample," Nature 602, 420-424 (2022);
preprint (title differs: "within a millimeter atomic sample")
arXiv:2109.12238 [physics.atom-ph]. Author list verified against the
arXiv listing in the WP22 independent review (an earlier staged draft
carried a fabricated middle initial, caught there; first author is
Tobias Bothwell). This block was checked against the arXiv
listing directly.**

- Abstract page: <https://arxiv.org/abs/2109.12238>
- PDF: <https://arxiv.org/pdf/2109.12238>
- Read cover to cover during the WP22 research sweep, 2026-08-11 (see
  Bothwell et al., Nature 602, 420 (2022), arXiv:2109.12238, for the
  full extraction);
  this specific benchmark-building pass (WP22 builder, this document
  section) did NOT independently re-fetch or re-verify the checksum:
  flagged here so the coordinator can complete that step to this
  document's usual standard (SHA-256 + independent verification) before
  treating this citation as fully audited to the same bar as sources 1-5
  above.
- Ancillary/data files: none found by the dossier's research pass (a hard
  negative: see `benchmarks/MAPPING.md`'s WP22 addendum "Data
  availability").

**Secondary source for the reference gravity value:** D. van Westrum,
NOAA Technical Memorandum NOS NGS-77 (2019): cited BY Bothwell et al.
in their Methods ("Known Redshift") for the USGS-surveyed local
gravitational acceleration at their Boulder, CO site, `g = 9.796 m/s^2`.
Not independently fetched by this project; used only as Bothwell's own
cited value (transcribed from their Methods text, not re-derived).

**What this source was used to conclude:** the WP22 Part 3 Bothwell
reproducibility case (`benchmarks/RESULTS.md`'s "Reproducibility case:
Bothwell..." section, staged above): the sample geometry, reference
gravity, and both corrected measured slopes this project's
`benchmarks/run_bothwell_redshift.py` compares its own real-pipeline
prediction against.
