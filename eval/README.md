# Evaluation notes

## Eval A: the plan reader

### Documents
- **Held-out set:** the 20 SBCs whose printed coverage period starts on 1 January 2027 (`eval/split.json`, frozen in DECISIONS 8). They come from Washington PEBB (9), Tennessee (6), Oregon PEBB (4) and CalPERS (1).
- **Dev set:** the other 25 SBCs, from 7 other sponsors.
- Prompts and parsers were developed only on dev documents. The held-out set is scored once, by `make eval-reader`.

### Gold labels (`eval/gold/heldout_gold.json`)
Thirteen fields per held-out SBC, 260 values in all:
- deductible, individual and family
- out-of-pocket limit, individual and family
- in-network cost sharing for primary care, specialist, urgent care, emergency room, generic drugs and inpatient facility
- the three printed coverage-example totals (what Peg, Joe and Mia would pay)

**How they were made, honestly:**
- The values were transcribed by Claude (Opus 5.5), the AI coding agent that also wrote the reader. They were not entered by a person, and no person has checked them.
- Claude read the page text that pypdfium2 extracts, printed around each field by `scripts/gold_snippets.py`. That script only searches the text; no model is involved. Values were typed by hand.
- The gold file was committed before any reader output existed for a held-out document. The commit order in git shows this.
- **One leak, disclosed:** reading held-out text to transcribe it showed some cell patterns, such as "15% coinsurance after $75 copay". To keep the score honest, the reader's parsers were changed afterwards only for failures seen on dev documents. Any held-out-only pattern the parser misses is left in and scored as a failure.

**Judgment calls** are written down per field (`notes`), and 25 values are flagged `ambiguous`. Accuracy is reported both with and without the flagged values. They include:
- separate medical and pharmacy (or specialty-drug) out-of-pocket limits
- Tennessee's tier names, where "employee + spouse + child(ren)" is recorded as family
- cells that list two in-network rates

The conventions are listed in the gold file.

### Scoring (`scripts/reader_eval.py`)
- **A field is correct only if** its quote was verified in the PDF text and the value parsed from the quote equals the gold value. Money must match to the dollar and rates exactly. For cost sharing, copay and coinsurance are compared, and "deductible applies" only where the gold states it.
- **Every gold field ends up in one of three states:**
  - correct
  - wrong: verified, but the value differs from gold
  - needs_check: no verifiable or parseable quote
- **What is reported:** field accuracy with a document-bootstrap 95% CI, the coverage-example match rate, seconds per PDF, peak memory, and a failure log of every miss, with the quote the model returned.

### Coverage-example reconciliation
CMS's service-level inputs for the Peg, Joe and Mia examples were not found, so the examples are not recomputed claim by claim. Instead, `sameaslastyear/rules.py` checks each example against the rest of the same SBC, with a $1 tolerance for rounding:
- **Arithmetic:** deductibles + copayments + coinsurance + limits equals the printed total.
- **Bounds:** the example's deductible is no more than the plan deductible; its cost sharing is no more than the individual out-of-pocket limit; its total is no more than the total example cost.
- **Parameters:** the example box's plan deductible, specialist cost and hospital coinsurance equal the fields read from the summary and the medical-events table.

An example "matches" when every check whose inputs were verified passes.
