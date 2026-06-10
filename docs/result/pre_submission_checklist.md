# Pre-Submission Checklist

Companion to [scientific_article_draft.md](scientific_article_draft.md).
Target journal: **Pharmacology and Clinical Pharmacy Research (PCPR)**, Universitas Padjadjaran — Original Research.
Last reviewed: 2026-06-10.

Work top to bottom. Items marked **(blocker)** will cause desk rejection or a return-for-revision if missed.

---

## 1. Required PCPR templates and forms

- [ ] **(blocker)** Download the journal templates from the PCPR submission page: **Title Page Template**, **Manuscript Template**, and **Table & Figure Format**.
- [ ] **(blocker)** Split the current single draft into the two required files:
  - **Title Page** ← the `TITLE PAGE` block (title, author, affiliation, correspondence).
  - **Manuscript** ← everything from `Abstract` onward (no author-identifying text, for blind review).
- [ ] **(blocker)** Prepare and sign the three mandatory uploads: **Copyright Transfer Agreement**, **Publication Ethics Statement**, and **Cover Letter** (use PCPR's formats).
- [ ] Confirm the manuscript has not been published or submitted elsewhere (PCPR condition of publication).

## 2. Formatting

- [ ] **(blocker)** Convert Markdown → the journal's Word manuscript template.
- [ ] Set body text to **Times New Roman, 12 pt, A4**.
- [ ] Re-check **word count ≤ 3000** (Original Research) after all edits — the qualitative section grew the body; recount in Word.
- [ ] Apply the journal's **three-line table style** (top/header/bottom rules, no vertical rules) to Table 1 and Table 2 per the Table & Figure Format template.
- [ ] Table titles **above** the table; figure captions **below** the figure.
- [ ] Re-export Figures 1–2 at the resolution the template requires (currently 200 DPI; many journals want **300 DPI** — regenerate via `outputs/wandb_charts/make_figures.py` with `dpi=300` if needed).
- [ ] Ensure all tables are real tables, **not images** (already satisfied).

## 3. Abstract, keywords, structure

- [x] Unstructured English abstract ≤ 250 words (currently 243 — re-verify if abstract is edited).
- [x] 3–6 keywords (currently 6).
- [x] Section order matches PCPR: Abstract → Introduction → Methods → Results and Discussion → Conclusion → Acknowledgements → Funding → Conflict of Interest → References.
- [ ] Confirm where PCPR wants **author contributions** (often on the Title Page or in OJS metadata, not in the manuscript body — it was removed from the body to match the template).

## 4. References **(blocker)**

- [x] Vancouver style, numbered by order of appearance.
- [x] ≥ 20 references (currently 23).
- [x] ≥ 80% primary literature from the last 10 years.
- [ ] **(blocker)** **Verify every reference** against the original source: authors, year, **volume, issue, page range**, and DOI/URL. These were reconstructed and must be checked (especially #6 BioBERT pages, #7 PubMedBERT, #11 Kirkpatrick, #12 WHO report edition/year, conference page numbers).
- [ ] Manage references in **Mendeley / EndNote / Reference Manager** (PCPR recommendation) so numbering stays consistent if you add citations.
- [ ] Decide on optional **Cohen (1960)** citation — currently mentioned in Limitations as the *wrong* metric with no reference. Add as a numbered ref if a reviewer is likely to ask, or leave uncited.
- [ ] Re-confirm in-text citation numbering is still monotonic after any reference additions.

## 5. Content still to finalize

- [ ] **Table 2 verbatim quotes** — confirm you are comfortable publishing the exact GPT-2 / LoRA generations shown (they are unflattering by design). Swap prompts if preferred.
- [ ] Decide whether to **expand the qualitative examples** beyond 3 rows (the full battery is in `docs/base-vs-finetuned.md`).
- [ ] Confirm the **supplementary material** policy: does PCPR accept a supplementary file (`base-vs-finetuned.md`)? If not, fold the full battery into the manuscript or drop the reference to it.
- [ ] Add an **environment/reproducibility note** (Python 3.12, torch 2.12, transformers, device, training commit `b524352`) — the standalone Appendix was removed, so place this in Methods or supplementary.

## 6. Reproducibility / data and code

- [ ] **(blocker)** Make the GitHub repo **public** (or arrange reviewer access): https://github.com/arisetyo/lora-project-1b — the Methods section links to it.
- [ ] Confirm the repo contains the artifacts the paper cites: `outputs/20260609-second-ablation-run_focused-ranks/results.csv`, `scripts/run_prompt_battery.py`, `outputs/wandb_charts/make_figures.py`, and the figures.
- [ ] Verify the `data/tb_qa.json` distribution note is consistent with the **PubMedQA / BigBio license** (state whether the processed dataset can be redistributed or only regenerated via the script).
- [ ] Tag or reference the exact commit used for the reported ablation for reproducibility.

## 7. Declarations and metadata

- [x] Funding: "None."
- [x] Conflict of Interest: "None declared."
- [x] Ethics / informed-consent statement in Methods (N/A — public secondary data, no human subjects).
- [ ] Add author **ORCID** (commonly required in OJS author metadata).
- [ ] Confirm the corresponding-author email (arie@galenic.systems) is correct and monitored.
- [ ] Acknowledgements: confirm the Galenic R&D wording is approved internally for publication.

## 8. Language and proofreading

- [ ] Full English proofread (a TB/biomedical reviewer will read closely).
- [ ] **Define every abbreviation at first use**: LoRA, PEFT, GPT-2, QA, PPL and OOD (used in Table 1), HIV, BCG, MDR-TB, *M. tuberculosis*.
- [ ] Consistent spelling (e.g., "diarrhoea" vs "diarrhea"; pick UK or US and apply throughout).
- [ ] Consistent number/decimal and unit formatting; consistent use of "rank 16" vs "r = 16".
- [ ] Check that all in-text figure/table references resolve (Figure 1, Figure 2, Table 1, Table 2 rows 1–3).

## 9. Final pre-upload pass

- [ ] Read the assembled Word manuscript end-to-end once.
- [ ] Confirm no author-identifying information remains in the **Manuscript** file (blind review).
- [ ] Verify figures/tables are embedded and render correctly in Word (not just linked).
- [ ] Cross-check the Cover Letter states the article type (Original Research) and confirms originality/non-duplicate submission.
