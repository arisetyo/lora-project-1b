# Annotated Companion to the Article Draft

A section-by-section read-through of [scientific_article_draft.md](scientific_article_draft.md).
For each chapter/paragraph: **Explanation** (what it says, plainly), **Comment** (my editorial view),
and **Notes** (things to verify, risks, or reviewer flags).

Legend for Notes severity: 🔴 blocker / likely reviewer objection · 🟡 worth tightening · 🟢 fine, minor polish.

---

## Title

> "Evaluation of Manual LoRA Adaptation on GPT-2 for PubMed-Based Tuberculosis Clinical Q&A Generation"

- **Explanation:** States the method (manual LoRA), the base model (GPT-2), the data source (PubMed), and the task (TB clinical Q&A generation).
- **Comment:** Accurate and specific — it promises exactly what the paper delivers, which is good. "Manual" is a meaningful qualifier here because it distinguishes your hand-written `LoRALinear` from the HuggingFace PEFT library.
- **Notes:** 🟡 It's long (15 words). If the journal prefers shorter titles, "Manual LoRA Adaptation of GPT-2 for Tuberculosis Clinical Q&A" still carries the core claim. 🟢 "Q&A" may need to be spelled out ("Question Answering") depending on house style.

## Abstract

- **Explanation:** One-paragraph summary: motivation (efficient medical adaptation), data (707 TB QA pairs from PubMedQA), method (frozen GPT-2 + LoRA on `c_attn`), evaluation (in-domain + OOD perplexity), headline results (PPL 30.85→24.38→21.08 as rank rises; OOD stable at 55–57), and takeaway (PEFT works with few parameters; the study is positioned as an accessible, reproducible starting point for AI research on TB and other high-burden diseases in settings such as Indonesia).
- **Comment:** Strong and evidence-led — it leads with concrete numbers, which reviewers like. The claim is appropriately bounded ("can improve domain adaptation … with few trainable parameters") rather than overselling clinical usefulness. The new closing line carries the regional/accessibility framing, which improves fit for the PCPR audience.
- **Notes:** 🟡 The abstract reports rank and epochs varying together but doesn't flag the confound; that's acceptable for an abstract, but make sure the body (Limitations #2) is unmissable. 🟢 243/250 words — any edit risks tipping over the limit, so recount after changes. 🟡 It claims general capability is "preserved," but the qualitative section shows the model was never *generally capable* to begin with (124M GPT-2 is weak); "general-language modeling on WikiText-2 is preserved" is the precise claim. 🟢 The "starting point … such as Indonesia" line is a positioning claim, not a demonstrated outcome — keep it modest.

## Introduction

**Paragraph 1 (PEFT / LoRA framing).**
- **Explanation:** Sets up why full fine-tuning is costly and how PEFT — and LoRA specifically — solves it, name-checking adapters and prompt tuning as siblings.
- **Comment:** Clean funnel from general problem → PEFT → LoRA. Citations [1–4] are well placed.
- **Notes:** 🟢 Good. Consider one sentence explicitly stating that LoRA's appeal here is *iteration speed on limited hardware*, tying back to your real constraints.

**Paragraph 2 (medical domain adaptation + catastrophic forgetting).**
- **Explanation:** Establishes that domain adaptation helps in biomedical NLP (BioBERT/PubMedBERT/ClinicalBERT), and introduces catastrophic forgetting as the risk your OOD metric guards against.
- **Comment:** This paragraph does double duty — it justifies both the domain-adaptation goal and the OOD-monitoring design choice. That's efficient writing.
- **Notes:** 🟡 References [6–8] are encoder (BERT-family) models, whereas your work is a decoder/generative LM; a reviewer may note the mismatch. One sentence acknowledging that prior biomedical adaptation focused on encoders, and that you study a generative setting, would pre-empt this.

**Paragraph 3 (TB motivation + objective).**
- **Explanation:** Motivates TB specifically — now including Indonesia's high TB burden — then states the objective: can manual LoRA improve in-domain modeling while preserving general capability, and how does rank affect the trade-off.
- **Comment:** The objective sentence is the backbone of the paper and is phrased well. Good that it explicitly names the rank trade-off, which the results then deliver on. Anchoring TB to Indonesia early sets up the significance framing that pays off in the Discussion.
- **Notes:** 🔴 The "Indonesia carries one of the highest TB burdens" claim currently has no citation; add a source (the WHO Global TB Report's country ranking already underlies [12] and can support it). 🟢 Confirm the WHO report year you cite.

**Paragraph 4 (accessibility / positioning — NEW).**
- **Explanation:** Frames the study as an accessible, fully reproducible entry point for life-science/pharmacy newcomers (small open model, open data, lab-grade hardware, open code) — a starting point, not a finished clinical solution.
- **Comment:** This is the paragraph that finally states the article's real value out loud. For the PCPR readership and your "spark" goal, it is arguably the most strategically important addition, and it is well hedged.
- **Notes:** 🟡 "Hardware within reach of a typical laboratory" is true (you ran it on an RTX 3060 / Apple MPS) — state the actual hardware once in Methods so the claim is concrete. 🟢 Keep the "not a finished clinical solution" caveat; it manages reviewer expectations.

**Contributions list.**
- **Explanation:** Four bullet contributions: end-to-end manual pipeline, dual-metric evaluation, rank trade-off analysis, and an accessible open reference implementation for resource-constrained settings.
- **Comment:** Reasonable and honest — these are engineering/methodological contributions, not a novel algorithm, and the list doesn't pretend otherwise.
- **Notes:** 🟡 Contribution 1 ("demonstrating a pipeline") is weak as a scientific contribution on its own. The strongest framing is the empirical finding (in-domain gains with stable OOD); consider leading with that. 🟢 New contribution #4 makes the accessibility/reuse value explicit, consistent with the new positioning paragraph.

## Methods

**Experimental Design.**
- **Explanation:** Two conditions (base vs. LoRA), focused ranks 4/8/16, with epochs scaled to rank; flags the confound forward to Limitations. Second paragraph is the ethics/consent statement (N/A — public secondary data).
- **Comment:** Putting the rank/epoch confound right in the design section (not hiding it) is exactly the right call and builds reviewer trust. Ethics statement correctly placed in Methods per PCPR.
- **Notes:** 🔴 The biggest scientific vulnerability of the paper lives here: epochs are confounded with rank. You've disclosed it, which is the minimum; the *fix* (a fixed-epoch run) is what a tough reviewer will ask for. Be ready to either run it or argue why the trend is still informative.

**Dataset.**
- **Explanation:** Source (PubMedQA via BigBio), TB keyword filtering with stated inclusion/exclusion, ~707 pairs, "Question/Answer" formatting, local cache, train/val/test split.
- **Comment:** Now meets PCPR's "inclusion/exclusion criteria" requirement. Stating 707 explicitly is important — reviewers always ask dataset size.
- **Notes:** 🔴 The actual train/val/test split sizes are not given ("according to the project pipeline"). Add the concrete counts and the split ratio/seed — reproducibility reviewers will require it. 🟡 List the TB keywords used (or point to the script) so the filtering is reproducible. 🟡 Note whether PubMedQA's "long answer" or "context" field was used as the answer.

**Model and Training Configuration.**
- **Explanation:** Frozen GPT-2, LoRA on `c_attn`, the BA decomposition math, A=Kaiming/B=0 init, α=r coupling, and the hyperparameter list.
- **Comment:** The math and init rationale (B=0 ⇒ ΔW=0 at start) are correctly explained. α=r ⇒ scale 1 is a sensible, clearly stated choice.
- **Notes:** 🟡 You apply LoRA only to `c_attn` (not `c_proj`, MLP, etc.); state that explicitly as a scoping choice — it affects parameter counts and is a common reviewer question. 🟢 Confirm batch size 4 + max-len 512 matches what actually ran. 🟡 No seed reported for training; add it.

**Evaluation Protocol.**
- **Explanation:** Quant = in-domain PPL + OOD PPL (WikiText-2); qual = prompt battery (greedy + sampled), side-by-side. Second paragraph is the reproducibility/repo statement.
- **Comment:** Compact and complete. Good that the repo + artifacts are named for reproducibility.
- **Notes:** 🟡 Define perplexity in one sentence (exp of mean token NLL) — a pharmacy-journal audience may not assume it. 🟡 State how many validation texts and how many WikiText-2 samples were used (the code uses up to 100 / 50 respectively); these affect the reported numbers. 🔴 Repo must be public before submission.

## Results and Discussion

**Quantitative Results (Table 1 + text).**
- **Explanation:** Table 1 gives params/PPL/OOD per rank; text reports the ~31.7% in-domain reduction (r4→r16), the stable OOD band (55–57), and the smooth rank-16 val-loss decline (0.607→0.517, no overfitting).
- **Comment:** This is the core evidence and it's presented cleanly. The val-loss-stability point is a nice supporting detail. Figures 1–2 are well integrated.
- **Notes:** 🔴 **No baseline (un-adapted GPT-2) row in Table 1.** The whole "improvement" framing implies a comparison, but the base model's val PPL isn't tabulated. The training script *does* compute a baseline PPL — add it as a row so "reduction" is anchored to a real number, not just to rank-4. 🟡 The "31.7% reduction" is r16 vs r4, not vs base — phrase carefully so readers don't think it's vs baseline. 🟡 No variance/seeds: single run per config means no error bars; acknowledge or add seeds.

**Qualitative Results (Table 2 + two analysis paragraphs).**
- **Explanation:** Replays the prompt battery on base vs. r16; Table 2 shows 3 representative completions. Paragraph 1: LoRA shifts to biomedical register and shorter answers. Paragraph 2: neither model is factually reliable; OOD sanity prompts stay wrong.
- **Comment:** This is, to me, the most intellectually honest part of the paper and paradoxically its strongest. Most "LoRA works" papers stop at perplexity; you show that perplexity ↓ does **not** equal clinical correctness. That candor is publishable and reviewer-disarming.
- **Notes:** 🟡 The verbatim quotes are unflattering (repetition, hallucinated `rpoB`→IL-6 etc.). Decide if you're comfortable publishing them. I'd keep them — they make the limitations concrete — but it's your call. 🟡 "register improved but facts didn't" is a qualitative judgment by you (the author); a reviewer may want a small structured rubric/count (e.g., N/8 prompts with on-topic terminology) to make it less subjective. 🟢 The OOD sanity-check result is a nice touch that reinforces the catastrophic-forgetting framing.

**Discussion (paragraphs 1–3).**
- **Explanation:** Restates that LoRA adapted GPT-2 with few params, that OOD stability implies low forgetting, and that the manual implementation behaves consistently with PEFT literature; states the defensible core claim.
- **Comment:** Appropriately measured. The explicit "strongest scientific claim" sentence is a good rhetorical move — it tells the reviewer exactly how far you're willing to generalize.
- **Notes:** 🟡 "consistent with reports in the PEFT literature" would be stronger with a one-line quantitative comparison to a published LoRA result, even informal. 🟡 Slight redundancy with the Abstract and Conclusion; trim if word count is tight.

**Discussion (inference-server paragraph).**
- **Explanation:** Argues for a standardized inference server to enable consistent human A/B evaluation and cross-institution collaboration.
- **Comment:** This is really *future work* framed as discussion. It's well written but forward-looking.
- **Notes:** 🟡 It partly duplicates the Future Directions list. Consider compressing one of the two so the paper doesn't say the same thing twice.

**Discussion (applied relevance / "spark" paragraph — NEW).**
- **Explanation:** Lays out the roadmap: the same recipe scaled to modern medical foundation models (MedGemma [20]) on larger/representative corpora (incl. national health data with governance safeguards), supporting frontline services (Puskesmas, apotek), and generalizing to other high-burden Indonesian conditions (smoking-related respiratory disease, diabetes) — framed as an "early spark."
- **Comment:** This is the strategic payload for PCPR and the Center-of-Excellence vision. The hedging ("in principle," "could eventually," "subject to safeguards") is exactly right — it signals ambition without promising what the 124M result cannot support.
- **Notes:** 🟡 The BPJS/government-data mention is sensitive; the privacy/governance caveat is in place — keep it. 🟡 MedGemma is multimodal and Gemma-3-based, far larger than GPT-2; a reviewer may note the leap, so framing it as a future direction (not continuity) is important. 🟢 Reads as vision, not result — appropriate for Discussion.

**Limitations.**
- **Explanation:** Six limitations: automatic-metric-only, the rank/epoch confound, no inference server, no inter-rater reliability yet, small scale, and no question-token masking.
- **Comment:** Thorough and self-aware — this section will earn goodwill. #2 (confound) and #5 (scale) are the ones reviewers care about most, and they're both here.
- **Notes:** 🟢 Strong as is. 🟡 Consider adding one limitation you currently omit: evaluation used ≤100 validation texts / ≤50 WikiText-2 samples, so PPL estimates have sampling noise.

**Future Directions.**
- **Explanation:** Seven-item plan centered on the inference server, blinded A/B human eval, rubric design, inter-rater metrics, a 50–100 question benchmark, the fixed-epoch ablation, and multi-institution collaboration.
- **Comment:** Concrete and credible. Item (6), the fixed-epoch ablation, directly answers the confound — good that it's promised.
- **Notes:** 🟡 Long single paragraph; PCPR allows it, but a reviewer skims better with the items as the genuine priorities. The human-eval benchmark (5) and fixed-epoch ablation (6) are the two that most increase the *next* paper's value — consider ordering by impact.

## Conclusion

- **Explanation:** Recaps that manual LoRA is viable, r16 is best (PPL 21.08, val-loss ≈0.517, OOD stable), and that capacity↑ helps domain fit if OOD is monitored; lists follow-up work, and closes with a vision sentence offering the study as a reproducible spark for further Indonesian AI health research.
- **Comment:** Tightly tied to the objective and the numbers. Doesn't overclaim. The closing vision sentence mirrors the Introduction's positioning, giving the paper a consistent through-line. Good.
- **Notes:** 🟡 "supports the hypothesis" — the Introduction frames an *objective*, not a formal hypothesis; align the wording (use "objective" in both, or introduce the hypothesis explicitly up front).

## Acknowledgements / Funding / Conflict of Interest

- **Explanation:** Acknowledges Galenic's AI/ML-in-life-sciences R&D context; Funding "None."; COI "None declared."
- **Comment:** All three conform to PCPR's stated formats.
- **Notes:** 🟡 If Galenic provided the compute, some journals consider that a funding/in-kind contribution rather than only an acknowledgement — make sure "Funding: None" is strictly accurate, or move the compute support into Funding as in-kind. 🟢 Confirm the acknowledgement wording is cleared internally.

## References

- **Explanation:** 24 references (MedGemma technical report added as [20]), Vancouver style, numbered by order of appearance.
- **Comment:** Meets PCPR's hard rules (≥20, numbered-by-appearance, ≥80% recent primary). Structurally done.
- **Notes:** 🔴 **Every entry's volume/issue/pages/DOI was reconstructed and must be verified** against the original source — this is the single most error-prone part of the manuscript. 🟡 #1 (Ding, review), #20 (MedGemma, preprint), #23 (PEFT library), and #12 (WHO report) are non-primary; you're within the 80% rule but keep an eye on the ratio if you drop any. 🟡 GPT-2 [15] and MedGemma [20] are technical reports/preprints, not peer-reviewed — fine and standard, but expect no journal DOI. Verify arXiv:2507.05201 and the MedGemma author list; if you specifically mean MedGemma 1.5, cite that model card instead.

---

## Cross-cutting observations (my overall take)

1. 🔴 **Add a base-GPT-2 baseline row to Table 1.** It's the most important missing number; the paper's narrative is comparative but the comparison anchor isn't tabulated.
2. 🔴 **The rank/epoch confound is the #1 reviewer target.** Disclosed well, but the fixed-epoch ablation is the move that converts a "revise" into an "accept."
3. 🟡 **Single run per config, small eval sets.** No error bars; at least state seeds and sample sizes, ideally repeat the best config across a few seeds.
4. 🟢 **The honesty about factual unreliability is an asset, not a weakness.** Keep it. It's what separates this from a naive "LoRA improves medical QA" paper.
5. 🟡 **Encoder-vs-decoder framing.** Your biomedical-NLP citations are mostly encoders; one sentence positioning your generative setting avoids an easy reviewer jab.
6. 🟢 **Scope discipline is good throughout.** The paper consistently claims "in-domain modeling improves, general modeling preserved" and never overreaches into clinical efficacy.
7. 🟢 **Significance framing now present.** The Abstract, Introduction (new positioning paragraph + contribution #4), Discussion (the "spark"/roadmap paragraph), and Conclusion now state the accessibility + Indonesian-TB value explicitly — materially improving fit for PCPR. Keep every such claim in the positioning/future register, since the present result does not yet demonstrate clinical utility.
8. 🟡 **Disclose the institutional relationship.** Given the preliminary PT Galenic–Faculty discussions (including a proposed AI Center of Excellence) and the author's ties to the PCPR-owning faculty, a Conflict-of-Interest *disclosure* is safer than "None declared." Per COPE/ICMJE, disclose relationships that *could be perceived* to influence the work, even if preliminary and non-financial; disclosure protects both author and editors.
