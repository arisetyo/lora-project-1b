<!--
JOURNAL: Pharmacology and Clinical Pharmacy Research (PCPR), Universitas Padjadjaran.
Article type: Original Research (target ≤ 3000 words; final formatting: Times New Roman 12 pt, A4).
Submission is in two parts:
  1) TITLE PAGE (separate template): title, author(s), affiliation, correspondence — see block below.
  2) MANUSCRIPT (this file): from Abstract onward.
Required section order for Original Research:
  Abstract → Introduction → Methods → Results and Discussion → Conclusion →
  Acknowledgements → Funding → Conflict of Interest → References.
-->

# TITLE PAGE (submit on the separate Title Page Template)

## Title

Evaluation of Manual LoRA Adaptation on GPT-2 for PubMed-Based Tuberculosis Clinical Q&A Generation

## Authors and Affiliation

M. Arie Prasetyo¹*

1. PT Galenic Systems Indonesia, Bandung, Indonesia

*Correspondence: arie@galenic.systems

---

# MANUSCRIPT (submit on the Manuscript Template)

## Abstract

This study evaluates the effectiveness of a manually implemented Low-Rank Adaptation (LoRA) method for adapting GPT-2 to a generative tuberculosis (TB) clinical question-answering task. The dataset was built from the PubMedQA collection (`bigbio/pubmed_qa`) by filtering TB-related articles and formatting them as causal language modeling pairs in the "Question: ... Answer: ..." structure, yielding approximately 707 TB-related QA pairs. All base GPT-2 parameters were frozen, while LoRA parameters were injected into the `c_attn` attention projections and trained using a parameter-efficient fine-tuning approach. Experiments varied the LoRA rank (r = 4, 8, and 16) together with the number of training epochs (3, 5, and 10, respectively) to evaluate the effect of adapter capacity on domain adaptation. Quantitative evaluation used in-domain perplexity on the TB validation set and out-of-domain perplexity on WikiText-2 as an indicator of general-language retention. The results show that increasing the LoRA rank consistently reduces in-domain perplexity, from 30.85 at rank 4 to 24.38 at rank 8 and 21.08 at rank 16. In contrast, out-of-domain perplexity remained relatively stable in the 55–57 range, indicating that increased domain specialization was not accompanied by a significant change in the model's general-language capability. These findings show that a manual LoRA implementation can improve domain adaptation on GPT-2 with few trainable parameters, and the study is positioned as an accessible, reproducible starting point for parameter-efficient AI research on TB and other high-burden diseases in settings such as Indonesia.

Keywords: LoRA, GPT-2, Tuberculosis, PubMedQA, Domain Adaptation, Parameter-Efficient Fine-Tuning

## Introduction

Adapting language models to the medical domain requires parameter efficiency due to limited computational resources and the need for repeated experimentation. Full fine-tuning of every model weight is costly and risks degrading the broad linguistic competence acquired during pretraining. Parameter-efficient fine-tuning (PEFT) addresses this by updating only a small set of additional parameters [1]. Among PEFT methods, Low-Rank Adaptation (LoRA) enables fine-tuning by adding trainable low-rank matrices to selected weight matrices without updating the base model [2], while related approaches such as adapter modules [3] and prompt tuning [4] pursue the same parameter-efficiency goal through different mechanisms.

In medical and biomedical natural language processing, domain-adaptive pretraining has repeatedly been shown to improve downstream performance [5], motivating domain-specialized language models trained on biomedical and clinical corpora [6–8]; more recently, large language models have also been shown to encode substantial clinical knowledge [9]. A recurring concern when adapting a general model to a narrow domain is catastrophic forgetting, in which gains on the target domain come at the expense of previously acquired general capability [10,11]. Monitoring out-of-domain performance during adaptation is therefore a useful safeguard.

Tuberculosis remains one of the leading causes of death from a single infectious agent worldwide [12], and Indonesia carries one of the highest TB burdens of any country, which makes locally accessible tools for organizing and surfacing TB knowledge especially valuable. In this study, a manual LoRA implementation previously used on a classification task is re-evaluated on a more challenging generative task: PubMed-literature-based TB clinical question answering. The objective is to determine whether manual LoRA can improve in-domain modeling of TB clinical text while preserving general-language capability, and to characterize how the LoRA rank affects this trade-off.

Equally important to the specific results is the study's intended role as an accessible and fully reproducible entry point for life-science, pharmacy, and clinical researchers who are beginning to explore AI/ML. The pipeline uses a small open base model, an openly available dataset, and hardware within reach of a typical laboratory, and all code and artifacts are released openly. The work is therefore positioned not as a finished clinical solution but as a transparent starting point and reference implementation on which larger, more capable, and more clinically useful systems can subsequently be built.

The main contributions of this study are:
1. Demonstrating an end-to-end manual LoRA pipeline for a clinical generative task on GPT-2.
2. Evaluating the impact of adaptation using both in-domain and out-of-domain metrics to monitor forgetting.
3. Analyzing the effect of LoRA rank on the trade-off between domain specificity and the stability of general-language capability.
4. Providing an accessible, openly released reference implementation intended to lower the entry barrier for biomedical and pharmacy researchers in resource-constrained settings.

## Methods

### Experimental Design

The study compares two conditions: (1) base GPT-2 without LoRA fine-tuning, and (2) GPT-2 with a trained LoRA adapter. A rank ablation is conducted across the focused configurations r = 4, 8, and 16. In this ablation, the number of training epochs was scaled with rank (r4 = 3 epochs, r8 = 5 epochs, r16 = 10 epochs); the implications of this coupled design are discussed in the Limitations subsection.

This study used a public secondary dataset only and did not involve human participants or patient-identifiable data; therefore, informed consent and ethics-committee approval were not applicable. The model outputs are research artifacts and are not intended for clinical use.

### Dataset

The primary source was the PubMedQA collection [13], accessed through the BigBio framework (`bigbio/pubmed_qa`) [14]. The data preparation steps were:
1. Filter records using TB-related keywords (inclusion criterion: presence of TB-related terms in the record; records without such terms were excluded), yielding approximately 707 TB-related QA pairs.
2. Format each record as a causal text pair, "Question: ...\nAnswer: ...".
3. Store a local cache at `data/tb_qa.json`.
4. Split the data into train/validation/test partitions according to the project pipeline.

### Model and Training Configuration

The base model was GPT-2 (`GPT2LMHeadModel`, 124 million parameters) [15]. All base-model parameters were frozen before LoRA injection. LoRA was injected into the fused QKV attention projection (`c_attn`) in each GPT-2 transformer block [16].

The LoRA mechanism decomposes the weight update $W_0 \in \mathbb{R}^{d \times k}$ as:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)$$

so that the forward pass becomes:

$$h = W_0 x + \frac{\alpha}{r} \, BAx$$

Matrix $A$ was initialized with Kaiming uniform initialization [17]; matrix $B$ was initialized with zeros so that $\Delta W = 0$ at the start of training and the initial model behavior was preserved. In this study, $\alpha = r$ (coupled with the rank) so that the effective scale $\alpha/r = 1$ was consistent across all rank values in the ablation.

Core hyperparameters were:
- Optimizer: AdamW [18]
- Learning rate: $2 \times 10^{-4}$
- Batch size: 4
- Max sequence length: 512 tokens
- Warmup: 10% of total training steps
- Rank ablation (focused): $r \in \{4, 8, 16\}$ with epochs $\{3, 5, 10\}$ respectively, $\alpha = r$

### Evaluation Protocol

Quantitative evaluation used two metrics: (1) in-domain perplexity on the TB validation set, and (2) out-of-domain perplexity on WikiText-2 [19] as a control for catastrophic forgetting. Qualitative evaluation used a TB prompt battery (greedy and sampled decoding) and a side-by-side comparison of outputs before and after fine-tuning.

To support reproducibility, the experiment code, processed dataset (`data/tb_qa.json`), and result artifacts (`outputs/20260609-second-ablation-run_focused-ranks/`) are available in the project repository at https://github.com/arisetyo/lora-project-1b (subject to the applicable data-distribution policy).

## Results and Discussion

### Quantitative Results

Table 1 presents the evaluation of the LoRA adapter across the focused rank configurations. As the rank increases, the number of trainable parameters grows linearly, from 147,456 parameters at rank 4 to 589,824 parameters at rank 16.

**Table 1.** In-domain and out-of-domain perplexity across LoRA rank configurations.

| Rank | Epochs | Trainable Parameters | Validation PPL | OOD PPL |
| ---- | ------ | -------------------: | -------------: | ------: |
| 4    | 3      |              147,456 |          30.85 |   55.81 |
| 8    | 5      |              294,912 |          24.38 |   56.78 |
| 16   | 10     |              589,824 |          21.08 |   56.71 |

The results show a consistent downward trend in in-domain perplexity as the LoRA rank increases. Compared to the rank-4 configuration, rank 16 yields a perplexity reduction of approximately 31.7%, indicating an improved ability of the model to model the distribution of the TB text used during training.

On the other hand, the out-of-domain perplexity values do not change significantly and stay within a narrow range (55–57). The stability of this metric indicates that the domain adaptation obtained through LoRA does not cause large degradation in GPT-2's general-language capability. This finding is consistent with the main goal of parameter-efficient fine-tuning, namely improving performance on the target domain without making large modifications to the base model parameters.

Furthermore, the validation loss curves (Figure 1) show a stable, monotonic decrease across all three configurations, with no indication of a rebound (overfitting). For the rank-16 configuration, validation loss dropped from approximately 0.607 at the start of training to 0.517 at the final epoch. The rank-8 and rank-4 configurations plateaued at higher values (≈0.540 and ≈0.586, respectively), consistent with their lower capacity and shorter training schedules. This suggests that the adapter capacity can still be exploited effectively at the dataset size used in this study.

![Validation loss per epoch for the focused LoRA configurations (r4/e3, r8/e5, r16/e10).](../../outputs/wandb_charts/val_loss_by_epoch.png)

**Figure 1.** Validation loss per epoch for the focused ablation runs `ablation_r4_e3`, `ablation_r8_e5`, and `ablation_r16_e10`. Final-epoch values are annotated (0.586, 0.541, 0.517). Source: Weights & Biases project `lora-phase-1b` (`val/loss`); underlying data in `outputs/wandb_charts/wandb_export_2026-06-09T23_09_42.866+07_00.csv`. Plot regenerated via `outputs/wandb_charts/make_figures.py`.

![Final validation perplexity by LoRA rank configuration, with value labels.](../../outputs/wandb_charts/val_perplexity_bars.png)

**Figure 2.** Final validation perplexity for each focused configuration (r16/e10 = 21.08, r8/e5 = 24.38, r4/e3 = 30.85; lower is better). Source: Weights & Biases project `lora-phase-1b` (`val/perplexity`); underlying data in `outputs/wandb_charts/wandb_export_2026-06-09T23_10_06.974+07_00.csv`. Plot regenerated via `outputs/wandb_charts/make_figures.py`.

### Qualitative Results

To complement the perplexity metrics, a fixed battery of ten prompts in the training "Question: ... Answer:" format was run on the base GPT-2 model and on the best adapter (rank 16, epoch 10), using both greedy decoding and sampled decoding (temperature = 0.7, top-p = 0.9, seed = 42). Eight prompts concerned TB clinical content and two were out-of-domain sanity checks. The full side-by-side battery is provided as supplementary material (`docs/base-vs-finetuned.md`); Table 2 reproduces representative greedy completions.

**Table 2.** Representative greedy completions for base GPT-2 and GPT-2 + LoRA (r = 16)

| No. | Prompt (TB clinical question) | Base GPT-2 | GPT-2 + LoRA (r = 16) |
| :--: | --- | --- | --- |
| 1 | Primary mechanisms of *M. tuberculosis* resistance to rifampicin | "…a bacterial infection that is transmitted through contact with contaminated food, water… a common cause of diarrhea, vomiting… pneumonia in children." | "…through the inhibition of the growth factor receptor (GFR) and the growth factor receptor-1 (GFR-1) receptor… mediated by the activation of the growth factor receptor-1 receptor…" |
| 2 | How HIV co-infection affects TB presentation and treatment | "The clinical presentation of tuberculosis is a complex and complex process." (repeated) | "HIV co-infection is associated with a reduced clinical presentation and treatment of tuberculosis. This is a major limitation of the current study." |
| 3 | Standard first-line regimen for drug-sensitive TB | "…the standard first-line regimen for drug-resistant tuberculosis." (repeated) | "…the recommended regimen for drug-resistant tuberculosis." (shorter; terminates early) |

Note: Completions were generated with greedy decoding (maximum 80 new tokens) and are lightly trimmed for length; the complete prompt battery, including sampled decoding, is provided as supplementary material. Abbreviations: GFR, growth factor receptor; HIV, human immunodeficiency virus; LoRA, low-rank adaptation; *M. tuberculosis*, *Mycobacterium tuberculosis*; TB, tuberculosis.

Two patterns are consistent across the battery. First, the adapter shifts the model's lexical register toward the biomedical and PubMed-abstract style of the training corpus: LoRA completions more frequently use domain vocabulary (e.g., "antiretroviral therapy," "bacilli," "*B. tuberculosis*," cytokine terms such as IL-4/IL-6, and abstract-style hedges such as "this is a major limitation of the current study," seen in the HIV co-infection completion (Table 2, row 2)), whereas the base model drifts to generic or unrelated content, such as describing TB as a food- or water-borne cause of diarrhoea and pneumonia (Table 2, row 1). The adapter also tends to produce shorter completions that terminate earlier (Table 2, row 3), consistent with having learned the answer-length distribution of the QA pairs. These observations align with the in-domain perplexity reduction reported in Section 4.1.

Second, neither model is clinically reliable. Improved register did not translate into factual accuracy: neither model named the actual first-line regimen (rifampicin, isoniazid, pyrazinamide, ethambutol; Table 2, row 3), and the LoRA model's mechanistic explanations (e.g., attributing rifampicin resistance to growth-factor-receptor or interleukin pathways rather than *rpoB* mutations; Table 2, row 1) are confidently stated but incorrect. The out-of-domain sanity prompts behaved as expected for a 124M model: both versions answered the boiling-point and authorship questions incorrectly, with no improvement from fine-tuning. Taken together, the qualitative analysis supports the central claim that LoRA improves in-domain *modeling* of TB text while leaving general capability largely unchanged, but it also shows that perplexity gains at this model scale do not yet yield trustworthy clinical answers — directly motivating the human-evaluation protocol outlined in the Limitations and Future Directions.

### Discussion

The results show that manual LoRA can adapt GPT-2 to the tuberculosis domain even though it uses only a small number of additional parameters. The consistent decrease in in-domain perplexity at higher ranks shows that the adapter successfully captured the linguistic patterns and terminology present in the TB corpus derived from PubMedQA.

Interestingly, the improvement in in-domain performance was not accompanied by a meaningful change in out-of-domain perplexity. This finding indicates that the base model's general knowledge is largely preserved, so that the risk of catastrophic forgetting in the tested configurations is relatively low. LoRA therefore provides an efficient adaptation mechanism for the medical domain without requiring a full update of all model parameters.

From a methodological standpoint, this study also shows that a manual LoRA implementation can produce behavior consistent with reports in the parameter-efficient fine-tuning literature. Despite using a relatively small GPT-2 (124 million parameters) and a dataset of approximately 707 question-answer pairs, the adapter was still able to produce measurable improvements on the domain validation metric. The strongest scientific claim supported by these results is that manual LoRA adaptation improves in-domain modeling performance on TB-related QA data while maintaining relatively stable out-of-domain perplexity.

An important direction for strengthening this work is the preparation of a standardized inference server for human evaluation. Such a server would enable consistent A/B comparison between the base GPT-2 model and the best GPT-2 + LoRA adapter model using the same endpoint, decoding parameters, and logging. With this approach, evaluation would not rely solely on perplexity but could also incorporate answer-quality assessment by human raters (e.g., clinicians or biomedical researchers) in a more structured manner. Beyond strengthening external validity, an inference-server design opens opportunities for cross-institutional collaboration, in which partners access a uniform interface for rubric-based annotation (terminology accuracy, regimen completeness, hallucination potential, and clinical usefulness), making cross-team results easier to compare and replicate.

From an applied perspective, the value of this work lies in its role as a foundation rather than an endpoint. The same parameter-efficient recipe demonstrated here on GPT-2 can, in principle, be applied to modern medical foundation models (for example, MedGemma [20]) and trained on substantially larger and more representative corpora — including, subject to appropriate data-governance and privacy safeguards, national health datasets such as those held by Indonesian public-health and insurance systems. With stronger base models and better data, the resulting systems could eventually support frontline services — for instance, knowledge access or decision support for primary-care clinics (Puskesmas) and community pharmacies (apotek) — in being better prepared against TB. The TB use case also serves as a template: the same approach is relevant to other high-burden conditions in Indonesia, such as respiratory disease associated with high smoking prevalence and diabetes. In this sense, the present study is intended as an early spark for a broader programme of locally grounded, AI-assisted research on TB and related health problems, rather than as a deployable tool in its own right.

### Limitations

This study has several limitations:
1. The main evaluation is still dominated by automatic metrics (in-domain and out-of-domain perplexity), so it does not yet fully capture clinical factual quality.
2. In the reported ablation, rank and the number of epochs were varied simultaneously (r4 = 3 epochs, r8 = 5 epochs, r16 = 10 epochs). Consequently, the observed in-domain improvement cannot be attributed to rank alone, as it may partly stem from longer training. A controlled experiment that fixes the number of epochs across ranks is needed to disentangle these two factors.
3. No dedicated inference server is yet available to conduct controlled, blinded human evaluation between the base and LoRA models.
4. Inter-rater reliability has not been measured because the human evaluation protocol has not been run. If more than two raters are used, the appropriate metrics are Fleiss' kappa [21] or Krippendorff's alpha [22], not Cohen's kappa.
5. The study is small in scale, with a single base architecture (GPT-2, 124M parameters), a modest dataset (~707 QA pairs), and a limited hyperparameter search space.
6. Training uses the entire token sequence as the loss target without question-token masking; this technique is planned for the next phase and may improve answer-generation quality.

### Future Directions

Future work will focus on building a research inference server to support methodologically stronger human evaluation, including: (1) providing two standardized inference modes (base GPT-2 and GPT-2 + best LoRA adapter); (2) running blinded A/B studies with the same TB prompts, fixed decoding parameters, and controlled seeds; (3) developing an evaluation rubric together with clinical/academic partners (medical accuracy, specificity, readability, and hallucination risk); (4) computing inter-rater reliability metrics (e.g., Fleiss' kappa [21] or Krippendorff's alpha [22]); (5) building a human-evaluation benchmark of 50–100 unseen TB questions comparing base GPT-2 against GPT-2 + LoRA r16, since in biomedical NLP reviewers generally find actual answer quality more compelling than perplexity alone; (6) conducting a controlled ablation that fixes epochs across ranks to isolate the effect of rank; and (7) using the inference server as a multi-institution collaboration platform to broaden annotation coverage and external validation. This approach is expected to strengthen the project's scientific contribution on the evaluation side while opening opportunities for formal collaboration with teaching hospitals, medical faculties, or health NLP laboratories.

## Conclusion

Manual LoRA proved viable as a domain-adaptation approach for a generative language model on the tuberculosis question-answering task. Consistent with the study objective, the rank-16 configuration provided the best performance in this experiment, with a validation perplexity of 21.08 and a validation loss of approximately 0.517, while out-of-domain perplexity remained stable. These results support the hypothesis that increasing adapter capacity can improve the model's domain capability without significantly sacrificing general-language ability, provided that model evaluation also considers out-of-domain metrics so that the risk of catastrophic forgetting remains monitored. Follow-up research should involve clinical-expert evaluation, additional evaluation metrics, comparison with standard PEFT implementations (e.g., the HuggingFace PEFT library [23]), a controlled rank-vs-epoch ablation, and experiments on larger models, including memory-efficient fine-tuning of quantized models [24]. More broadly, this work is offered as an accessible and reproducible starting point intended to encourage further locally grounded AI research on TB and other high-burden health problems in Indonesia.

## Acknowledgements

This work was conducted as part of PT Galenic Systems Indonesia's research and development efforts in applying artificial intelligence (AI) and machine learning (ML) to the life sciences. The author thanks Galenic for the computational resources and research support that made this study possible.

## Funding

None.

## Conflict of Interest

The author is affiliated with PT Galenic Systems Indonesia, which is in preliminary discussions with the Faculty of Pharmacy, Universitas Padjadjaran, regarding a potential collaboration on life-science AI research, including a proposed AI Center of Excellence. No formal agreement has been signed, and this relationship had no role in the design, execution, or reporting of the study. The author declares no other conflicts of interest.

## References

<!--
PCPR reference style: Vancouver, numbered by order of appearance in the text.
NOTE TO AUTHOR: please verify every entry (authors, year, volume, pages, DOI/URL) against the
original source before submission. Volume/page details below should be double-checked.
Recency: ≥80% are primary literature from the last 10 years; older entries (#10, #17, #21, #22)
are foundational methodological references.
-->

1. Ding N, Qin Y, Yang G, Wei Z, Yang Z, Su Y, et al. Parameter-efficient fine-tuning of large-scale pre-trained language models. Nat Mach Intell. 2023;5(3):220-235.
2. Hu EJ, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, et al. LoRA: low-rank adaptation of large language models. In: Proceedings of the International Conference on Learning Representations (ICLR); 2022.
3. Houlsby N, Giurgiu A, Jastrzebski S, Morrone B, de Laroussilhe Q, Gesmundo A, et al. Parameter-efficient transfer learning for NLP. In: Proceedings of the 36th International Conference on Machine Learning (ICML); 2019. p. 2790-2799.
4. Lester B, Al-Rfou R, Constant N. The power of scale for parameter-efficient prompt tuning. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP); 2021. p. 3045-3059.
5. Gururangan S, Marasović A, Swayamdipta S, Lo K, Beltagy I, Downey D, et al. Don't stop pretraining: adapt language models to domains and tasks. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL); 2020. p. 8342-8360.
6. Lee J, Yoon W, Kim S, Kim D, Kim S, So CH, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics. 2020;36(4):1234-1240.
7. Gu Y, Tinn R, Cheng H, Lucas M, Usuyama N, Liu X, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Trans Comput Healthc. 2021;3(1):1-23.
8. Alsentzer E, Murphy JR, Boag W, Weng WH, Jin D, Naumann T, et al. Publicly available clinical BERT embeddings. In: Proceedings of the 2nd Clinical Natural Language Processing Workshop (NAACL); 2019. p. 72-78.
9. Singhal K, Azizi S, Tu T, Mahdavi SS, Wei J, Chung HW, et al. Large language models encode clinical knowledge. Nature. 2023;620(7972):172-180.
10. French RM. Catastrophic forgetting in connectionist networks. Trends Cogn Sci. 1999;3(4):128-135.
11. Kirkpatrick J, Pascanu R, Rabinowitz N, Veness J, Desjardins G, Rusu AA, et al. Overcoming catastrophic forgetting in neural networks. Proc Natl Acad Sci U S A. 2017;114(13):3521-3526.
12. World Health Organization. Global tuberculosis report 2023. Geneva: World Health Organization; 2023.
13. Jin Q, Dhingra B, Liu Z, Cohen WW, Lu X. PubMedQA: a dataset for biomedical research question answering. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); 2019. p. 2567-2577.
14. Fries JA, Weber L, Seelam N, Altay G, Datta D, Garda S, et al. BigBio: a framework for data-centric biomedical natural language processing. In: Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track; 2022.
15. Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I. Language models are unsupervised multitask learners. OpenAI Technical Report; 2019.
16. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. In: Advances in Neural Information Processing Systems (NeurIPS); 2017. p. 5998-6008.
17. He K, Zhang X, Ren S, Sun J. Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV); 2015. p. 1026-1034.
18. Loshchilov I, Hutter F. Decoupled weight decay regularization. In: Proceedings of the International Conference on Learning Representations (ICLR); 2019.
19. Merity S, Xiong C, Bradbury J, Socher R. Pointer sentinel mixture models. In: Proceedings of the International Conference on Learning Representations (ICLR); 2017.
20. Sellergren A, Kazemzadeh S, Jaroensri T, Kiraly A, Traverse M, Kohlberger T, et al. MedGemma technical report. arXiv preprint arXiv:2507.05201; 2025.
21. Fleiss JL. Measuring nominal scale agreement among many raters. Psychol Bull. 1971;76(5):378-382.
22. Krippendorff K. Content analysis: an introduction to its methodology. 2nd ed. Thousand Oaks (CA): Sage; 2004.
23. Mangrulkar S, Gugger S, Debut L, Belkada Y, Paul S, Bossan B. PEFT: state-of-the-art parameter-efficient fine-tuning methods [Internet]. Hugging Face; 2022 [cited 2026 Jun 9]. Available from: https://github.com/huggingface/peft
24. Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: efficient finetuning of quantized LLMs. In: Advances in Neural Information Processing Systems (NeurIPS); 2023.
