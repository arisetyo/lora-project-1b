# Draft Artikel Ilmiah (Format Umum Jurnal Terindeks SINTA)

Catatan penting:
- SINTA adalah sistem indeks jurnal, bukan satu gaya selingkung tunggal.
- Draft ini mengikuti format umum yang paling sering dipakai jurnal SINTA (IMRaD + abstrak bilingual + metadata etik/pendanaan/kontribusi).
- Sebelum submit, sesuaikan lagi dengan template jurnal target (mis. ukuran font, gaya sitasi APA/IEEE/Vancouver, jumlah kata abstrak, aturan tabel/gambar).

## Judul (Bahasa Indonesia)

Evaluasi Adaptasi Manual LoRA pada GPT-2 untuk Generasi Tanya-Jawab Klinis Tuberkulosis Berbasis PubMed

## Title (English)

Evaluation of Manual LoRA Adaptation on GPT-2 for PubMed-Based Tuberculosis Clinical Q&A Generation

## Penulis dan Afiliasi

M. Arie Prasetyo1*

1. PT Galenic Systems Indonesia, Bandung, Indonesia

*Korespondensi: arie@galenic.systems

## Abstrak (Bahasa Indonesia)

Penelitian ini mengevaluasi apakah metode Low-Rank Adaptation (LoRA) yang diimplementasikan secara manual dapat mengadaptasi GPT-2 secara efisien untuk tugas generatif tanya-jawab klinis tuberkulosis (TB). Dataset disusun dari `bigbio/pubmed_qa` dengan penyaringan kata kunci TB, kemudian diformat sebagai pasangan teks kausal "Question: ... Answer: ...". Model dasar GPT-2 dibekukan seluruh parameternya, dan hanya parameter LoRA yang dilatih pada lapisan perhatian `c_attn`. Evaluasi dilakukan menggunakan tiga komponen: perplexity in-domain (validasi TB), perplexity out-of-domain (wikitext-2) untuk memantau catastrophic forgetting, dan analisis kualitatif melalui prompt battery. Ablasi rank dilakukan pada $r \in \{1,4,8,16,32\}$. [DRAFT — diperbarui setelah eksperimen selesai] Secara metodologis, penelitian ini mengharapkan bahwa LoRA dapat meningkatkan kesesuaian domain TB dengan tetap menjaga performa umum bahasa ketika pemilihan rank mempertimbangkan metrik in-domain dan out-of-domain secara bersamaan.

Kata kunci: LoRA, GPT-2, Tuberkulosis, PubMed QA, Perplexity, Parameter-Efficient Fine-Tuning

## Abstract (English)

This study evaluates whether a manually implemented Low-Rank Adaptation (LoRA) method can efficiently adapt GPT-2 to a generative tuberculosis (TB) clinical question-answering task. The dataset is built from `bigbio/pubmed_qa` using TB keyword filtering and formatted as causal language modeling pairs in the "Question: ... Answer: ..." structure. All base GPT-2 parameters are frozen, and only LoRA parameters are trained on `c_attn` attention projections. Evaluation uses three components: in-domain perplexity (TB validation), out-of-domain perplexity (wikitext-2) to monitor catastrophic forgetting, and qualitative prompt-battery analysis. Rank ablation is performed for $r \in \{1,4,8,16,32\}$. [DRAFT — update after experiments complete] Methodologically, the study expects that LoRA can improve TB-domain fit while preserving general-language capability when rank selection jointly considers in-domain and out-of-domain metrics.

Keywords: LoRA, GPT-2, Tuberculosis, PubMed QA, Perplexity, Parameter-Efficient Fine-Tuning

## 1. Pendahuluan

Adaptasi model bahasa untuk domain medis membutuhkan efisiensi parameter karena keterbatasan komputasi dan kebutuhan eksperimen berulang. Pendekatan LoRA memungkinkan fine-tuning dengan menambahkan matriks ber-rank rendah tanpa memperbarui seluruh bobot model dasar. Pada penelitian ini, implementasi LoRA manual yang sebelumnya digunakan pada tugas klasifikasi dievaluasi kembali pada tugas generatif yang lebih menantang, yaitu tanya-jawab klinis TB berbasis literatur PubMed.

Kontribusi utama penelitian:
1. Menunjukkan pipeline end-to-end LoRA manual untuk tugas generatif klinis pada GPT-2.
2. Mengevaluasi dampak adaptasi menggunakan metrik in-domain dan out-of-domain untuk memantau forgetting.
3. Menganalisis pengaruh rank LoRA terhadap trade-off spesifisitas domain dan stabilitas kemampuan umum.

## 2. Tinjauan Pustaka

Uraikan ringkas:
1. Parameter-Efficient Fine-Tuning dan LoRA.
2. Domain adaptation pada NLP medis.
3. Evaluasi model generatif berbasis perplexity dan analisis kualitatif.

Tambahkan sitasi sesuai gaya jurnal target.

## 3. Metode Penelitian

### 3.1 Desain Eksperimen

Penelitian membandingkan dua kondisi:
1. GPT-2 dasar (tanpa fine-tuning LoRA).
2. GPT-2 + adapter LoRA terlatih.

Eksperimen tambahan dilakukan melalui ablation rank: $r = 1,4,8,16,32$.

### 3.2 Dataset

Sumber utama: `bigbio/pubmed_qa`.

Langkah data:
1. Filter rekaman berbasis kata kunci TB.
2. Format data menjadi "Question: ...\nAnswer: ...".
3. Simpan cache lokal di `data/tb_qa.json`.
4. Bagi data menjadi train/validation/test sesuai pipeline proyek.

### 3.3 Model dan Konfigurasi Pelatihan

1. Model dasar: GPT-2 (`GPT2LMHeadModel`, 124 juta parameter).
2. Seluruh parameter model dasar dibekukan sebelum injeksi LoRA.
3. Injeksi LoRA pada proyeksi perhatian gabungan QKV (`c_attn`) di setiap blok transformer GPT-2.

Mekanisme LoRA medekomposisi pembaruan bobot $W_0 \in \mathbb{R}^{d \times k}$ menjadi:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)$$

Sehingga forward pass menjadi:

$$h = W_0 x + \frac{\alpha}{r} \, BAx$$

Matriks $A$ diinisialisasi dengan Kaiming uniform; matriks $B$ diinisialisasi dengan nol sehingga $\Delta W = 0$ pada awal pelatihan dan perilaku model awal tetap terjaga. Dalam penelitian ini, $\alpha = r$ (dikopel dengan rank) sehingga skala efektif $\alpha/r = 1$ konsisten di seluruh nilai rank pada ablasi.

4. Hiperparameter inti:
   - Optimizer: AdamW
   - Learning rate: $2 \times 10^{-4}$
   - Epoch: 3
   - Batch size: 4
   - Max sequence length: 512 token
   - Warmup: 10% dari total langkah pelatihan
   - Rank ablation: $r \in \{1, 4, 8, 16, 32\}$, $\alpha = r$

### 3.4 Protokol Evaluasi

Evaluasi kuantitatif:
1. Perplexity in-domain pada validasi TB.
2. Perplexity out-of-domain pada wikitext-2 sebagai kontrol catastrophic forgetting.

Evaluasi kualitatif:
1. Prompt battery TB (greedy dan sampled decoding).
2. Perbandingan side-by-side sebelum dan sesudah fine-tuning.

## 4. Hasil dan Pembahasan

### 4.1 Hasil Kuantitatif

Isi tabel berikut menggunakan hasil eksperimen:

| Model | Rank | In-domain PPL | OOD PPL | Delta In-domain (%) | Delta OOD (%) |
|---|---:|---:|---:|---:|---:|
| GPT-2 dasar | - | TODO | TODO | - | - |
| GPT-2 + LoRA | 1 | TODO | TODO | TODO | TODO |
| GPT-2 + LoRA | 4 | TODO | TODO | TODO | TODO |
| GPT-2 + LoRA | 8 | TODO | TODO | TODO | TODO |
| GPT-2 + LoRA | 16 | TODO | TODO | TODO | TODO |
| GPT-2 + LoRA | 32 | TODO | TODO | TODO | TODO |

Interpretasi minimum yang harus dibahas:
1. Besaran penurunan in-domain PPL.
2. Stabilitas OOD PPL sebagai indikator forgetting.
3. Rank terbaik berdasarkan gabungan kedua metrik.

### 4.2 Hasil Kualitatif

Sajikan 5-10 contoh prompt TB berikut:
1. Output model dasar (greedy + sampled).
2. Output model LoRA terbaik (greedy + sampled).
3. Catatan analisis: spesifisitas istilah klinis, kelengkapan regimen, potensi halusinasi.

### 4.3 Pembahasan

Bahas:
1. Apakah LoRA manual cukup untuk meningkatkan kompetensi domain TB.
2. Dampak pemilihan rank terhadap kualitas domain vs retensi kemampuan umum.
3. Implikasi metodologis untuk eksperimen fase berikutnya.

Topik penting untuk diskusi lanjutan adalah penyiapan inference server terstandar untuk evaluasi manusia. Dalam konteks penelitian ini, server inferensi memungkinkan perbandingan A/B yang konsisten antara model dasar GPT-2 dan model GPT-2 + adapter LoRA terbaik menggunakan endpoint, parameter decoding, dan logging yang sama. Dengan pendekatan ini, evaluasi tidak hanya bertumpu pada perplexity, tetapi juga dapat memasukkan penilaian kualitas jawaban oleh penilai manusia (misalnya klinisi atau peneliti biomedis) secara lebih terstruktur.

Selain memperkuat validitas eksternal, desain server inferensi membuka peluang kolaborasi lintas institusi. Mitra kolaborasi dapat mengakses antarmuka evaluasi yang seragam untuk melakukan anotasi berbasis rubric (akurasi terminologi, kelengkapan regimen, potensi halusinasi, dan kegunaan klinis), sehingga hasil antar-tim lebih mudah dibandingkan dan direplikasi.

## 5. Keterbatasan Penelitian

Penelitian ini memiliki beberapa keterbatasan:
1. Evaluasi utama masih didominasi metrik otomatis (in-domain dan out-of-domain perplexity), sehingga belum sepenuhnya menangkap kualitas faktual klinis.
2. Belum tersedia inference server khusus untuk menyelenggarakan evaluasi manusia terblind secara terkontrol antara model dasar dan model LoRA.
3. Belum dilakukan pengukuran reliabilitas antar-penilai (inter-rater reliability) karena protokol evaluasi manusia belum dijalankan. Bila lebih dari dua penilai digunakan, metrik yang sesuai adalah Fleiss' kappa atau Krippendorff's alpha, bukan Cohen's kappa.
4. Studi masih berskala kecil dengan satu arsitektur dasar (GPT-2, 124M parameter) dan ruang eksplorasi hiperparameter terbatas.
5. Pelatihan menggunakan seluruh token sequence sebagai target loss tanpa masking token pertanyaan (question-token masking); teknik ini direncanakan pada fase berikutnya dan dapat meningkatkan kualitas generasi jawaban.

## 6. Kesimpulan

[DRAFT — diperbarui setelah eksperimen selesai] LoRA manual pada GPT-2 diharapkan dapat meningkatkan performa generatif in-domain TB pada skala eksperimen kecil, dengan syarat evaluasi model mempertimbangkan metrik out-of-domain agar risiko catastrophic forgetting tetap terpantau. Penelitian lanjutan perlu melibatkan evaluasi ahli klinis, metrik evaluasi tambahan, pembandingan dengan implementasi PEFT standar (mis. HuggingFace PEFT), dan ekperimen pada model yang lebih besar.

## 7. Arah Pengembangan Lanjutan (Future Work)

Rencana pengembangan berikutnya berfokus pada pembangunan inference server penelitian untuk mendukung evaluasi manusia yang lebih kuat secara metodologis:
1. Menyediakan dua mode inferensi terstandar: GPT-2 dasar dan GPT-2 + adapter LoRA terbaik.
2. Menjalankan studi A/B terblind dengan prompt TB yang sama, parameter decoding tetap, dan seed terkontrol.
3. Menyusun rubric penilaian bersama mitra klinis/akademik (akurasi medis, spesifisitas, keterbacaan, dan risiko halusinasi).
4. Menghitung metrik reliabilitas antar-penilai (misalnya Cohen's kappa) untuk meningkatkan kredibilitas hasil.
5. Menjadikan server inferensi sebagai platform kolaborasi multi-institusi untuk memperluas cakupan anotasi dan validasi eksternal.

Pendekatan ini diharapkan memperkuat kontribusi ilmiah proyek dari sisi evaluasi, sekaligus membuka peluang kolaborasi formal dengan rumah sakit pendidikan, fakultas kedokteran, atau laboratorium NLP kesehatan.

## Ucapan Terima Kasih

Contoh:
Penulis mengucapkan terima kasih kepada [institusi/lab/pendana] atas dukungan komputasi dan diskusi penelitian.

## Kontribusi Penulis

Gunakan format CRediT (contoh):
1. Konseptualisasi: A.M.P.
2. Metodologi: A.M.P.
3. Implementasi perangkat lunak: A.M.P.
4. Analisis formal: A.M.P.
5. Penulisan draf awal: A.M.P.
6. Review dan editing: A.M.P.

> Tambahkan penulis lain beserta kontribusinya jika ada kolaborator di kemudian hari.

## Pernyataan Etik

Penelitian ini menggunakan dataset publik sekunder dan tidak melibatkan intervensi pada subjek manusia. Luaran model tidak ditujukan untuk penggunaan klinis.

## Pendanaan

Isi salah satu:
1. Penelitian ini tidak menerima pendanaan khusus.
2. Penelitian ini didanai oleh [nama skema], nomor kontrak [xxx].

## Konflik Kepentingan

Penulis menyatakan tidak ada konflik kepentingan.

## Ketersediaan Data dan Kode

1. Kode eksperimen: repository proyek ini.
2. Dataset olahan: `data/tb_qa.json` (sesuai kebijakan distribusi data).
3. Artefak hasil: `outputs/results.csv`, `outputs/checkpoints/`.

## Daftar Pustaka

Gunakan gaya sitasi sesuai jurnal target (umum di SINTA: APA 7, IEEE, atau Vancouver).

Referensi wajib (sudah terverifikasi):
1. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*.
2. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8). [Referensi utama GPT-2]
3. Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2019). PubMedQA: A dataset for biomedical research question answering. In *Proceedings of EMNLP-IJCNLP*. [Referensi dataset PubMed QA]
4. Fries, J. A., et al. (2022). BigBio: A framework for data-centric biomedical natural language processing. In *Advances in Neural Information Processing Systems (NeurIPS)*. [Referensi HuggingFace bigbio/pubmed_qa]
5. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *ICLR*. [Referensi AdamW]
6. Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). Pointer sentinel mixture models. In *ICLR*. [Referensi wikitext-2]

Referensi perlu ditambahkan:
- [Tambahkan referensi domain adaptation pada NLP medis]
- [Tambahkan referensi catastrophic forgetting pada LM]
- [Tambahkan referensi PEFT pembanding, mis. Lester et al. 2021 (prompt tuning), Houlsby et al. 2019 (adapter)]
- [Tambahkan referensi evaluasi inter-rater (Fleiss 1971, Krippendorff 2004) jika studi human eval dilanjutkan]

## Lampiran (Opsional)

1. Prompt battery lengkap.
2. Tabel hasil per-seed.
3. Detail environment (GPU, versi library, commit hash).
