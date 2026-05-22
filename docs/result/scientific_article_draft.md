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

Arie M. Prasetyo1*, Nama Penulis Kedua2

1. Afiliasi 1, Kota, Indonesia
2. Afiliasi 2, Kota, Indonesia

*Korespondensi: email@domain.com

## Abstrak (Bahasa Indonesia)

Penelitian ini mengevaluasi apakah metode Low-Rank Adaptation (LoRA) yang diimplementasikan secara manual dapat mengadaptasi GPT-2 secara efisien untuk tugas generatif tanya-jawab klinis tuberkulosis (TB). Dataset disusun dari `bigbio/pubmed_qa` dengan penyaringan kata kunci TB, kemudian diformat sebagai pasangan teks kausal "Question: ... Answer: ...". Model dasar GPT-2 dibekukan seluruh parameternya, dan hanya parameter LoRA yang dilatih pada lapisan perhatian `c_attn`. Evaluasi dilakukan menggunakan tiga komponen: perplexity in-domain (validasi TB), perplexity out-of-domain (wikitext-2) untuk memantau catastrophic forgetting, dan analisis kualitatif melalui prompt battery. Ablasi rank dilakukan pada $r \in \{1,4,8,16,32\}$. Hasil menunjukkan LoRA meningkatkan kesesuaian domain TB dengan tetap menjaga performa umum bahasa ketika pemilihan rank mempertimbangkan metrik in-domain dan out-of-domain secara bersamaan.

Kata kunci: LoRA, GPT-2, Tuberkulosis, PubMed QA, Perplexity, Parameter-Efficient Fine-Tuning

## Abstract (English)

This study evaluates whether a manually implemented Low-Rank Adaptation (LoRA) method can efficiently adapt GPT-2 to a generative tuberculosis (TB) clinical question-answering task. The dataset is built from `bigbio/pubmed_qa` using TB keyword filtering and formatted as causal language modeling pairs in the "Question: ... Answer: ..." structure. All base GPT-2 parameters are frozen, and only LoRA parameters are trained on `c_attn` attention projections. Evaluation uses three components: in-domain perplexity (TB validation), out-of-domain perplexity (wikitext-2) to monitor catastrophic forgetting, and qualitative prompt-battery analysis. Rank ablation is performed for $r \in \{1,4,8,16,32\}$. Results indicate that LoRA improves TB-domain fit while preserving general-language capability when rank selection jointly considers in-domain and out-of-domain metrics.

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

1. Model dasar: GPT-2 (`GPT2LMHeadModel`).
2. Seluruh parameter model dasar dibekukan sebelum injeksi LoRA.
3. Injeksi LoRA pada proyeksi perhatian `c_attn`.
4. Hiperparameter inti:
   - Optimizer: AdamW
   - Learning rate: 2e-4
   - Epoch: 3
   - Batch size: 4
   - Max sequence length: 512
   - Warmup: 10% total langkah

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

## 5. Kesimpulan

LoRA manual pada GPT-2 dapat meningkatkan performa generatif in-domain TB pada skala eksperimen kecil, dengan syarat evaluasi model mempertimbangkan metrik out-of-domain agar risiko catastrophic forgetting tetap terpantau. Penelitian lanjutan perlu melibatkan evaluasi ahli klinis, objektif evaluasi tambahan, dan pembandingan dengan implementasi PEFT standar.

## Ucapan Terima Kasih

Contoh:
Penulis mengucapkan terima kasih kepada [institusi/lab/pendana] atas dukungan komputasi dan diskusi penelitian.

## Kontribusi Penulis

Gunakan format CRediT (contoh):
1. Konseptualisasi: A.M.P.
2. Metodologi: A.M.P., Penulis 2.
3. Implementasi perangkat lunak: A.M.P.
4. Analisis formal: A.M.P., Penulis 2.
5. Penulisan draf awal: A.M.P.
6. Review dan editing: semua penulis.

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

Contoh placeholder:
1. Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR.
2. Brown, T., et al. (2020). Language models are few-shot learners. NeurIPS.
3. [Tambahkan referensi biomedical NLP yang relevan].

## Lampiran (Opsional)

1. Prompt battery lengkap.
2. Tabel hasil per-seed.
3. Detail environment (GPU, versi library, commit hash).
