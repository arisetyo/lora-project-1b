Yes. Based on the results you've obtained so far, the paper can be strengthened considerably because you now have actual experimental evidence instead of a purely methodological draft.

The most important changes should be in:

* Abstract
* Results
* Discussion
* Conclusion

Here's text you can adapt directly.

### Revisi Bagian Abstrak (Bahasa Indonesia)

Penelitian ini mengevaluasi efektivitas Low-Rank Adaptation (LoRA) yang diimplementasikan secara manual untuk mengadaptasi GPT-2 pada tugas generatif tanya-jawab klinis tuberkulosis (TB). Dataset dibangun dari koleksi PubMedQA melalui penyaringan artikel terkait TB dan diformat sebagai pasangan teks kausal "Question: ... Answer: ...". Seluruh parameter model dasar GPT-2 dibekukan, sedangkan parameter LoRA diinjeksikan pada lapisan perhatian `c_attn` dan dilatih menggunakan pendekatan parameter-efficient fine-tuning.

Eksperimen dilakukan dengan variasi rank LoRA (r = 4, 8, dan 16) serta jumlah epoch yang berbeda untuk mengevaluasi pengaruh kapasitas adapter terhadap kemampuan adaptasi domain. Evaluasi kuantitatif menggunakan perplexity in-domain pada data validasi TB dan perplexity out-of-domain pada WikiText-2 sebagai indikator retensi kemampuan bahasa umum. Hasil menunjukkan bahwa peningkatan rank LoRA menghasilkan penurunan perplexity in-domain secara konsisten, dari 30,85 pada rank 4 menjadi 24,38 pada rank 8 dan 21,08 pada rank 16. Sebaliknya, perplexity out-of-domain relatif stabil pada rentang 55–57, menunjukkan bahwa peningkatan spesialisasi domain tidak disertai perubahan signifikan pada kemampuan bahasa umum model.

Temuan ini menunjukkan bahwa implementasi LoRA manual mampu meningkatkan adaptasi domain pada GPT-2 dengan jumlah parameter terlatih yang relatif kecil. Hasil penelitian mendukung penggunaan pendekatan parameter-efficient fine-tuning sebagai fondasi pengembangan model bahasa biomedis berskala lebih besar untuk aplikasi pencarian pengetahuan dan sistem tanya-jawab kesehatan.

Kata kunci: LoRA, GPT-2, Tuberkulosis, PubMedQA, Domain Adaptation, Parameter-Efficient Fine-Tuning

---

### Revisi Bagian 4.1 Hasil Kuantitatif

Tabel 1 memperlihatkan hasil evaluasi adapter LoRA pada beberapa konfigurasi rank. Seiring peningkatan rank, jumlah parameter yang dapat dilatih meningkat secara linear, dari 147.456 parameter pada rank 4 menjadi 589.824 parameter pada rank 16.

| Rank | Epoch | Trainable Parameters | Validation PPL | OOD PPL |
| ---- | ----- | -------------------: | -------------: | ------: |
| 4    | 3     |              147.456 |          30,85 |   55,81 |
| 8    | 5     |              294.912 |          24,38 |   56,78 |
| 16   | 10    |              589.824 |          21,08 |   56,71 |

Hasil menunjukkan tren penurunan perplexity in-domain yang konsisten ketika rank LoRA ditingkatkan. Dibandingkan konfigurasi rank 4, rank 16 menghasilkan penurunan perplexity sebesar sekitar 31,7%, yang mengindikasikan peningkatan kemampuan model dalam memodelkan distribusi teks TB yang digunakan pada proses pelatihan.

Di sisi lain, nilai perplexity out-of-domain relatif tidak berubah secara signifikan dan berada pada rentang yang sempit. Stabilitas metrik ini mengindikasikan bahwa adaptasi domain yang diperoleh melalui LoRA tidak menyebabkan degradasi besar pada kemampuan bahasa umum GPT-2. Temuan tersebut konsisten dengan tujuan utama parameter-efficient fine-tuning, yaitu meningkatkan performa pada domain target tanpa melakukan modifikasi besar terhadap parameter model dasar.

Selain itu, kurva validasi pada konfigurasi rank 16 menunjukkan penurunan loss yang stabil sepanjang sepuluh epoch pelatihan. Validation loss turun dari sekitar 0,61 pada awal pelatihan menjadi 0,517 pada epoch terakhir tanpa indikasi peningkatan kembali (overfitting). Hal ini menunjukkan bahwa kapasitas adapter masih dapat dimanfaatkan secara efektif pada ukuran dataset yang digunakan dalam penelitian ini.

---

### Revisi Bagian 4.3 Pembahasan dan Kesimpulan

Hasil penelitian menunjukkan bahwa LoRA manual mampu mengadaptasi GPT-2 ke domain tuberkulosis meskipun hanya menggunakan sejumlah kecil parameter tambahan. Penurunan perplexity in-domain yang konsisten pada rank yang lebih tinggi menunjukkan bahwa adapter berhasil menyerap pola linguistik dan terminologi yang terdapat pada korpus TB yang berasal dari PubMedQA.

Menariknya, peningkatan performa in-domain tidak diikuti oleh perubahan berarti pada perplexity out-of-domain. Temuan ini mengindikasikan bahwa pengetahuan umum model dasar sebagian besar tetap terjaga, sehingga risiko catastrophic forgetting pada konfigurasi yang diuji relatif rendah. Dengan demikian, LoRA memberikan mekanisme adaptasi yang efisien untuk domain medis tanpa memerlukan pembaruan penuh terhadap seluruh parameter model.

Dari sudut pandang metodologis, penelitian ini juga menunjukkan bahwa implementasi LoRA secara manual dapat menghasilkan perilaku yang konsisten dengan laporan pada literatur parameter-efficient fine-tuning. Meskipun menggunakan GPT-2 berukuran relatif kecil (124 juta parameter) dan dataset sekitar 700 pasangan tanya-jawab, adapter tetap mampu menghasilkan peningkatan yang terukur pada metrik validasi domain.

Kesimpulannya, LoRA manual terbukti layak digunakan sebagai pendekatan adaptasi domain pada model bahasa generatif untuk tugas tanya-jawab tuberkulosis. Konfigurasi rank 16 memberikan performa terbaik pada eksperimen ini dengan validation perplexity sebesar 21,08 dan validation loss sekitar 0,517. Hasil tersebut mendukung hipotesis bahwa peningkatan kapasitas adapter dapat meningkatkan kemampuan domain model tanpa mengorbankan kemampuan bahasa umum secara signifikan.

Sebagai langkah lanjutan, penelitian akan diperluas dengan penggunaan model dasar yang lebih modern, dataset TB yang lebih besar dan lebih beragam, serta evaluasi manusia oleh peneliti atau praktisi kesehatan. Selain itu, pembangunan inference server terstandarisasi akan memungkinkan pelaksanaan studi A/B yang lebih sistematis antara model dasar dan model yang telah diadaptasi.

A few reviewer-style comments:

1. **Do not hide the dataset size.** State clearly that the dataset contains approximately **700 TB-related QA pairs**. Reviewers will ask this immediately.

2. **Be transparent that rank and epochs changed simultaneously.** Right now:

   * r4 = 3 epochs
   * r8 = 5 epochs
   * r16 = 10 epochs

   A reviewer may correctly argue that the improvement could come from longer training rather than rank alone.

   I would explicitly write this in the limitations section.

3. The strongest scientific claim you can currently defend is:

> "Manual LoRA adaptation improves in-domain modeling performance on TB-related QA data while maintaining relatively stable out-of-domain perplexity."

That claim is well supported by your results.

4. For a follow-up paper, the most valuable addition would be a **human evaluation benchmark** of 50–100 unseen TB questions comparing:

   * Base GPT-2
   * GPT-2 + LoRA r16

   In biomedical NLP, reviewers generally find actual answer quality more compelling than perplexity alone.
