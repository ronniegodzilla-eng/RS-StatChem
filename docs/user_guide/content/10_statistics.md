# Modul Statistics

Modul **Statistics** adalah tempat Anda mengolah data penelitian secara statistik, mulai dari
statistik deskriptif, uji t dan ANOVA, korelasi dan regresi, hingga analisis multivariat,
survival, dan deret waktu. Alur kerjanya sengaja dibuat mirip SPSS. Data diketik atau diimpor di
**Data Editor**, sifat setiap variabel diatur di **Variable View**, analisis dipilih dari menu
**Analyze**, dan semua hasil dikumpulkan di **Output Viewer** dalam bentuk tabel dan grafik yang
bisa disalin langsung ke Word.

Jenis berkas yang dipakai modul ini:

| Berkas | Isi |
|---|---|
| `.rsd` | Berkas data RS-StatChem: data beserta semua definisi variabel dan pengaturannya |
| `.sav`, `.por` | Data SPSS (dibuka dan disimpan dengan label variabel dan label nilai) |
| `.csv`, `.txt` | Data teks berpemisah koma, titik koma, atau Tab |
| `.xlsx` | Lembar kerja Excel |
| `.dta`, `.sas7bdat`, `.xpt` | Data Stata dan SAS |
| `.rso` | Dokumen output (isi Output Viewer) |

Bab ini memakai satu contoh data sepanjang pembahasan: 45 mahasiswa yang diajar dengan tiga
metode pembelajaran (Konvensional, Daring, Campuran), lengkap dengan nilai pretest dan posttest
serta tingkat kepuasan.

## Mengenal jendela kerja

Saat modul Statistics dibuka, muncul jendela **Data Editor**. Satu jendela Data Editor berisi
satu dataset. Jika Anda membuka berkas data kedua, jendela baru akan muncul, dan Anda bisa
berpindah di antara keduanya lewat menu **Window**.

![stats.window]

Beberapa hal yang perlu diperhatikan pada jendela ini:

- Ikon di kepala kolom menunjukkan tingkat pengukuran variabel: tiga titik berwarna untuk
  *nominal*, batang bertingkat untuk *ordinal*, dan penggaris untuk *scale*.
- Kolom berjudul `var` masih kosong dan siap diisi variabel baru.
- Status bar di kanan bawah menunjukkan jumlah kasus dan variabel. Jika Split File, Select
  Cases, atau Weight Cases sedang aktif, keterangannya (*Filter On*, *Weight On*, atau
  *Split by* dan nama variabelnya) juga muncul di sini sebagai pengingat.
- Menu **Data** dan **Transform** mengolah data, **Analyze** berisi seluruh prosedur
  statistik, **Graphs** membuat grafik, dan **Utilities** berisi alat bantu seperti informasi
  variabel. Ringkasan isi setiap menu ada di bagian *Gambaran menu* di akhir bab ini.

### Toolbar

![stats.toolbar]

Semua tombol toolbar juga tersedia di menu. Tombol **Frequencies** dan **Descriptives** di ujung
kanan adalah pintas untuk dua analisis yang paling sering dipakai, yaitu
[[Analyze > Descriptive Statistics > Frequencies]] dan
[[Analyze > Descriptive Statistics > Descriptives]].

## Menyiapkan data

Data statistik selalu disusun dengan pola yang sama: **satu baris untuk satu kasus, satu kolom
untuk satu variabel**. Jika Anda membandingkan kelompok, kelompok itu dinyatakan dengan
variabel kode, misalnya `metode` bernilai 1, 2, atau 3. Kelompok tidak dipisah ke kolom-kolom
berbeda.

### Mendefinisikan variabel di Variable View

Buka tab **Variable View** (atau tekan ++Ctrl+T++). Di sini setiap variabel tampil sebagai satu
baris, dan setiap kolom mengatur satu sifat variabel tersebut.

![stats.variable_view]

| Kolom | Fungsi |
|---|---|
| **Name** | Nama singkat variabel. Harus diawali huruf, tanpa spasi (boleh memakai `_`), maksimal 64 karakter, dan tidak boleh kembar. Nama inilah yang dipakai di rumus Compute. |
| **Type** | *Numeric* (angka), *String* (teks), atau *Date* (tanggal). Pilih dari daftar di sel. |
| **Width / Decimals** | Lebar tampilan dan jumlah angka di belakang koma. Ini hanya memengaruhi tampilan; perhitungan tetap memakai nilai lengkap. |
| **Label** | Keterangan lengkap, misalnya "Nilai posttest". Label inilah yang tampil di tabel output, jadi isilah dengan rapi. |
| **Values** | Label nilai untuk kode, misalnya 1 = Laki-laki dan 2 = Perempuan. Klik dua kali sel untuk mengisinya. |
| **Missing** | Kode yang dianggap data hilang, misalnya `9` untuk "tidak menjawab". Beberapa kode dipisah dengan koma. Sel kosong selalu dianggap hilang. |
| **Columns / Align** | Lebar kolom dan perataan teks di Data View. |
| **Measure** | Tingkat pengukuran: *Scale* (interval/rasio), *Ordinal*, atau *Nominal*. Beberapa dialog memakainya untuk memilah variabel yang cocok. |
| **Role** | Peran variabel untuk prosedur yang membutuhkannya (Input, Target, Both, None, Partition, Split). |

Untuk menambah variabel, ketik namanya di baris kosong pertama Variable View. Klik kanan nomor
baris untuk menyisipkan atau menghapus variabel.

> [!TIP] Biasakan mengisi **Label** dan **Values** sejak awal. Tabel output akan menampilkan
> "Metode pembelajaran: Daring", bukan "metode: 2", sehingga hasil analisis bisa langsung
> dipakai di laporan.

### Mengisi label nilai

1. Di Variable View, klik dua kali sel **Values** pada baris variabel, misalnya `metode`.
2. Ketik kode di **Value** dan teksnya di **Label**, lalu klik **Add** (atau tekan ++Enter++).
3. Ulangi untuk setiap kode. Untuk memperbaiki label, pilih di daftar, ubah teksnya, lalu klik
   **Change**.
4. Klik **OK**.

![stats.value_labels]

### Mengetik dan menempel data di Data View

Kembali ke tab **Data View**, klik sel, lalu ketik nilainya. Tekan ++Enter++ untuk turun satu
baris atau ++Tab++ untuk pindah ke kanan. Mengetik di kolom `var` yang masih kosong otomatis
membuat variabel baru (misalnya `VAR00001`) yang dapat Anda ganti namanya di Variable View.
Untuk variabel berlabel, ketiklah kodenya (misalnya `2`), bukan labelnya.

Data dari Excel atau tabel lain dapat disalin seperti biasa. Blok sel di Excel, salin dengan
++Ctrl+C++, klik sel tujuan di Data View, lalu tempel dengan ++Ctrl+V++. Jika blok yang ditempel
lebih besar dari data yang ada, kasus dan variabel baru dibuat otomatis. Sebaliknya,
[[Edit > Copy with Variable Names]] menyalin sel terpilih beserta nama variabelnya sebagai baris
pertama.

Tombol **Value Labels** di toolbar (atau [[View > Value Labels]]) mengganti tampilan kode angka
menjadi labelnya. Tombol ini hanya mengubah tampilan; data yang tersimpan tetap berupa kode.

![stats.labels_on]

> [!NOTE] Perubahan data bisa dibatalkan dengan [[Edit > Undo]] (++Ctrl+Z++), termasuk hasil
> perintah Data dan Transform seperti Sort Cases atau Compute Variable.

### Mengimpor data dari CSV, Excel, dan SPSS

Gunakan [[File > Open > Open Data]] (++Ctrl+O++) atau submenu [[File > Import Data]]. Berkas
juga bisa diseret langsung ke jendela Data Editor.

- **CSV/teks**: jendela **Import Text Data** muncul lebih dulu. Pemisah kolom dan simbol desimal
  ditebak otomatis. Periksa pratinjaunya, lalu klik **Import**.
- **Excel (.xlsx)**: jika buku kerja berisi beberapa lembar, Anda diminta memilih lembarnya.
  Baris pertama dipakai sebagai nama variabel. Nama yang tidak valid diperbaiki otomatis,
  sedangkan judul aslinya disimpan sebagai label.
- **SPSS (.sav, .por), Stata, SAS**: dibuka beserta label variabel, label nilai, dan kode
  missing yang tersimpan di dalam berkasnya.

![stats.import_text]

> [!TIP] Berkas CSV dari Excel berbahasa Indonesia biasanya memakai titik koma sebagai pemisah
> dan koma sebagai desimal. Jika angka terbaca sebagai teks di pratinjau, ubah
> **Decimal symbol** menjadi *Comma*.

### Menyimpan data

[[File > Save]] (++Ctrl+S++) menyimpan data ke berkas `.rsd`. Berkas ini memuat data, semua
definisi variabel, serta pengaturan seperti Split File, bobot, dan aturan validasi. Pada
[[File > Save As]] Anda juga bisa memilih format SPSS `.sav`. Untuk format lain (CSV, Excel,
Stata, SAS) gunakan submenu [[File > Export]].

Data hasil impor CSV atau Excel belum punya berkas `.rsd`, sehingga **Save** pertama akan
menanyakan nama berkas.

### Membuat variabel baru dengan Compute

Variabel turunan, misalnya selisih posttest dan pretest, dibuat dengan
[[Transform > Compute Variable]]. Ketik nama variabel baru di **Target Variable**, susun rumus di
**Numeric Expression** (klik dua kali variabel atau fungsi untuk menyisipkannya), lalu klik
**OK**. Variabel baru langsung ditambahkan di kolom paling kanan.

![stats.compute]

Rumus bisa memakai operator aritmetika, perbandingan, dan ratusan fungsi gaya SPSS, misalnya
`MEAN(item1 TO item5)`, `SQRT(x)`, atau `LN(x)`. Tombol **If...** membatasi perhitungan pada
kasus yang memenuhi syarat tertentu.

## Contoh lengkap: One-Way ANOVA

Pertanyaan penelitian contoh: *apakah rata-rata nilai posttest berbeda di antara tiga metode
pembelajaran?* Variabel terikatnya `posttest` (scale), dan variabel pengelompoknya `metode`
(nominal, tiga kategori). Analisis yang tepat adalah **One-Way ANOVA**, diikuti uji *post hoc*
untuk mengetahui pasangan metode mana yang berbeda.

### Langkah 1: buka dialog

Pilih [[Analyze > Compare Means > One-Way ANOVA]].

![stats.menu_compare]

### Langkah 2: isi variabel

1. Pilih **Nilai posttest [posttest]** di daftar kiri, lalu klik tombol panah di samping
   **Dependent List**. Klik dua kali nama variabel juga memindahkannya.
2. Pilih **Metode pembelajaran [metode]**, lalu klik panah di samping **Factor**.
3. Biarkan **Estimate effect size for overall tests** tercentang agar eta-squared dan
   omega-squared ikut dihitung.

![stats.anova_dialog]

Untuk mengembalikan variabel ke daftar kiri, pilih variabel di kotak tujuan lalu klik panah
(yang kini menghadap ke kiri). **Reset** mengosongkan seluruh isian. Dialog mengingat isian
terakhir selama aplikasi terbuka, sehingga analisis ulang tidak perlu diisi dari awal.

### Langkah 3: pilih uji lanjut dan opsi

1. Klik **Post Hoc...**, centang **Tukey**, lalu klik **Continue**.
2. Klik **Options...**, centang **Descriptive**, **Homogeneity of variance test**, **Welch**, dan
   **Means plot**, lalu klik **Continue**.
3. Klik **OK**.

![stats.posthoc]

> [!TIP] Pilihan post hoc bergantung pada uji homogenitas varians (Levene). Jika Levene tidak
> signifikan, gunakan **Tukey**. Jika signifikan, gunakan **Games-Howell** dan laporkan uji
> **Welch** sebagai pengganti F biasa.

### Langkah 4: baca hasilnya

Output Viewer terbuka otomatis dan menampilkan hasil analisis di bawah judul **Oneway**:

![stats.output]

Cara membacanya, dari atas ke bawah:

- **Log** berisi sintaks setara SPSS untuk analisis ini, berguna sebagai catatan prosedur.
- **Descriptives**: rata-rata posttest metode Konvensional 63,94, Daring 69,17, dan Campuran
  79,93.
- **Tests of Homogeneity of Variances**: Levene *Based on Mean* Sig. = ,851 (> 0,05), artinya
  varians ketiga kelompok dapat dianggap sama. ANOVA biasa dan uji Tukey layak dipakai.
- **ANOVA**: F(2, 42) = 17,70 dengan Sig. < 0,001. Paling tidak ada satu pasang metode yang
  rata-ratanya berbeda. Uji Welch di bawahnya memberi kesimpulan yang sama.
- **ANOVA Effect Sizes**: eta-squared 0,457, artinya sekitar 46% variasi nilai posttest
  berkaitan dengan metode pembelajaran.

Gulir ke bawah untuk melihat hasil uji lanjut:

![stats.output_posthoc]

Pada **Multiple Comparisons**, tanda bintang menandai selisih rata-rata yang signifikan.
Metode Campuran lebih tinggi daripada Konvensional (selisih 15,99) dan Daring (10,75), keduanya
dengan Sig. ≤ 0,001. Selisih Konvensional dan Daring (5,23; Sig. = ,148) tidak signifikan. Tabel
**Homogeneous Subsets** merangkum hal yang sama: Konvensional dan Daring berada dalam satu
subset, sedangkan Campuran berdiri sendiri. **Means Plot** menggambarkan pola tersebut.

Contoh kalimat laporan: *Terdapat perbedaan nilai posttest yang signifikan di antara ketiga
metode pembelajaran, F(2, 42) = 17,70, p < 0,001, η² = 0,46. Uji Tukey menunjukkan metode
campuran menghasilkan nilai lebih tinggi daripada metode konvensional dan daring, sedangkan
metode konvensional dan daring tidak berbeda nyata.*

> [!NOTE] Output memakai titik sebagai pemisah desimal dan menghilangkan nol di depan angka
> kurang dari 1 (`.851`), seperti SPSS. Nol di depan dapat dimunculkan lewat
> [[Edit > Options]] dengan mencentang *Display a leading zero for numbers below 1*.

### Contoh lain: Independent-Samples T Test

Untuk membandingkan dua kelompok, misalnya nilai posttest laki-laki dan perempuan, pilih
[[Analyze > Compare Means > Independent-Samples T Test]]. Masukkan `posttest` ke
**Test Variable(s)** dan `jk` ke **Grouping Variable**. Setelah itu klik **Define Groups...**
dan isi kode kedua kelompok (1 dan 2). Label nilai variabel ditampilkan di bagian bawah
sub-dialog sebagai pengingat.

![stats.ttest]

Pada data contoh, rata-rata laki-laki 70,18 dan perempuan 72,06. Uji Levene tidak signifikan
(Sig. = ,719), jadi yang dibaca adalah baris *Equal variances assumed*: t(43) = −0,63,
p (two-sided) = ,534, Cohen's d = −0,19. Dengan demikian, tidak ada perbedaan nilai posttest
yang berarti antara laki-laki dan perempuan.

## Output Viewer

Semua hasil analisis dari semua jendela Data Editor terkumpul di satu **Output Viewer**. Jendela
ini bisa dibuka kapan saja dari tombol **Output Viewer** di toolbar atau
[[Window > Output Viewer]].

Nomor-nomor berikut merujuk ke gambar Output Viewer pada Langkah 4 di atas.

- **Navigasi (1)**: setiap analisis menjadi satu cabang berisi tabel dan grafiknya. Klik sebuah
  judul untuk melompat ke objek tersebut. Klik kanan untuk menyalin atau menghapusnya.
- **Isi output (2)**: klik tabel atau grafik untuk memilihnya; objek terpilih diberi bingkai
  merah. Klik dua kali tabel untuk membuka **Edit Table**, tempat Anda bisa mengubah isi sel,
  menambah catatan kaki, menyembunyikan baris/kolom, atau mengatur jumlah desimal.
- **Export to Word (3)** (++Ctrl+E++) menyimpan seluruh output sebagai dokumen `.docx`.
  [[File > Export]] juga menyediakan Excel, PDF, HTML, dan teks.
- **Copy (4)** (++Ctrl+C++) menyalin objek terpilih. Saat ditempel di Word, hasilnya menjadi
  tabel Word asli yang bisa diedit, lengkap dengan garis gaya SPSS atau APA. Pilihan lain ada di
  [[Edit > Copy Special]]: sebagai tabel Word (HTML), sebagai gambar, atau sebagai teks untuk
  Excel.
- **Zoom (5)** memperbesar atau memperkecil tampilan tanpa mengubah hasil salinan.
- **TableLook (6)** mengganti gaya semua tabel antara *SPSS Default* dan *APA (7th edition)*.
  Gaya APA langsung siap untuk skripsi dan artikel.

Gambar berikut menunjukkan output yang sama dengan TableLook APA: setiap tabel diberi nomor
(*Table 3*) dan judul miring, hanya memakai garis horizontal, dan catatan diawali *Note.*
Tabel ANOVA sedang dipilih dan siap disalin lewat [[Edit > Copy Special]].

![stats.output_copy]

Fitur lain yang berguna:

- [[Pivot > Pivoting Trays]] (++Ctrl+P++) memindahkan dimensi tabel antara baris, kolom, dan
  lapisan. [[Pivot > Transpose Table (swap rows and columns)]] menukar baris dan kolom.
- [[Insert > Insert Title]] dan [[Insert > Insert Text]] menambahkan judul atau catatan Anda
  sendiri di antara hasil analisis.
- [[File > Save Output]] menyimpan isi Output Viewer sebagai berkas `.rso`, yang nanti bisa
  dibuka lagi dengan [[File > Open > Output]] dari Data Editor.

> [!TIP] Untuk skripsi atau artikel, pilih TableLook **APA (7th edition)**, klik tabel yang
> dibutuhkan, tekan ++Ctrl+C++, lalu tempel di Word. Garis, huruf, dan perataan tabel sudah
> mengikuti gaya APA, sehingga Anda tidak perlu membuat ulang tabel secara manual.

## Menganalisis sebagian data

Tiga perintah di bagian bawah menu **Data** mengubah data mana yang dipakai oleh analisis
berikutnya, tanpa menghapus data apa pun:

- [[Data > Split File]] menjalankan setiap analisis terpisah per kelompok, misalnya per jenis
  kelamin, atau membandingkannya dalam satu tabel.
- [[Data > Select Cases]] menyaring kasus menurut syarat (misalnya `usia >= 21`), sampel acak,
  rentang, atau variabel filter. Kasus yang tidak terpilih diberi coretan pada nomor barisnya.
- [[Data > Weight Cases]] memberi bobot kasus dengan variabel frekuensi, berguna untuk data
  tabel silang yang sudah dirangkum.

Selama salah satu pengaturan ini aktif, status bar menampilkan *Filter On*, *Weight On*, atau
*Split by* diikuti nama variabelnya. Jangan lupa mematikannya lagi setelah selesai.

## Gambaran menu

| Menu | Isi utama |
|---|---|
| **File** | Data baru, buka, impor (CSV, Excel, SPSS, Stata, SAS), simpan `.rsd`/`.sav`, ekspor, informasi berkas data, berkas terakhir |
| **Edit** | Undo/Redo, salin-tempel, sisipkan variabel atau kasus, Find/Replace, Go to Case/Variable, Options (TableLook, nol di depan desimal) |
| **View** | Status bar, toolbar, huruf, garis grid, Value Labels, berpindah Data View/Variable View |
| **Data** | Definisi properti variabel, validasi dan pencarian kasus ganda/tidak wajar, Sort, Transpose, Merge Files, Restructure, Aggregate, matching, Split File, Select Cases, Weight Cases |
| **Transform** | Compute, Count, Shift Values, Recode, Automatic Recode, variabel dummy, Visual/Optimal Binning, Rank Cases, Date and Time Wizard, deret waktu, Replace Missing Values, bilangan acak |
| **Analyze: dasar** | Power Analysis, Meta Analysis, Reports, Descriptive Statistics, Bayesian Statistics, Tables, Compare Means |
| **Analyze: model** | General Linear Model, Generalized Linear Models, Mixed Models, Correlate, Regression, Loglinear, Mediation & Moderation (model PROCESS), Neural Networks |
| **Analyze: multivariat** | Classify (cluster, tree, discriminant, ROC), Dimension Reduction (factor, correspondence), Scale (reliabilitas, kappa, Bland-Altman, MDS), Structural Equation Modeling (CB-SEM dan PLS-SEM) |
| **Analyze: khusus** | Nonparametric Tests, Forecasting, Survival, Multiple Response, Missing Value Analysis dan Multiple Imputation, Complex Samples, Quality Control, Direct Marketing, Conjoint Analysis |
| **Graphs** | Chart Builder gaya SPSS, Legacy Dialogs, Weibull Plot, serta Graph Studio untuk grafik ilmiah (garis, batang, kontur, 3D, dan lain-lain) |
| **Utilities** | Informasi variabel, komentar berkas data, variable sets untuk menyembunyikan variabel yang tidak dipakai |
| **Window / Help** | Output Viewer, daftar jendela data, pindah modul, cek versi baru, About |

> [!NOTE] Sebagian besar dialog analisis memakai pola yang sama dengan contoh ANOVA di atas:
> daftar variabel di kiri, kotak tujuan di tengah, tombol sub-dialog (Statistics, Options,
> Plots, Save, Bootstrap ...) di kanan, serta **OK**, **Reset**, dan **Cancel** di bawah.
> Prosedur yang membuat variabel baru (misalnya residual regresi atau skor faktor) menambahkannya
> ke Data View dengan nama gaya SPSS seperti `PRE_1` dan `RES_1`.

## Referensi Fitur

Berikut daftar lengkap perintah di menu Data Editor dan Output Viewer modul Statistics beserta fungsinya.

!features stats
