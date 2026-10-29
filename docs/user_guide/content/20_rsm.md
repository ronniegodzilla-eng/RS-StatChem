# Modul RSM / DOE

Modul **RSM / DOE** dipakai untuk merancang dan menganalisis percobaan. *Design of
Experiments* (DOE) membantu Anda menentukan kombinasi faktor yang perlu dicoba agar informasi
yang diperoleh maksimal dengan jumlah percobaan sesedikit mungkin. *Response Surface
Methodology* (RSM) kemudian memodelkan hubungan antara faktor dan respons dengan persamaan
polinomial, sehingga Anda dapat melihat permukaan responsnya dan mencari kondisi optimum.

Alur kerjanya mengikuti Design-Expert: (1) buat desain dengan wizard, (2) jalankan percobaan
sesuai urutan run dan isi hasilnya ke design layout, (3) analisis setiap respons dari kiri ke
kanan melalui tab Transform, Fit Summary, Model, ANOVA, Diagnostics, dan Model Graphs, lalu
(4) cari kondisi optimum. Satu desain beserta data, model, dan pengaturan optimasinya disimpan
dalam satu berkas `.rsx`.

## Mengenal jendela kerja

Setelah desain dibuat atau dibuka, jendela RSM / DOE tampak seperti gambar di bawah. Contoh yang
dipakai di seluruh bab ini adalah desain Box-Behnken untuk ekstraksi senyawa fenolik dengan
tiga faktor dan dua respons. Bagian-bagian jendela yang bernomor adalah:

- **Menu bar** (1) memuat semua perintah. Daftar lengkapnya ada di Referensi Fitur di akhir
  bab ini.
- **Toolbar** (2) berisi tombol pintas, dari kiri: New Design, Open, Save, Undo, Redo, Copy,
  Paste, Export Graph, tampilan Coded/Actual, Evaluate Design, Numerical, Graphical, Point
  Prediction, dan Screen Tips.
- **Navigation pane** di kiri dibagi menjadi empat kelompok node. Klik sebuah node untuk
  membuka layarnya di bagian kanan.
- Kelompok **Design** (3) berisi *design layout* (tabel run), **Summary** (ringkasan desain dan
  statistik respons), **Graph Columns** (plot antarkolom), **Evaluation** (evaluasi desain),
  dan **Notes** (catatan bebas yang ikut tersimpan).
- Kelompok **Analysis** (4) berisi satu node untuk setiap respons, ditambah **Analysis
  Summary** yang merangkum model semua respons. Respons yang datanya belum lengkap diberi
  keterangan seperti *(empty)* atau *(13/17)*.
- Kelompok **Optimization** (5) berisi **Numerical**, **Graphical**, dan **Propagation of
  Error**, sedangkan **Post Analysis** (6) berisi **Point Prediction**, **Confirmation**, dan
  **Coefficients Table**.
- **Petunjuk** (7) berwarna kuning di bagian atas setiap layar menjelaskan langkah
  berikutnya, **design layout** (8) adalah tabel run percobaan, dan **status bar** (9)
  menampilkan pesan singkat serta hasil perintah terakhir.

![rsm.window]

> [!TIP] Tekan ++F1++ ([[Help > Screen Tips]]) kapan saja untuk membaca penjelasan singkat
> tentang layar yang sedang terbuka.

## Membuat desain baru

Saat modul RSM / DOE dibuka dari Module Launcher, wizard **New Design** langsung muncul. Jika
wizard ditutup, jendela menampilkan halaman awal seperti gambar di bawah.

Dari halaman ini Anda dapat membuat desain baru, membuka berkas `.rsx`, atau membangun desain
dari data lama (*historical data*). Tombol **Tutorial data** membuka contoh desain dari buku
teks lengkap dengan datanya, cocok untuk berlatih. Contoh yang sama tersedia di
[[Help > Tutorial Data]].

![rsm.start]

### Halaman 1: jenis desain dan faktor

Pilih [[File > New Design]] (++Ctrl+N++) untuk membuka wizard.

1. Di kolom paling kiri (1), pilih **keluarga desain**. Kolom tengah (2) lalu menampilkan
   desain-desain dalam keluarga tersebut, dan penjelasan singkatnya muncul di bawah judul.
2. Atur **jumlah faktor** (3).
3. Isi **tabel faktor** (4): **Name**, **Units**, **Type** (Numeric atau Categoric), serta
   **Minimum** dan **Maximum**. Kolom Coded Low dan Coded High menunjukkan padanan nilai
   aktual dengan kode −1 dan +1. Untuk CCD juga muncul kolom −alpha dan +alpha.
4. Sesuaikan **opsi desain** (5), misalnya jumlah titik pusat (*center points*), blok, atau
   replikasi. Pilihan opsinya berbeda-beda sesuai jenis desain.
5. Periksa **ringkasan** (6), yang memperbarui jumlah run secara langsung, lalu klik
   **Next >** (7).

![rsm.wizard]

Keluarga desain yang tersedia:

| Keluarga | Desain | Kegunaan |
|---|---|---|
| **Factorial** | Regular Two-Level, Min-Run, Definitive Screening, Plackett-Burman, Taguchi, Multilevel Categoric | Penyaringan (*screening*) faktor dan efek utama atau interaksinya |
| **Response Surface** | Central Composite (CCD), Box-Behnken, 3-Level Factorial, Optimal, One Factor, Historical Data, Space Filling | Model kuadratik untuk mencari kondisi optimum |
| **Mixture** | Simplex Lattice, Simplex Centroid, Simplex Screening, Extreme Vertices, Optimal | Formulasi, yaitu komponen yang jumlahnya tetap (misalnya 100%) |
| **Split-Plot** | Two-Level, Multilevel Categoric, CCD, Optimal | Ada faktor yang sulit diubah (*hard to change*) |
| **Optimal (Custom)** | Optimal Process, Optimal Mixture, Combined Mixture-Process | Daerah desain berbatasan, model khusus, atau gabungan mixture dan proses |

> [!NOTE] Tanda centang **Randomize run order** sebaiknya tetap aktif. Urutan acak melindungi
> hasil dari pengaruh faktor luar yang berubah selama percobaan, misalnya suhu ruang atau
> kondisi alat.

### Halaman 2: respons

Di halaman kedua, tentukan jumlah respons (1), nama desain (2), serta nama dan satuan setiap
respons (3). Nilai respons diisi nanti di design layout. Klik **Finish** (4) untuk membuat
desain.

![rsm.wizard_responses]

> [!TIP] Sebelum menjalankan percobaan, buka node **Evaluation**
> ([[Design Tools > Evaluate Design]]) untuk memeriksa apakah desain cukup untuk model yang
> direncanakan: derajat bebas, VIF koefisien (idealnya 1), power, dan *leverage* setiap run.

## Mengisi data respons

Setelah Finish, design layout menampilkan semua run. Kerjakan percobaan menurut kolom **Run**
(1), yaitu urutan acak, lalu catat hasilnya di sel respons (3) yang berwarna kuning. Kolom
**Std** menunjukkan urutan standar desain. Kolom faktor (2) memuat pengaturan setiap run dan
tidak perlu diubah.

Ada tiga cara mengisi respons:

- **Mengetik** langsung di sel respons, lalu tekan ++Enter++ untuk turun ke baris berikutnya.
- **Menempel** dari Excel: salin satu kolom (atau beberapa kolom) di Excel, klik sel respons
  pertama, lalu tekan ++Ctrl+V++. Pastikan urutan baris di Excel sama dengan urutan baris di
  layout.
- **Mengimpor** dengan [[File > Import Responses (CSV / Excel)]]. Kolom berkas dicocokkan
  dengan nama respons (atau R1, R2, ...). Jika berkas memiliki kolom Std atau Run, barisnya
  dicocokkan lewat kolom itu sehingga urutannya tidak harus sama.

![rsm.layout]

Klik kanan judul kolom untuk mengurutkan layout menurut Std atau Run, menampilkan jenis titik
(*space point type*), atau beralih antara satuan aktual dan kode. Klik kanan sel untuk
menyalin, menempel, mengosongkan sel respons, mengabaikan run (*Ignore Run*), atau
menyisipkan, menghapus, dan menggandakan run. Hilangkan centang di kolom **Select** untuk
mengecualikan sebuah run dari analisis tanpa menghapusnya.

> [!NOTE] Analisis baru dapat dimulai setelah cukup banyak nilai respons terisi, setidaknya
> jumlah faktor ditambah dua. Hasil yang lengkap tentu memerlukan semua run.

## Contoh lengkap: optimasi ekstraksi fenolik

Seorang peneliti ingin memaksimalkan rendemen ekstrak dan kadar fenolik total dari suatu
bahan tanaman. Tiga faktor diteliti dengan desain **Box-Behnken** berisi 5 titik pusat, sehingga
total ada 17 run.

| Faktor | Nama | Satuan | Rendah (−1) | Tinggi (+1) |
|---|---|---|---|---|
| A | Suhu | °C | 40 | 60 |
| B | Waktu | min | 30 | 90 |
| C | Etanol | % | 50 | 90 |

Responsnya adalah **R1: Rendemen** (%) dan **R2: Fenolik** (mg GAE/g).

### Langkah 1: membuat desain dan mengisi data

1. Pilih [[File > New Design]]. Di wizard, klik **Response Surface**, lalu **Box-Behnken**.
2. Atur **Factors** menjadi 3, lalu isi nama, satuan, minimum, dan maksimum seperti tabel di
   atas. Biarkan **Center points** bernilai 5. Ringkasan akan menunjukkan 17 run.
3. Klik **Next >**, atur **Responses** menjadi 2, lalu isi nama dan satuannya. Beri nama
   desain *Ekstraksi fenolik*, kemudian klik **Finish**.
4. Kerjakan percobaan sesuai urutan Run dan isi hasilnya di kolom respons.
5. Simpan desain dengan [[File > Save]] (++Ctrl+S++).

### Langkah 2: Transform dan Fit Summary

Klik **R1:Rendemen** di bawah **Analysis** pada navigation pane. Kerjakan tab-tab di bagian
atas dari kiri ke kanan.

Tab **Transform** menentukan transformasi respons. Pilih **None** kecuali rasio maksimum
terhadap minimum respons lebih dari 10 atau plot Box-Cox menyarankan transformasi tertentu.
Tombol **Use Box-Cox Recommendation** memilihkan transformasi yang disarankan.

Tab **Fit Summary** membandingkan model Linear, 2FI, Quadratic, dan Cubic. Model yang
disarankan (5) adalah model berorde tertinggi yang suku tambahannya signifikan
(*sequential p* < 0,05), tidak ter-alias, *lack of fit*-nya tidak signifikan, serta memiliki
Adjusted R² dan Predicted R² terbaik. Pada contoh ini model **Quadratic** disarankan, dengan
Adjusted R² 0,9962 dan Predicted R² 0,9853. Model Cubic ditandai *Aliased* karena desain
Box-Behnken tidak cukup untuk menaksir semua sukunya. Klik **Use Suggested Model** (4) untuk
memakai model tersebut dan langsung membuka tab ANOVA. Kotak **Analysis 1** dengan tombol
**+** dan **−** (2) memungkinkan Anda membuat beberapa analisis berbeda untuk satu respons.

![rsm.fit_summary]

### Langkah 3: Model

Tab **Model** menampilkan semua suku kandidat. Huruf **M** berarti suku masuk model, sedangkan
**e** berarti suku tidak dipakai dan menjadi bagian galat. Klik kolom pertama sebuah suku
untuk mengganti statusnya.

Pada model kuadratik penuh, suku **BC** memiliki p = 0,2466, jauh di atas 0,10. Klik **M** di
baris BC sehingga berubah menjadi **e** (4). Model kini menjadi *Reduced Quadratic* dengan 8
suku (5). Biarkan **Keep model hierarchical** tetap aktif agar suku induk, misalnya A dan B
untuk AB, tidak terbuang.

Sebagai alternatif, gunakan **Auto Select** (2): pilih kriteria (p-value, AICc, atau BIC) dan
metode (Backward, Forward, atau Stepwise), lalu klik **Auto Select**. Daftar **Process Order**
(1) mengembalikan model ke orde tertentu, dan kotak **Add Term** dapat menambahkan suku
berorde tinggi seperti `A2B`.

![rsm.model]

### Langkah 4: ANOVA

Tab **ANOVA** menguji model yang dipilih. Periksa tiga hal berikut:

| Yang diperiksa | Kriteria | Hasil contoh |
|---|---|---|
| Model (1) | p < 0,05 (signifikan) | F = 492,80; p < 0,0001 |
| Lack of Fit (2) | p > 0,05 (tidak signifikan) | p = 0,3532 |
| Fit Statistics (3) | Selisih Adjusted R² dan Predicted R² < 0,2; Adeq Precision > 4 | 0,9959 vs 0,9877; Adeq Precision 68,5 |

Warna p-value membantu membaca tabel: merah < 0,01, hijau < 0,05, biru < 0,10, dan hitam
tidak signifikan. Gulir ke bawah untuk melihat koefisien dan **persamaan model** dalam satuan
kode maupun satuan aktual. Persamaan dalam satuan kode cocok untuk membandingkan besar
pengaruh setiap faktor, sedangkan persamaan aktual dipakai untuk menghitung prediksi
langsung dari nilai faktor.

![rsm.anova]

> [!TIP] Catatan penjelasan di bawah tabel ANOVA dapat dimatikan melalui
> [[View > Annotated ANOVA]]. Untuk menyalin semua tabel ke Word atau Excel, klik laporan lalu
> pilih [[Edit > Copy]].

### Langkah 5: Diagnostics

Tab **Diagnostics** memeriksa apakah asumsi model terpenuhi. Pilih jenis plot di daftar kiri (1). Pada **Normal Plot**, titik-titik residual sebaiknya
mengikuti garis lurus. Pada **Residuals vs. Predicted** dan **Residuals vs. Run**, titik
sebaiknya tersebar acak tanpa pola seperti corong atau tren. **Cook's Distance**, **Leverage**,
dan **DFFITS** membantu menemukan run yang berpengaruh terlalu besar, dengan garis merah sebagai
batasnya. **Box-Cox** menunjukkan apakah transformasi diperlukan. Jenis residual dapat dipilih
di kotak **Residuals** (2), dan pilihan **Report** menampilkan semua nilai diagnostik dalam
bentuk tabel.

![rsm.diagnostics]

### Langkah 6: Model Graphs

Tab **Model Graphs** memvisualisasikan model.

1. Pilih jenis grafik (1): **Perturbation**, **One Factor**, **Interaction**, **Contour**,
   **3D Surface**, **Cube**, **All Factors**, atau **Standard Error**.
2. Di **Factors Tool** (2), tentukan faktor untuk sumbu X1 dan X2. Faktor lain, di sini
   Etanol, diatur dengan slider. Nilai bawaannya adalah titik tengah rentang.
3. Klik sembarang titik pada plot kontur (3) untuk memasang **bendera** berisi nilai faktor,
   prediksi respons, dan CI 95%. **Clear Flags** menghapus semua bendera.

![rsm.model_graphs]

Pada contoh ini rendemen tertinggi berada di sekitar suhu 55 °C dan waktu 80 menit. Pilih
**3D Surface** untuk melihat bentuk permukaannya. Seret dengan tombol kiri mouse untuk memutar,
gunakan roda mouse untuk memperbesar, dan klik **Reset View** untuk kembali ke tampilan awal.

![rsm.model_graphs.2](Permukaan respons 3D rendemen terhadap suhu dan waktu)

> [!TIP] Klik kanan grafik untuk menyalin (**Copy Graph**) atau menyimpannya sebagai gambar
> (**Export Graph As**). Pilihan **Colors** dan **Contours** di panel kiri mengatur palet warna
> dan nilai garis kontur.

### Langkah 7: respons kedua

Klik **R2:Fenolik** di navigation pane dan ulangi Langkah 2 sampai 6. Setiap respons memiliki
transformasi dan modelnya sendiri. Model-model yang dipilih inilah yang dipakai pada tahap
optimasi. [[Optimization > Coefficients Table]] menampilkan koefisien semua respons
berdampingan.

### Langkah 8: optimasi numerik

Pilih [[Optimization > Numerical]]. Tab **Criteria** menentukan tujuan optimasi.

1. Untuk setiap respons, pilih **Goal**: *maximize*, *minimize*, *is target*, *is in range*,
   atau Cpk. Pada contoh ini kedua respons di-*maximize*.
2. Isi **Lower Limit** dan **Upper Limit**. Untuk *maximize*, nilai di bawah Lower Limit
   dianggap tidak dapat diterima (desirability 0), sedangkan nilai di atas Upper Limit dianggap
   ideal (desirability 1). Contoh: Rendemen 15 sampai 19,5 dan Fenolik 38 sampai 44.
3. Atur **Importance** (satu sampai lima tanda plus) jika satu respons lebih penting daripada yang lain.
   Baris faktor biasanya dibiarkan *is in range*. Pilih *is equal to* untuk menahan sebuah
   faktor pada nilai tertentu.
4. Panel bawah (3) mengatur jumlah titik awal pencarian, interval kepercayaan, dan simpangan
   baku faktor untuk *propagation of error*.

![rsm.numerical]

Buka tab **Solutions**. Pencarian berjalan otomatis, dan tombol **Find Solutions** (1)
mengulanginya. Solusi diurutkan menurut **desirability** gabungan D (3), dengan nilai 0 berarti tidak dapat
diterima dan 1 berarti ideal. Pada contoh ini solusi terbaik adalah suhu sekitar 55 °C, waktu
sekitar 83 menit, dan etanol sekitar 68%, dengan rendemen sekitar 19,5% dan fenolik sekitar
43 mg GAE/g. **Ramps** (4) menunjukkan posisi setiap faktor dan respons pada solusi yang
dipilih. Tombol **Bar Graph** dan **Report** (2) memberi tampilan lain. Tab **Graphs**
menggambar kontur atau permukaan 3D desirability di sekitar solusi.

![rsm.solutions]

### Langkah 9: optimasi grafis

[[Optimization > Graphical]] memberi gambaran daerah kerja yang aman. Di tab **Criteria**,
isi batas bawah dan/atau atas setiap respons, misalnya Rendemen ≥ 17,5 dan Fenolik ≥ 40, lalu
buka tab **Graphs**. Daerah kuning (2) adalah kombinasi suhu dan waktu yang memenuhi semua batas, dengan nilai
etanol yang diatur di **Factors Tool** (1). Daerah abu-abu gagal memenuhi sedikitnya satu
batas. Kolom **Interval** di tab Criteria dapat menerapkan batas pada CI, PI, atau TI, bukan
hanya pada nilai prediksi, sehingga hasilnya lebih konservatif.

![rsm.graphical]

### Langkah 10: prediksi dan konfirmasi

[[Optimization > Point Prediction]] menghitung prediksi pada pengaturan faktor mana pun.
Klik **Set to Optimization Solution 1** (2) untuk memakai solusi optimasi terbaik, atau
geser slider di **Factors Tool** (1). Laporan (3) memuat prediksi rata-rata, SE, interval
kepercayaan 95% (CI) untuk rata-rata, interval prediksi 95% (PI) untuk satu pengamatan baru,
dan *tolerance interval* (TI).

![rsm.prediction]

Setelah itu, lakukan beberapa run konfirmasi di laboratorium pada pengaturan tersebut.
Masukkan hasilnya di [[Optimization > Confirmation]]. Hasil dinyatakan terkonfirmasi jika
rata-rata data berada di dalam interval prediksi untuk n run.

> [!WARNING] Prediksi hanya berlaku di dalam rentang faktor yang diteliti. Jangan
> mengekstrapolasi model ke luar daerah desain tanpa percobaan tambahan.

## Alat desain lainnya

Menu **Design Tools** berisi perintah untuk mengubah desain yang sudah ada. Semua perubahan
dapat dibatalkan dengan [[Edit > Undo]].

![rsm.augment_menu]

- **Augment Design**: menambahkan run ke desain lama. **Foldover** dipakai untuk desain dua
  level, **Add Center Points** dan **Add Replicates** menambah titik pusat atau replikasi, dan
  **Add Axial Points (to CCD)** mengubah faktorial menjadi CCD. **Optimal Augment** menambahkan
  run optimal untuk model yang lebih besar. **Modify Design Space** dan **Edit Constraints**
  mengubah daerah desain.
- **Edit Layout**: menyisipkan dan menghapus run, menambah respons, membulatkan nilai faktor,
  dan menamai blok.
- **Edit Factor and Response Info**: mengganti nama dan satuan, serta mengisi simpangan baku
  faktor untuk *propagation of error* dan level diskret untuk optimasi.
- [[File > Import Data Set (Historical)]]: membangun desain dari data penelitian lama di
  CSV/Excel, lalu menganalisisnya dengan cara yang sama.
- [[Edit > Preferences]]: mengubah taraf signifikansi (alpha) dan tingkat kepercayaan interval
  untuk semua analisis.

## Referensi Fitur

Berikut semua perintah di menu modul RSM / DOE beserta fungsinya.

!features rsm
