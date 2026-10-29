# Modul Image Analysis

Modul **Image Analysis** dipakai untuk membuka, mengolah, dan mengukur citra ilmiah: foto
mikroskop, hasil pemindaian gel, citra medis, atau citra astronomi. Cara kerjanya mengikuti
**ImageJ**, sehingga nama menu, dialog, pintasan keyboard, bahkan bahasa makronya sama. Jika
Anda sudah pernah memakai ImageJ atau Fiji, Anda akan langsung merasa akrab.

Modul ini membuka TIFF (termasuk TIFF gaya ImageJ beserta kalibrasi dan ROI-nya), PNG, JPEG,
GIF, BMP, DICOM, FITS, PGM, AVI, dan ZIP. Selain citra, modul ini juga membuka berkas seleksi
`.roi` dan `RoiSet.zip`, tabel CSV, makro `.ijm`, serta proyek `.rsij` yang menyimpan citra,
ROI Manager, dan tabel hasil dalam satu berkas.

## Mengenal jendela kerja

Berbeda dengan modul lain, Image Analysis tidak memakai satu jendela besar. Seperti di ImageJ,
yang muncul pertama adalah sebuah **bar utama** kecil berisi menu, ikon alat, dan baris status.
Setiap citra yang Anda buka tampil di **jendelanya sendiri**. Tabel Results, ROI Manager, Log,
dan editor makro juga berupa jendela terpisah.

### Bar utama

![image.bar]

- **Menu** (1) memuat semua perintah, dikelompokkan seperti di ImageJ: **File** untuk membuka
  dan menyimpan, **Edit** untuk seleksi dan pengaturan, **Image** untuk tipe, ukuran, stack,
  dan warna, **Process** untuk filter dan operasi biner, **Analyze** untuk pengukuran, serta
  **Plugins** untuk makro dan plugin Fiji.
- **Alat seleksi area** (2) membuat ROI berbentuk persegi, oval, poligon, atau bebas. Klik kanan
  ikon untuk memilih variannya, misalnya persegi bersudut bulat atau elips.
- **Garis dan sudut** (3) dipakai untuk mengukur panjang, membuat profil intensitas, dan mengukur
  sudut.
- **Point, Wand, dan Text** (4): Point menandai atau menghitung titik, Wand memilih satu objek
  secara otomatis dengan sekali klik, dan Text menulis teks di citra.
- **Zoom dan Scrolling** (5): klik citra dengan kaca pembesar untuk memperbesar, klik kanan untuk
  memperkecil. Alat tangan menggeser bagian citra yang tampil.
- **Color picker** (6) menunjukkan warna depan dan warna latar yang dipakai Fill, Draw, dan Clear.
- **Baris status** menampilkan posisi kursor dan nilai piksel di bawahnya. Setelah sebuah perintah
  selesai, baris ini menampilkan lama prosesnya.

> [!TIP] Klik dua kali sebuah ikon alat untuk membuka pengaturannya, misalnya lebar garis,
> toleransi Wand, atau tipe penanda Point.

### Jendela citra

![image.window]

Judul jendela berisi nama citra dan, bila tidak 100%, tingkat zoom-nya. Baris info di bawah
judul meringkas ukuran citra (dalam satuan kalibrasi bila sudah diatur), tipe, LUT, dan memori
yang dipakai. Seleksi yang Anda buat tampil sebagai garis kuning. Seleksi inilah yang diukur oleh
[[Analyze > Measure]] dan diolah oleh sebagian besar perintah Process.

Zoom diatur dengan tombol plus dan minus di keyboard (atau [[Image > Zoom]]), dengan roda mouse
sambil menahan ++Ctrl++, atau dengan alat kaca pembesar. Saat kursor bergerak di atas citra, baris
status di bar utama menampilkan koordinat dan nilai piksel.

## Membuka citra

Pilih [[File > Open]] (++Ctrl+O++), atau seret berkas ke bar utama. Anda juga bisa membuka citra
dari Module Launcher atau dengan klik dua kali di Finder/Explorer, karena berkas citra otomatis
dibuka di modul ini. Untuk membuka semua citra dalam satu folder sebagai satu stack, pakai
[[File > Import > Image Sequence]].

![image.open_samples]

[[File > Open Samples]] berisi citra contoh resmi ImageJ yang sering dipakai di tutorial, seperti
**Blobs**, **Cell Colony**, dan **MRI Stack**.

> [!NOTE] Citra contoh **diunduh dari internet** saat pertama kali dibuka, lalu disimpan di cache.
> Setelah itu citra tersebut bisa dibuka tanpa koneksi. Pilih
> [[File > Open Samples > Cache Sample Images]] untuk mengunduh semuanya sekaligus, misalnya
> sebelum praktikum di ruang tanpa internet.

### Tipe citra

Tipe citra menentukan rentang nilai piksel dan perintah yang bisa dipakai. Tipe diubah melalui
[[Image > Type]].

| Tipe | Nilai piksel | Dipakai untuk |
|---|---|---|
| **8-bit** | bilangan bulat 0 sampai 255 | sebagian besar analisis; wajib untuk banyak operasi biner |
| **16-bit** | bilangan bulat 0 sampai 65535 | kamera ilmiah, mikroskop fluoresens, DICOM |
| **32-bit** | bilangan desimal (float) | hasil perhitungan, rasio, citra dengan NaN |
| **RGB Color** | tiga channel 8-bit (merah, hijau, biru) | foto berwarna; diukur per channel atau setelah diubah ke 8-bit |
| **Composite** | beberapa channel, masing-masing dengan LUT | citra fluoresens multi-channel |

> [!NOTE] Citra Blobs memakai *inverting LUT*: objek yang tampak gelap justru bernilai tinggi.
> Perhatikan tulisan "inverting LUT" di baris info, karena hal ini memengaruhi threshold dan
> pembacaan nilai piksel.

### Kalibrasi jarak (Set Scale)

Secara bawaan semua ukuran dinyatakan dalam piksel. Agar luas dan panjang keluar dalam satuan
nyata (µm, mm), kalibrasi citra terlebih dahulu:

1. Pilih alat **Straight Line**, lalu tarik garis sepanjang benda yang ukurannya diketahui,
   misalnya *scale bar* pada foto mikroskop atau mistar yang ikut terpotret.
2. Pilih [[Analyze > Set Scale]]. **Distance in pixels** sudah terisi panjang garis tadi.
3. Isi **Known distance** dan **Unit of length**, lalu klik **OK**.

![image.set_scale]

Setelah dikalibrasi, baris info menampilkan ukuran citra dalam satuan tersebut, dan semua hasil
pengukuran memakai satuan yang sama. Centang **Global** bila semua citra yang terbuka diambil
dengan perbesaran yang sama. Untuk menambahkan batang skala ke citra yang akan dipublikasikan,
gunakan [[Analyze > Tools > Scale Bar]].

> [!TIP] Kalibrasi tersimpan di dalam berkas TIFF. Simpan citra dengan [[File > Save]] agar Anda
> tidak perlu mengkalibrasinya lagi saat membukanya kembali.

## Seleksi (ROI) dan pengukuran

Hampir semua pengukuran berawal dari **seleksi** atau ROI (*region of interest*). Buat seleksi
dengan alat di bar utama, lalu pilih [[Analyze > Measure]] (++Ctrl+M++). Satu baris hasil
ditambahkan ke tabel **Results**. Tanpa seleksi, yang diukur adalah seluruh citra.

Beberapa cara cepat membuat seleksi:

- tahan ++Shift++ saat menyeret untuk membuat persegi atau lingkaran sempurna;
- klik objek dengan alat **Wand** untuk menyeleksi tepinya secara otomatis (paling mudah pada
  citra yang sudah di-threshold);
- [[Edit > Selection > Specify]] untuk seleksi dengan ukuran dan posisi yang tepat;
- [[Edit > Selection > Restore Selection]] untuk memunculkan kembali seleksi terakhir.

### ROI Manager

Bila Anda perlu mengukur banyak daerah, simpan setiap seleksi di **ROI Manager**
([[Analyze > Tools > ROI Manager]]). Buat seleksi, lalu tekan ++T++ atau klik **Add**. Setiap
ROI diberi nama otomatis berdasarkan posisinya, dan bisa diganti dengan **Rename**.

![image.roi_manager]

Klik nama ROI untuk memilihnya kembali di citra. Tombol **Measure** mengukur ROI yang dipilih,
atau semuanya bila tidak ada yang dipilih. Menu **More >>** berisi perintah untuk menyimpan dan
membuka kumpulan ROI (`RoiSet.zip`), menggabungkan ROI, serta **Multi Measure** yang mengukur
semua ROI di setiap slice stack.

### Memilih besaran yang diukur

Besaran yang dihitung oleh Measure dan Analyze Particles dipilih di
[[Analyze > Set Measurements]]. Pilihan ini berlaku sampai Anda mengubahnya lagi.

![image.set_measurements]

Beberapa pilihan yang sering dipakai:

- **Area**, **Mean gray value**, dan **Min & max gray value** untuk luas dan intensitas;
- **Centroid** dan **Perimeter** untuk posisi pusat dan keliling;
- **Shape descriptors** untuk *circularity*, *aspect ratio*, *roundness*, dan *solidity*;
- **Feret's diameter** untuk diameter terpanjang (berguna untuk ukuran partikel);
- **Integrated density** untuk jumlah intensitas, misalnya pada kuantifikasi pita gel atau
  fluoresensi;
- **Limit to threshold** agar hanya piksel dalam rentang threshold yang diukur;
- **Display label** agar nama citra dan ROI tercatat di kolom Label.

### Tabel Results

![image.results]

Setiap pengukuran menambah satu baris. Melalui menu jendela Results, Anda bisa menyimpan tabel
ke CSV (**File**), menyalinnya ke Excel atau ke modul Statistics (**Edit ▸ Copy**), dan
menambahkan baris ringkasan rata-rata, simpangan baku, minimum, dan maksimum (**Results**).
Tabel dikosongkan dengan [[Analyze > Clear Results]].

> [!TIP] Data dari tabel Results bisa langsung dianalisis lebih lanjut. Salin tabelnya, lalu
> tempel di Data Editor modul Statistics untuk uji statistik, atau di workbook modul Graph &
> Analysis untuk membuat grafik.

## Contoh lengkap: menghitung partikel

Contoh ini menghitung dan mengukur bintik-bintik pada citra **Blobs**. Alur kerjanya sama dengan
menghitung koloni bakteri, sel, butir pati, atau partikel serbuk. Anggap saja 100 piksel pada
citra ini setara dengan 10 µm.

### Langkah 1: membuka dan mengkalibrasi citra

1. Buka citra dengan [[File > Open Samples > Blobs]] (++Ctrl+Shift+B++), atau buka berkas citra
   Anda sendiri dengan [[File > Open]].
2. Tarik garis lurus sepanjang 100 piksel, pilih [[Analyze > Set Scale]], lalu isi
   **Known distance** 10 dan **Unit of length** `µm`, seperti pada bagian Set Scale di atas.

### Langkah 2: memisahkan partikel dari latar (threshold)

1. Pilih [[Image > Adjust > Threshold]] (++Ctrl+Shift+T++). Piksel yang masuk rentang threshold
   ditandai merah.
2. Periksa citranya: semua bintik harus berwarna merah dan latarnya tidak ikut.

![image.threshold]

Jika metode **Default** kurang pas, coba metode lain dari daftar (misalnya Otsu atau Li), atau
geser kedua slider. Centang **Dark background** bila objek Anda terang di atas latar gelap.

Analyze Particles bisa langsung dijalankan pada citra yang sudah di-threshold seperti ini. Bila
objek perlu dirapikan dulu, klik **Apply** (atau pilih [[Process > Binary > Convert to Mask]])
untuk mengubah citra menjadi biner hitam-putih. Pada citra biner, Anda bisa mengisi lubang di
dalam objek dengan [[Process > Binary > Fill Holes]] atau memisahkan partikel yang saling
menempel dengan [[Process > Binary > Watershed]].

### Langkah 3: menjalankan Analyze Particles

1. Pilih [[Analyze > Set Measurements]] dan centang besaran yang Anda perlukan, misalnya
   **Area**, **Perimeter**, dan **Shape descriptors**.
2. Pilih [[Analyze > Analyze Particles]], lalu atur dialognya seperti di bawah.
3. Klik **OK**.

![image.analyze_particles]

- **Size** membatasi luas partikel yang dihitung, dalam satuan kalibrasi. Isi misalnya
  `0.5-Infinity` agar bintik kecil (derau) tidak ikut terhitung.
- **Circularity** membatasi bentuk, dari 0 (memanjang) sampai 1 (lingkaran sempurna).
- **Show: Outlines** membuat gambar baru berisi garis tepi setiap partikel beserta nomornya.
- Centang **Display results**, **Clear results**, dan **Summarize**. Centang
  **Exclude on edges** bila partikel yang terpotong tepi citra tidak boleh dihitung.

### Langkah 4: membaca hasil

Setelah **OK**, tiga jendela muncul:

![image.particles_result]

Jendela **Summary** memuat jumlah partikel (**Count**), luas total, ukuran rata-rata, dan
persentase luas yang ditempati partikel (**%Area**). Pada contoh ini terdapat 64 partikel yang
menempati sekitar 34% luas citra. Tabel **Results** memuat ukuran setiap partikel. Nomor
barisnya sama dengan nomor pada gambar *Drawing of blobs.gif*, sehingga partikel yang janggal
mudah ditelusuri.

> [!TIP] Agar hasil bisa diperiksa ulang, pilih **Show: Overlay** atau centang
> **Add to Manager**. Garis tepi partikel lalu tampil di atas citra asli atau masuk ke ROI
> Manager, dan bisa diklik satu per satu.

> [!WARNING] Hasil penghitungan sangat bergantung pada threshold dan batas **Size**. Gunakan
> pengaturan yang sama untuk semua citra dalam satu percobaan, dan laporkan metode threshold
> serta batas ukuran yang Anda pakai di bagian metode penelitian.

## Filter dan pengolahan citra

Menu **Process** berisi perintah pengolahan. Sebagian besar bekerja pada seleksi aktif, atau pada
seluruh citra bila tidak ada seleksi, dan bisa dibatalkan dengan [[Edit > Undo]].

![image.filters]

- **Mengurangi derau:** [[Process > Filters > Gaussian Blur]], [[Process > Filters > Median]],
  atau [[Process > Noise > Despeckle]].
- **Latar tidak rata:** [[Process > Subtract Background]] (rolling ball) sebelum threshold.
- **Kontras:** [[Image > Adjust > Brightness/Contrast]] hanya mengubah tampilan, sedangkan
  [[Process > Enhance Contrast]] dan **Apply** di panel Brightness/Contrast mengubah nilai piksel.
- **Operasi biner:** Erode, Dilate, Open, Close-, Fill Holes, Skeletonize, Distance Map, dan
  Watershed di [[Process > Binary]].
- **Aritmetika:** konstanta di [[Process > Math]], antarcitra di [[Process > Image Calculator]].
- **Domain frekuensi:** FFT, Bandpass Filter, dan FD Math di [[Process > FFT]].
- **Banyak berkas sekaligus:** [[Process > Batch > Macro]] menjalankan makro pada setiap citra
  dalam satu folder.

> [!NOTE] Brightness/Contrast mengubah cara citra *ditampilkan*, bukan datanya. Untuk pengukuran
> intensitas, ukurlah citra asli, bukan citra yang kontrasnya sudah di-**Apply**.

## Stack dan hyperstack

**Stack** adalah sekumpulan citra berukuran sama dalam satu jendela, misalnya irisan *z* dari
mikroskop konfokal atau frame dari time-lapse. Jendela stack memiliki slider di bawah citra.

![image.stack]

Pindah slice dengan slider, roda mouse, atau tombol ++<++ dan ++>++. Perintah penting ada di
[[Image > Stacks]]:

- **Z Project** menggabungkan semua slice menjadi satu citra (proyeksi maksimum, rata-rata, dan
  lainnya);
- **Make Montage** menyusun slice-slice berdampingan untuk publikasi;
- **Plot Z-axis Profile** memplot intensitas seleksi terhadap slice atau waktu;
- **Orthogonal Views** dan **Reslice** menampilkan irisan XZ dan YZ;
- **Animation** memutar stack seperti video.

**Hyperstack** memiliki lebih dari satu dimensi sekaligus: channel (*c*), irisan (*z*), dan
waktu (*t*). Setiap dimensi mendapat slider sendiri. Stack biasa diubah menjadi hyperstack dengan
[[Image > Hyperstacks > Stack to Hyperstack]], dan tampilan channel diatur di
[[Image > Color > Channels Tool]].

## Makro: mengulang pekerjaan secara otomatis

Semua perintah menu bisa dijalankan dari **makro** berbahasa ImageJ (*IJ1 macro*). Skrip Python
juga didukung. Makro sangat berguna bila analisis yang sama harus diulang untuk puluhan citra.

![image.macro]

Setiap baris `run("Nama perintah", "opsi")` sama dengan memilih perintah itu dari menu dan mengisi
dialognya. Makro di atas menjalankan langkah-langkah contoh hitung partikel sekaligus, lalu
menuliskan jumlah partikel ke jendela **Log** dengan `print`.

Cara termudah membuat makro adalah merekamnya:

1. Pilih [[Plugins > Macros > Record]]. Jendela Recorder terbuka.
2. Kerjakan analisis Anda seperti biasa melalui menu. Setiap langkah tercatat sebagai satu baris
   `run(...)`.
3. Klik **Create** di Recorder untuk membuka hasilnya di editor makro, lalu simpan sebagai berkas
   `.ijm`.
4. Jalankan dengan [[Macros > Run Macro]] di editor (++Ctrl+R++), atau dari
   [[Plugins > Macros > Run]].

Makro yang terdiri dari beberapa blok `macro "Nama" { ... }` bisa dipasang dengan
[[Plugins > Macros > Install]], sehingga setiap blok muncul sebagai perintah di menu
**Plugins ▸ Macros**. Daftar lengkap fungsi makro tersedia di [[Help > Macro Functions]].

> [!TIP] Makro ImageJ yang Anda temukan di internet atau di lampiran artikel umumnya bisa
> langsung dijalankan di sini, selama perintah yang dipakainya ada di menu modul ini. Gunakan
> [[Plugins > Utilities > Find Commands]] (++Ctrl+L++) untuk mencari sebuah perintah berdasarkan
> namanya.

## Plugin gaya Fiji

Beberapa plugin Fiji yang paling sering dipakai sudah terpasang di [[Plugins > Fiji]], tanpa
perlu instalasi tambahan:

![image.fiji]

- **Auto Threshold** dan **Auto Local Threshold**: threshold otomatis global dan lokal, termasuk
  pilihan **Try all** untuk membandingkan semua metode sekaligus;
- **Enhance Local Contrast (CLAHE)**: peningkatan kontras lokal;
- **MorphoLibJ**: filter morfologi, rekonstruksi, watershed berbasis marker atau peta jarak,
  penghapusan objek di tepi, dan pelabelan objek;
- **Cell Counter**: penghitungan manual sel per kategori dengan sekali klik;
- **Coloc 2**: analisis kolokalisasi dua channel fluoresens (Pearson, Manders, dan lainnya);
- **Analyze Skeleton**: analisis cabang dan sambungan pada kerangka;
- **Bleach Correction**, **Colour Deconvolution**, dan **KymographBuilder** untuk time-lapse dan
  histologi.

Penjelasan setiap perintah tersebut ada di Referensi Fitur di bawah.

## Referensi Fitur

Berikut daftar lengkap semua perintah di menu modul Image Analysis beserta fungsinya.

!features image
