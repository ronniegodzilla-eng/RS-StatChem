# Modul Graph & Analysis

Modul **Graph & Analysis** dipakai untuk mengolah data numerik menjadi grafik ilmiah dan
menganalisisnya. Cara kerjanya mengikuti OriginLab: data diketik atau diimpor ke dalam
**workbook** berkolom X dan Y, grafik dibuat dari kolom yang dipilih, lalu analisis seperti
*curve fitting*, analisis puncak, pengolahan sinyal, dan uji statistik dijalankan dari menu.
Hasil analisis berupa **report sheet** di dalam workbook dan kurva yang langsung ditambahkan
ke grafik.

Semua jendela dalam satu sesi kerja (workbook, matriks, grafik, catatan) membentuk satu
**proyek**. Proyek disimpan sebagai berkas `.rsop`, sedangkan satu grafik bisa disimpan
tersendiri sebagai berkas `.rsg`.

## Mengenal jendela kerja

Saat modul dibuka, Anda langsung mendapat proyek baru berisi workbook kosong **Book1**.
Gambar jendela kerja berikut diambil setelah data kalibrasi diketik di Book1 dan sebuah
grafik dibuat darinya; setiap nomor merah dijelaskan di bawah gambar.

Hal penting yang perlu diingat sejak awal: **isi menu bar berganti mengikuti jenis jendela
yang aktif**, sama seperti di OriginLab.

| Jendela aktif | Menu yang tampil |
|---|---|
| Workbook | File, Edit, View, Plot, Column, Worksheet, Analysis, Statistics, Tools, Format, Window, Help |
| Grafik | File, Edit, View, Graph, Format, Insert, Analysis, Tools, Window, Help |
| Matriks | File, Edit, View, Plot, Matrix, Tools, Format, Window, Help |

Jadi, jika Anda mencari menu **Plot** atau **Statistics** tetapi tidak menemukannya, klik dulu
jendela workbook. Menu **Graph** dan **Insert** baru muncul setelah jendela grafik diklik.

Panel-panel di tepi jendela bisa ditutup dan dibuka lagi dari menu **View** atau dengan
pintasan ++Alt+1++ (Project Explorer), ++Alt+2++ (Results Log), ++Alt+6++ (Messages Log), dan
++Alt+8++ (Object Manager). Panel juga bisa diseret ke posisi lain atau dilepas menjadi
jendela mengambang. Susunannya diingat untuk sesi berikutnya.

![graph.window]

> [!TIP] Project Explorer bisa memuat beberapa folder. Klik kanan di bagian folder untuk
> membuat folder baru, lalu seret jendela dari daftar di bawahnya ke folder tujuan. Dengan
> [[View > Show Windows in Active Folder Only]] tercentang, workspace hanya menampilkan
> jendela dari folder yang sedang dipilih, sehingga proyek besar tetap rapi.

## Workbook dan worksheet

Workbook adalah tempat data. Satu workbook bisa berisi beberapa **worksheet** (tab di bagian
bawah), dan report hasil analisis juga muncul sebagai tab baru di workbook yang sama.

### Kolom, designation, dan baris label

Setiap kolom punya **nama pendek** (A, B, C, ...) dan **designation** yang menentukan
perannya dalam grafik dan analisis. Designation ditulis dalam kurung di judul kolom, misalnya
A(X) dan B(Y). Di atas data terdapat beberapa baris label.

![graph.workbook]

| Designation | Arti |
|---|---|
| **X** | Variabel bebas (sumbu mendatar) |
| **Y** | Variabel terikat; setiap Y diplot terhadap kolom X terdekat di kirinya |
| **Z** | Nilai ketiga untuk data XYZ (kontur, permukaan 3D) |
| **xEr / yEr** | Galat X atau Y; otomatis menjadi *error bar* dan bisa dipakai sebagai bobot fit |
| **L** | Label, misalnya nama sampel atau kategori |
| **N** | *Disregard*: kolom diabaikan saat membuat grafik |

Baris **Long Name** dan **Units** sangat berguna: keduanya dipakai otomatis sebagai judul
sumbu, misalnya "Konsentrasi (mg/L)", dan sebagai teks legenda. Isilah keduanya sebelum
membuat grafik agar Anda tidak perlu mengedit judul sumbu satu per satu.

Untuk mengubah designation, pilih kolom dengan mengklik judulnya, lalu gunakan
[[Column > Set As]], tombol designation di toolbar Column, atau klik kanan judul kolom.

![graph.set_as]

Beberapa cara cepat saat mengisi worksheet:

- Ketik langsung di sel. Saat Anda mengetik di baris terakhir, worksheet otomatis menambah
  baris.
- ++Ctrl+D++ menambah kolom baru, ++Ctrl+Shift+O++ membuka **Column Properties** (nama,
  Long Name, Units, Comments, designation, lebar kolom, dan rumus sekaligus).
- Data dari Excel bisa ditempel dengan ++Ctrl+V++. Jika baris pertama yang disalin berisi
  judul, judul itu masuk ke Long Name.
- Pilih beberapa sel untuk melihat statistik cepat (rata-rata, jumlah, cacah, minimum,
  maksimum) di status bar.

### Mengisi kolom dengan rumus (Set Column Values)

Kolom bisa diisi dengan rumus. Pilih kolom tujuan, lalu buka [[Column > Set Column Values]]
(++Ctrl+Q++). Di contoh berikut, kolom D diisi absorbansi dalam satuan mAU, yaitu kolom B
dikali 1000.

![graph.set_values]

Rumus memakai sintaks gaya Origin: `col(B)` adalah isi kolom B, `i` adalah nomor baris, dan
tersedia fungsi seperti `sin`, `exp`, `ln`, `log`, `sqrt`, `abs`, `round`, serta `rnd()` dan
`grnd()` untuk bilangan acak. Contoh lain:

| Rumus | Hasil |
|---|---|
| `col(A)^2` | Kuadrat kolom A |
| `(col(B)-col(C))/col(C)*100` | Selisih persen B terhadap C |
| `ln(col(B))` | Logaritma natural kolom B |
| `sin(i/10)` | Gelombang sinus terhadap nomor baris |

Rumus disimpan di baris **F(x)=** dan dihitung ulang otomatis setiap kali kolom sumbernya
berubah (status bar menunjukkan **AU : ON**, *auto update*). Anda juga bisa mengetik rumus
langsung di sel F(x)= tanpa membuka dialog.

> [!NOTE] Jika Anda mengisi kolom From dan To (rentang baris), rumus hanya diterapkan pada
> baris itu dan **tidak** disimpan di F(x)=, sehingga tidak dihitung ulang otomatis.

### Mengimpor data dari berkas

Data dari instrumen atau spreadsheet tidak perlu diketik ulang:

- [[File > Import > Single ASCII / CSV]] (++Ctrl+K++) mengimpor satu berkas teks. Pemisah
  kolom dan tanda desimal (titik atau koma) dikenali otomatis.
- [[File > Import > Excel (XLS, XLSX)]] mengimpor satu sheet Excel.
- [[File > Import > Import Wizard]] memberi kendali penuh beserta pratinjau, seperti
  gambar di bawah.
- Format khusus instrumen juga didukung: JCAMP-DX, Thermo Galactic SPC, MATLAB, HDF5/NetCDF,
  WAV, dan citra ke matriks.

![graph.import]

Data diimpor ke workbook aktif jika workbook itu masih kosong; jika tidak, workbook baru
dibuat. Anda juga bisa langsung menyeret berkas CSV, TXT, atau Excel dari Finder/Explorer ke
jendela modul.

> [!TIP] Jika Anda sering mengimpor berkas dengan susunan yang sama (misalnya keluaran
> spektrofotometer laboratorium Anda), simpan pengaturannya di Import Wizard dengan
> **Save Filter...**. Lain kali cukup pilih filter tersebut.

## Membuat grafik

### Memilih kolom dan jenis grafik

1. Di worksheet, pilih kolom Y yang akan diplot dengan mengklik judul kolomnya. Tahan ++Ctrl++
   untuk memilih beberapa kolom. Kolom X tidak perlu ikut dipilih karena setiap Y otomatis
   dipasangkan dengan kolom X di kirinya.
2. Buka menu **Plot** dan pilih jenis grafik, atau klik tombol di toolbar **2D Graphs** di
   bagian bawah jendela.

![graph.plot_menu]

Menu Plot dikelompokkan sebagai berikut:

| Submenu | Isi |
|---|---|
| **Basic 2D** | Line, Scatter, Line + Symbol, Y Error, Bubble, Color Mapped, Spline, Lollipop, Error Band, ... |
| **Bar, Pie, Area** | Column, Bar (biasa, bertumpuk, 100%), Pie, Doughnut, Area, Fill Area |
| **Multi-Panel/Axis** | Double-Y, Stack, Waterfall, 2/4/9 panel |
| **Statistical** | Box Chart, Histogram, Violin, Q-Q Plot, Bland-Altman, Pareto, QC Chart, ... |
| **Contour, 3D** | Kontur, heatmap, scatter dan permukaan 3D |
| **Specialized** | Polar, Ternary, Vector, Radar, Wind Rose |

Jika susunan kolom Anda tidak mengikuti pola X-Y biasa, gunakan [[Plot > Plot Setup]] untuk
menentukan sendiri kolom mana yang menjadi X, Y, galat, atau label.

> [!TIP] Untuk memplot hanya sebagian baris, misalnya baris 3 sampai 10, blok sel-sel itu
> (bukan judul kolomnya) sebelum memilih jenis grafik. Grafik akan mengingat rentang itu,
> misalnya `[Book1]Sheet1!B[3:10]`.

Grafik tetap **terhubung** dengan data asalnya. Jika angka di worksheet diubah, grafik
langsung ikut berubah.

### Mengatur tampilan grafik

Hampir semua bagian grafik bisa diatur dengan **klik dua kali** pada bagian itu: judul
sumbu, angka sumbu, legenda, atau kurva data.

![graph.graph_parts]

- **Judul sumbu dan teks**: klik dua kali untuk mengedit langsung di tempat. Toolbar Format
  menyediakan jenis huruf, ukuran, tebal, miring, subskrip, superskrip, dan huruf Yunani.
- **Sumbu**: klik dua kali angka sumbu (atau [[Format > Axes]]) untuk mengatur rentang,
  skala linier/log, tick, dan grid. [[Graph > Rescale to Show All]] (++Ctrl+R++)
  mengembalikan rentang agar semua data terlihat.
- **Legenda**: klik dua kali untuk mengatur isi dan bingkainya; [[Graph > Legend]]
  menampilkan atau menyembunyikannya.
- **Plot data**: klik dua kali kurva atau titik untuk membuka dialog propertinya.

![graph.plot_details]

Menu **Graph** (aktif saat jendela grafik dipilih) berisi perintah untuk menyusun grafik
yang lebih kompleks:

- [[Graph > Add Plot to Layer]] menambah kolom dari workbook mana pun ke grafik yang sudah ada.
- [[Graph > New Layer (Axes)]] dan **Layer Management** membuat panel tambahan (bertumpuk,
  berdampingan, atau *inset*). **Panel Layout** menyusun 2 sampai 9 panel sekaligus.
- **Axis Scale**, **Axis Breaks**, dan **Top Axis Linked by Formula**, misalnya sumbu atas
  dalam nm untuk sumbu bawah dalam cm⁻¹.
- **Reference Lines**, **Add Error Bars**, dan **Add Function Plot** (kurva dari rumus).
- **Copy Format** dan **Paste Format** untuk menyamakan gaya beberapa grafik.

Menu **Format** juga menyediakan **Theme** untuk mengganti gaya grafik dengan satu klik dan
**Save as Template** untuk menyimpan gaya buatan Anda.

### Mengekspor grafik untuk publikasi

Pilih jendela grafik, lalu buka [[File > Export > Graphs]] (++Ctrl+G++).

![graph.export]

1. Pilih **Format**: PNG atau TIFF untuk gambar raster, SVG, PDF, atau EPS untuk gambar
   vektor yang tetap tajam saat diperbesar.
2. Untuk PNG atau TIFF, pilih **Resolution (DPI)**. Kebanyakan jurnal meminta 300 sampai
   600 DPI.
3. Centang **Custom size** jika jurnal menentukan lebar gambar, misalnya 8,5 cm untuk satu
   kolom. Grafik ditata ulang pada ukuran itu, sementara ukuran huruf tetap.
4. Klik **Export...** dan tentukan nama berkasnya.

> [!TIP] Untuk menempel grafik ke Word atau PowerPoint, cukup tekan ++Ctrl+J++
> ([[Edit > Copy Page]]) lalu tempel di dokumen. Semua grafik proyek bisa diekspor sekaligus
> dengan [[File > Export > All Graphs in Project]].

## Contoh lengkap: kurva kalibrasi dengan Linear Fit

Contoh ini mengikuti pekerjaan yang sangat umum di laboratorium kimia: membuat kurva
kalibrasi absorbansi terhadap konsentrasi larutan standar, lalu menentukan persamaan garis
dan koefisien determinasinya.

### Langkah 1: memasukkan data

1. Buka modul Graph & Analysis. Workbook **Book1** sudah tersedia.
2. Ketik konsentrasi larutan standar (0, 1, 2, 4, 6, 8, 10) di kolom A dan absorbansinya di
   kolom B.
3. Tambahkan kolom C dengan ++Ctrl+D++, isi dengan simpangan baku tiap titik, lalu jadikan
   **Y Error** dengan [[Column > Set As > Y Error]].
4. Isi Long Name kolom A dengan `Konsentrasi` dan Units-nya dengan `mg/L`, lalu Long Name
   kolom B dengan `Absorbansi`.

### Langkah 2: membuat scatter plot

5. Klik judul kolom **B(Y)** untuk memilihnya.
6. Pilih [[Plot > Basic 2D > Y Error]] (atau **Scatter**). Grafik baru **Graph1** muncul
   dengan judul sumbu yang diambil dari Long Name dan Units, serta *error bar* dari kolom C.

![graph.ex_scatter]

### Langkah 3: menjalankan Linear Fit

7. Pastikan jendela **Graph1** aktif (klik judulnya), lalu pilih
   [[Analysis > Fitting > Linear Fit]].
8. Periksa pengaturannya, lalu klik **Fit**.

![graph.fit_dialog]

Pengaturan penting di dialog ini:

- **Range**: data yang di-fit. Jika grafik aktif, pilihannya adalah plot-plot di grafik itu.
- **Weight**: biarkan **No Weighting** untuk regresi biasa. Pilih **Instrumental** agar
  setiap titik dibobot dengan 1/SD² dari kolom yEr.
- **Recalculate**: dengan **Auto**, hasil fit dihitung ulang otomatis jika data diubah;
  **Manual** hanya menandai bahwa hasilnya perlu dihitung ulang.
- **Add fit curve to the graph**: garis hasil fit ditambahkan ke grafik.

### Langkah 4: membaca hasilnya

Setelah **Fit** diklik, tiga hal terjadi sekaligus: garis fit merah dan kotak ringkasan
ditambahkan ke grafik, report sheet dibuat di workbook, dan ringkasan teks dicatat di Results
Log.

![graph.fit_result]

Klik tab **FitLinear1** di workbook untuk membuka report lengkapnya. Setiap bagian bisa dilipat
atau dibuka dengan tanda ⊟/⊞, dan tabelnya bisa disalin ke Excel atau Word.

![graph.fit_report]

| Bagian report | Isi |
|---|---|
| **Notes, Input Data** | Data masukan, model, persamaan, pembobotan, jumlah titik |
| **Parameters** | Intercept dan Slope beserta galat baku, nilai t, nilai p, dan selang kepercayaan 95% |
| **Statistics** | R-Square (COD), Adj. R-Square, Pearson's r, Reduced Chi-Sqr, dan lainnya |
| **ANOVA** | Uji signifikansi regresi |
| **Fitted Curves Plot, Residual Plots** | Grafik fit dengan pita kepercayaan 95% dan plot residual terhadap X |

Pada contoh ini persamaan yang diperoleh adalah *y* = 0,0113 + 0,0982 *x* dengan
R² = 0,99995. Data kurva fit (X dan Y yang dihitung) tersimpan di sheet **FitLinearCurve1**.

> [!TIP] Ikon gembok di pojok kiri atas report menunjukkan status **Recalculate**. Klik
> ikon itu untuk menghitung ulang atau memilih **Change Parameters** agar dialog fit dibuka
> lagi dengan pengaturan sebelumnya.

> [!NOTE] Untuk kebutuhan analitik yang lebih lengkap, gunakan
> [[Analysis > Fitting > Calibration Curve (LOD/LOQ)]]. Perintah ini menghitung kurva
> kalibrasi (opsional berbobot), batas deteksi (LOD) dan batas kuantitasi (LOQ), serta
> konsentrasi sampel dari absorbansi yang Anda masukkan.

### Fit nonlinier dan fungsi lain

Langkahnya sama untuk jenis fit lain di menu [[Analysis > Fitting]]:

- **Polynomial Fit** untuk polinomial orde 2 sampai 9.
- **Nonlinear Curve Fit** untuk fungsi dari pustaka yang dikelompokkan per kategori
  (eksponensial, puncak Gaussian/Lorentzian/Voigt, sigmoid, kinetika enzim seperti
  Michaelis-Menten, kinetika kimia, isoterm adsorpsi Langmuir/Freundlich/BET, reologi,
  dan banyak lagi) atau persamaan
  buatan Anda sendiri pada kategori **User Defined**, misalnya `y = a*exp(-b*x) + c`. Nilai
  awal parameter boleh dikosongkan karena aplikasi menebaknya otomatis. Parameter juga bisa
  ditetapkan (*Fixed*) atau diberi batas bawah dan atas.
- **Nonlinear Curve Fit (Multiple Data, Global, Constraints)** untuk mem-fit beberapa
  dataset sekaligus dengan parameter bersama (*global fit*).
- **Compare Models** dan **Rank Models** membantu memilih model terbaik berdasarkan uji F,
  AIC, dan BIC.
- Fungsi buatan sendiri yang sering dipakai bisa disimpan permanen melalui
  [[Tools > Fitting Function Organizer]].

## Analisis puncak dan sinyal

Menu [[Analysis > Peaks and Baseline]] dan [[Analysis > Signal Processing]] ditujukan untuk
data spektrum, kromatogram, dan sinyal instrumen lain.

**Peak Analyzer** adalah wizard bertahap: pilih tujuan (mencari puncak, mengintegrasi, atau
fit puncak), buat dan kurangkan baseline, cari puncak, lalu integrasikan atau fit. Gambar
berikut memperlihatkan tiga pita serapan yang tumpang tindih dipisahkan dengan fit Gaussian.
Tabelnya memberi pusat, tinggi, luas, persentase luas, dan FWHM setiap puncak.

![graph.peaks]

Perintah lain yang sering dipakai:

- **Baseline (create and subtract)**, **Find Peaks**, dan **Multiple Peak Fit** untuk
  langkah-langkah tunggal tanpa wizard.
- **Quick Peaks Gadget** dan **Quick Fit Gadget**: seret sebuah daerah di grafik, maka
  puncak di dalamnya langsung diintegrasi atau di-fit.
- **Smoothing** (Savitzky-Golay, LOESS, Adjacent-Averaging, ...), **FFT** dan filter
  frekuensi, filter IIR, konvolusi, korelasi, dan wavelet di menu Signal Processing.
- [[Analysis > Spectroscopy]]: konversi satuan nm, cm⁻¹, dan eV, plot Tauc untuk celah pita,
  Kubelka-Munk, serta konversi absorbansi ↔ transmitansi.
- [[Analysis > Mathematics]]: turunan, integral, normalisasi, interpolasi, rata-rata beberapa
  kurva, dan pengurangan data referensi (blanko).

Hasil setiap analisis ditulis ke kolom atau sheet baru di workbook data sumbernya, dan bila
dijalankan dari grafik, kurva hasilnya ikut ditambahkan ke grafik tersebut.

## Statistik

Menu **Statistics** (aktif saat workbook dipilih) menyediakan analisis statistik yang lazim
di laboratorium tanpa harus pindah ke modul Statistics.

![graph.stats_menu]

- **Descriptive Statistics**: statistik kolom dan baris, frekuensi, korelasi, uji normalitas,
  *distribution fit*, serta uji pencilan Grubbs dan Dixon's Q.
- **Hypothesis Testing**: t-test satu sampel, dua sampel, dan berpasangan, uji varians, dan
  uji proporsi.
- **ANOVA**: satu, dua, dan tiga faktor, pengukuran berulang, dan MANOVA, dengan uji lanjut
  Tukey, Bonferroni, Dunnett, dan lainnya.
- **Nonparametric Tests**, **Multivariate Analysis** (PCA, klaster, diskriminan, PLS),
  **Survival Analysis**, **ROC Curve**, dan **Power and Sample Size**.

Data bisa disusun sebagai satu kolom per kelompok (*raw*) atau sebagai kolom data dan kolom
penanda kelompok (*indexed*). Hasilnya selalu berupa report sheet dengan ikon gembok yang
sama seperti pada report fit.

> [!TIP] Jika data Anda lebih cocok diolah dengan tampilan gaya SPSS (Variable View, Output
> Viewer), gunakan modul **Statistics**. Modul Graph & Analysis paling nyaman untuk analisis
> yang terkait langsung dengan grafik dan data instrumen.

## Matriks dan grafik 3D

Data yang berupa nilai Z pada kisi X–Y, misalnya peta permukaan atau hasil simulasi,
disimpan di **matrix book**.

1. Buat matriks dengan [[File > New > Matrix]].
2. Atur ukuran kisi dan rentang X, Y dengan [[Matrix > Set Dimensions/Labels]].
3. Isi nilainya dengan [[Matrix > Set Values]], misalnya `sin(x)*cos(y)`, atau impor citra
   dengan [[File > Import > Image to Matrix]].
4. Dengan matriks aktif, pilih jenis grafik dari menu **Plot**: kontur, heatmap, citra, atau
   permukaan 3D.

![graph.matrix]

Data XYZ di worksheet bisa diubah menjadi matriks dengan
[[Worksheet > Convert to Matrix > XYZ Gridding]]. Grafik 3D bisa diputar dengan menyeretnya,
atau melalui [[Format > Rotate 3D]]. Menu **Matrix** juga berisi operasi seperti transpose,
rotasi, pemotongan (*crop*), filter, interpolasi 2D, dan FFT 2D.

## Menyimpan dan membuka proyek

- [[File > Save Project]] (++Ctrl+S++) menyimpan seluruh proyek ke satu berkas `.rsop`:
  workbook beserta rumus dan report, matriks, grafik, catatan, dan susunan folder.
- [[File > Open]] (++Ctrl+O++) membuka proyek `.rsop` atau grafik `.rsg`. Berkas yang
  terakhir dibuka tersedia di [[File > Recent Projects]].
- Setiap perubahan bisa dibatalkan dengan ++Ctrl+Z++, terpisah untuk setiap jendela.

> [!WARNING] Report dan grafik menyimpan rujukan ke nama workbook dan kolom sumbernya.
> Gunakan [[Edit > Rename Window]] (++F2++) untuk mengganti nama agar rujukan ikut
> diperbarui.

## Referensi Fitur

Berikut daftar semua perintah menu modul Graph & Analysis beserta fungsinya: pertama menu saat
workbook aktif, lalu perintah tambahan yang muncul saat jendela grafik atau matriks aktif.

!features graph
