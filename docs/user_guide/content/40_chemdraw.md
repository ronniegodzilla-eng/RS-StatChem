# Modul Chemical Drawing

Modul **Chemical Drawing** dipakai untuk menggambar struktur molekul dan skema reaksi dengan
alur kerja seperti ChemDraw. Anda bisa menggambar dengan alat ikatan, cincin, dan rantai,
mengubah nama senyawa menjadi struktur (dan sebaliknya), melihat sifat kimia dan spektrum NMR
prediksi, memeriksa kesetaraan dan stoikiometri reaksi, lalu mengekspor gambar beresolusi
tinggi untuk naskah publikasi.

Gambar disimpan sebagai berkas `.rsc` milik RS-StatChem, atau langsung sebagai dokumen ChemDraw
(`.cdx`, `.cdxml`) agar bisa dibuka rekan yang memakai ChemDraw. Modul ini juga membaca dan
menulis format pertukaran yang umum: MOL (V2000 dan V3000), SDF, RXN, SMILES, InChI, serta HELM
dan FASTA untuk peptida.

## Mengenal jendela kerja

Saat modul dibuka, Anda melihat satu halaman dokumen putih (ukuran Letter, bisa diganti ke A4)
di atas latar abu-abu, seperti selembar kertas. Gambar di bawah memperlihatkan jendela dengan
struktur aspirin. Bagian-bagian yang bernomor adalah:

- **Menu bar** (1) memuat semua perintah, dikelompokkan seperti di ChemDraw: File, Edit, View,
  Object, Structure, Text, Curves, Colors, Window, dan Help. Daftar lengkapnya ada di Referensi
  Fitur di akhir bab ini.
- **General Toolbar** (2) berisi tombol New, Open, Save, Print, Cut, Copy, Paste, Undo, Redo,
  Magnify, Reduce, Actual Size, **Clean Up Structure**, **Check Structure**, **Convert Name to
  Structure**, **Convert Structure to Name**, dan **Send to Output Viewer**.
- **Style Toolbar** (3) mengatur jenis huruf, ukuran, dan gaya teks (tebal, miring, garis
  bawah, Formula, subskrip, superskrip) untuk teks yang dipilih.
- **Main Toolbar** (4) di kiri adalah palet semua alat menggambar. Penjelasannya ada di bagian
  berikut.
- **Halaman dokumen** (5) adalah tempat Anda menggambar. Putar roda mouse sambil menahan
  ++Ctrl++ untuk memperbesar atau memperkecil.
- **Analysis** (6) menampilkan rumus kimia, massa eksak, berat molekul, pola m/z, dan analisis
  unsur dari struktur yang dipilih, atau seluruh dokumen bila tidak ada yang dipilih.
- **Chemical Properties** (7) menampilkan perkiraan sifat fisika-kimia satu molekul.
- **Status bar** menampilkan petunjuk alat yang sedang aktif dan jumlah atom yang valensinya
  bermasalah. Di kanannya ada kotak **Zoom** (8): ketik persentase atau pilih *Fit*.

Jendela Analysis, Chemical Properties, dan Templates bisa ditutup, dipindah, atau dilepas
menjadi jendela mengambang. Munculkan lagi lewat menu **View**.

![chem.window]

## Main Toolbar

Setiap tombol di Main Toolbar memilih satu alat. Tombol yang memiliki segitiga kecil di pojok
kanan bawah menyimpan beberapa alat sejenis dalam *fly-out*. Tahan tombol itu sebentar atau
klik kanan untuk membuka fly-out, lalu pilih alatnya. Alat yang dipilih akan menggantikan
gambar tombol tersebut. Arahkan penunjuk ke sebuah tombol untuk melihat nama alat dan hotkey-nya.

![chem.palette]

## Dasar menggambar

### Ikatan, cincin, dan rantai

1. Pilih **Solid Bond**, lalu klik di halaman untuk membuat satu ikatan. Klik ujung ikatan untuk
   menambah ikatan berikutnya, atau seret untuk menentukan arahnya. Selama **Fixed Angles**
   aktif (menu **Object**), arah ikatan mengikuti kelipatan 15°. Tahan ++Shift++ saat
   menyeret untuk sudut bebas.
2. Klik sebuah ikatan yang sudah ada untuk mengubahnya berturut-turut menjadi rangkap dua dan
   rangkap tiga.
3. Pilih alat cincin (misalnya **Benzene**), lalu klik di tempat kosong untuk meletakkan cincin
   baru. Klik sebuah atom untuk menempelkan cincin di atom itu, atau klik sebuah ikatan untuk
   menggabungkan (fusi) cincin pada ikatan itu.
4. Pilih **Acyclic Chain** dan seret untuk menggambar rantai zig-zag. Jumlah atomnya mengikuti
   panjang seretan.

### Label atom dan hotkey

Setiap titik sudut tanpa label adalah atom karbon. Untuk mengganti unsur, arahkan penunjuk ke
atom lalu ketik hurufnya: ++o++ untuk O, ++n++ untuk N, ++s++ untuk S, ++l++ untuk Cl, dan
seterusnya. Tombol ++Enter++ membuka kotak label sehingga Anda bisa mengetik label panjang
seperti `OH`, `NH2`, atau `CO2Me`. Anda juga bisa memilih alat **Text** lalu mengklik atom.

Hotkey juga bekerja pada ikatan. Arahkan penunjuk ke ikatan lalu ketik ++2++ (rangkap dua),
++w++ (wedge), ++h++ (hashed wedge), atau ++a++ (fusi benzena). Daftar lengkapnya ada di
[[Help > Shortcuts and Hotkeys]].

### Wedge, stereokimia, dan nickname

Ikatan **Wedge** menunjukkan ikatan yang mengarah ke depan bidang, sedangkan **Hashed Wedge**
mengarah ke belakang. Ujung sempit wedge berada di atom tempat Anda mulai menggambar. Aktifkan
[[Object > Show Stereochemistry]] untuk menampilkan label CIP (*R*/*S*, *E*/*Z*) yang dihitung
otomatis dari gambar.

Label seperti `Ph`, `OMe`, `CO2Me`, `Boc`, atau `Ts` dikenali sebagai **nickname**, yaitu
singkatan gugus yang tetap dihitung sebagai atom-atom lengkapnya di rumus, massa, dan berkas
ekspor. Daftar semua nickname ada di [[File > List Nicknames]]. Untuk menguraikan nickname
menjadi atom-atomnya, pilih [[Structure > Expand Label]].

![chem.drawing]

### Memilih, memindahkan, dan merapikan

- **Marquee** memilih objek dengan kotak, **Lasso** dengan garis bebas. Klik dua kali sebuah
  ikatan dengan alat ini untuk memilih seluruh molekulnya. Seret pilihan untuk memindahkannya.
  Tombol panah menggeser pilihan sedikit demi sedikit (dengan ++Shift++: 10 pt).
- **Rotate** memutar pilihan atau struktur di bawah penunjuk. Menu **Object** menyediakan
  Flip, Rotate, Scale, Align, dan Distribute untuk pengaturan yang tepat.
- [[Structure > Clean Up Structure]] (++Ctrl+Shift+K++) merapikan struktur yang digambar
  kurang rapi sehingga panjang ikatan dan sudutnya seragam.
- [[Structure > Check Structure]] memeriksa valensi. Atom yang valensinya salah juga ditandai
  merah di gambar dan jumlahnya tampil di status bar.

> [!TIP] Cara tercepat memasukkan struktur yang sudah Anda ketahui adalah
> [[File > Import > Insert Structure Text (SMILES, MOL, InChI)]] (++Ctrl+Alt+V++). Tempelkan
> SMILES, InChI, atau blok MOL, maka strukturnya langsung tergambar rapi. SMILES reaksi
> seperti `A.B>>C` juga digambar lengkap dengan panahnya.

### Templat

Templat adalah struktur siap pakai: asam amino, basa nukleat, gula, cincin heterosiklik,
aromatik, bisiklik, proyeksi Newman dan Fischer, rumus Haworth, konformasi kursi, gugus fungsi,
gugus pelindung, dan alat gelas. Buka [[View > Show Templates Window]], pilih kategori, klik
sebuah templat, lalu klik di halaman untuk meletakkannya. Klik dua kali templat untuk
meletakkannya di tengah tampilan. Pustaka yang sama tersedia lewat tombol **Templates** di
Main Toolbar.

![chem.templates]

## Nama dan struktur

### Structure to Name

Pilih satu atau beberapa struktur (atau tanpa pilihan untuk semua struktur), lalu klik
[[Structure > Convert Structure to Name]] (++Ctrl+Alt+N++). Nama IUPAC ditulis sebagai teks di
bawah setiap struktur. Nama dicari berturut-turut dari daftar nama offline, penamaan otomatis
berbasis aturan yang diverifikasi dengan OPSIN, dan basis data nama offline. Status bar
menyebutkan sumber nama yang dipakai.

![chem.names]

### Name to Structure

Klik [[Structure > Convert Name to Structure]] (++Ctrl+Shift+N++), ketik nama senyawa (nama
umum seperti *caffeine* atau nama IUPAC seperti *2-acetyloxybenzoic acid*), lalu klik **OK**.
Strukturnya digambar di halaman dan langsung terpilih. Jika sebuah teks keterangan sedang
dipilih, teks itulah yang diubah menjadi struktur, sehingga Anda bisa menulis beberapa nama
dulu lalu mengubahnya satu per satu. SMILES atau InChI juga boleh diketik di kotak yang sama.

> [!NOTE] Nama yang tidak ada di daftar dan basis data offline dibaca oleh **OPSIN**, pengurai
> nama IUPAC yang berjalan offline di komputer Anda. OPSIN memerlukan Java, tetapi Java sudah
> disertakan di dalam paket RS-StatChem sehingga Anda tidak perlu memasangnya.

> [!WARNING] Jika nama tetap tidak ditemukan dan [[Structure > Use PubChem for Names (online)]]
> aktif (bawaannya aktif), RS-StatChem mencarinya di PubChem melalui internet. Yang dikirim
> hanya nama senyawa atau InChIKey struktur, bukan gambar Anda. Matikan pilihan ini jika Anda
> bekerja dengan struktur rahasia atau tanpa koneksi internet.

## Sifat kimia: Analysis dan Chemical Properties

Kedua jendela di kanan diperbarui otomatis setiap kali gambar atau pilihan berubah. Baris
**Selection** atau **Whole document** di atasnya menunjukkan apa yang sedang dihitung.

- **Analysis** menampilkan rumus kimia, massa eksak (monoisotopik), berat molekul, pola isotop
  m/z dengan kelimpahan relatifnya, dan persen massa setiap unsur. Jika ada beberapa molekul,
  nilainya digabung. **Decimals** mengatur jumlah angka di belakang koma.
- **Chemical Properties** menampilkan titik didih dan titik leleh, suhu, tekanan, dan volume
  kritis, energi Gibbs dan kalor pembentukan, logP, refraktivitas molar (MR), tPSA, kelarutan
  (logS), tekanan uap, pKa, dan konstanta Henry untuk **satu** molekul. Nilai eksperimen dari
  basis data terbuka ditampilkan lebih dulu. Jika tidak ada, ditampilkan perkiraan dengan galat
  terkecil. Label di dalam kurung siku menyebut sumber data atau metode perkiraannya. Arahkan penunjuk ke sebuah nilai, atau klik
  **Details**, untuk melihat semua metode beserta galatnya.

Centang nilai yang Anda perlukan, lalu klik **Paste** untuk menuliskannya sebagai teks di bawah
struktur, atau **Copy** untuk menyalinnya sebagai teks biasa.

![chem.properties]

> [!NOTE] Sifat yang diperkirakan adalah hasil model, bukan pengukuran. Gunakan untuk
> perbandingan atau perkiraan awal, dan selalu laporkan metodenya jika dipakai dalam naskah.

## Prediksi spektrum NMR

Gambar atau pilih satu molekul, lalu klik [[Structure > Predict ¹H-NMR Shifts]] atau
[[Structure > Predict ¹³C-NMR Shifts]]. Jendela **NMR Prediction** menampilkan spektrum
prediksi dan tabel puncak. Untuk ¹H, multiplet orde pertama digambar sesuai frekuensi
spektrometer yang dipilih.

![chem.nmr]

- Klik sebuah puncak di spektrum atau sebuah baris di tabel, maka atom yang bersangkutan
  terpilih di gambar. Sebaliknya, memilih atom di gambar menandai puncaknya.
- **Open JCAMP-DX** menumpangkan spektrum hasil pengukuran Anda (`.jdx`, `.dx`) di atas
  spektrum prediksi. Spektrum yang sama bisa dibuka dari [[File > Open Spectrum (JCAMP-DX)]].
- **Copy Peak List** menyalin daftar puncak dalam bentuk teks, dan **Export JCAMP-DX**
  menyimpan spektrum prediksi.

> [!TIP] Proton yang dapat dipertukarkan (OH, NH, COOH) sangat bergantung pada pelarut dan
> konsentrasi, sehingga posisinya di spektrum prediksi hanya perkiraan kasar. Bandingkan
> selalu dengan spektrum terukur.

## Contoh lengkap: skema reaksi sintesis aspirin

Contoh ini membuat skema reaksi asetilasi asam salisilat dengan anhidrida asetat menjadi
aspirin dan asam asetat, lalu memeriksa kesetaraan dan stoikiometrinya, memetakan atomnya,
dan mengekspor gambarnya untuk naskah.

### Langkah 1: menggambar reaktan dan produk

1. Buka dokumen baru. Jika naskah Anda untuk jurnal ACS, pilih dulu
   [[File > Open Style Sheets > ACS Document 1996]] agar ukuran ikatan dan hurufnya sesuai.
2. Gambar asam salisilat: letakkan cincin **Benzene**, tambahkan dua ikatan dengan **Solid
   Bond**, lalu beri label `CO2H` dan `OH` (arahkan penunjuk ke atom, tekan ++Enter++, ketik
   labelnya). Cara lain, tempelkan SMILES `OC(=O)c1ccccc1O` lewat
   [[Structure > Insert Structure Text (SMILES, MOL, InChI)]].
3. Dengan cara yang sama, gambar anhidrida asetat di sebelah kanannya, lalu aspirin dan asam
   asetat lebih ke kanan lagi. Tanda plus boleh dilewati karena akan ditambahkan otomatis pada
   langkah 3.

> [!TIP] Setelah satu struktur dipilih, ++Ctrl+Right++ ([[Edit > Duplicate Across Arrow]])
> menambahkan panah reaksi beserta salinan struktur di seberangnya. Salinan itu tinggal Anda
> ubah menjadi produk.

### Langkah 2: panah dan kondisi reaksi

1. Buka fly-out panah di Main Toolbar dan pilih **Reaction Arrow**. Seret dari kanan reaktan
   terakhir ke kiri produk pertama.
2. Pilih alat **Marquee**, lalu klik dua kali panah tersebut. Jendela **Arrow Properties**
   terbuka.
3. Isi **Above the arrow** dengan `H2SO4` dan **Below the arrow** dengan `85 °C`. Angka dalam
   rumus otomatis menjadi subskrip. Gunakan `\n` untuk membuat baris baru. Klik **OK**.

![chem.ex_arrow]

### Langkah 3: merapikan dan memeriksa bacaan reaksi

1. Klik [[Structure > Clean Up Reaction]]. Jarak antarkomponen dibuat sama, tanda plus
   ditambahkan di antara reaktan dan di antara produk, dan panah dipanjangkan agar kondisi
   reaksi muat. Setelah itu klik [[Object > Center on Page]].
2. Aktifkan [[View > Show Reaction Interpretation]] untuk memastikan reaksi dibaca dengan
   benar: reaktan diberi kotak merah, produk magenta, dan panah beserta kondisinya biru.
   Warna ini hanya bantuan tampilan dan tidak ikut tercetak atau diekspor.

![chem.ex_reaction]

### Langkah 4: memeriksa kesetaraan reaksi

Klik [[Structure > Check Reaction Balance]]. Tabel membandingkan jumlah setiap unsur di ruas
reaktan dan produk. Kolom **Difference** yang tidak nol ditandai merah, dan aplikasi
menyarankan koefisien yang menyetarakan reaksi. Pada contoh ini reaksi sudah setara
(C₁₁H₁₂O₆ di kedua ruas). Klik **Send to Output Viewer** untuk menyimpan tabelnya bersama hasil
analisis lain.

![chem.ex_balance]

### Langkah 5: menghitung stoikiometri

1. Klik [[Structure > Reaction Stoichiometry]]. Setiap reaktan dan produk menjadi satu baris
   lengkap dengan rumus, berat molekul, dan koefisiennya.
2. Pada baris asam salisilat, isi **Amount** dengan `1.38` dan pilih satuan **g**. Pada baris
   anhidrida asetat, isi `3.06` g.
3. Aplikasi langsung menghitung mmol, ekuivalen, dan massa. Asam salisilat (9,99 mmol) menjadi
   **limiting reagent**, anhidrida asetat 3,00 ekuivalen, dan hasil teoretis aspirin
   1800 mg (1,80 g).

![chem.ex_stoich]

Hasil teoretis inilah yang Anda pakai untuk menghitung rendemen: misalnya jika diperoleh
1,53 g aspirin, rendemennya 1,53 / 1,80 × 100% = 85%.

### Langkah 6 (opsional): pemetaan atom

[[Structure > Map Reaction Atoms]] mencocokkan atom reaktan dengan atom produk secara otomatis
menggunakan toolkit Indigo. Setiap atom yang dipetakan diberi nomor peta kecil, dan nomor yang
sama di kedua ruas menandai atom yang dianggap sama. Pemetaan ini berguna untuk berkas RXN yang
dibaca program lain dan untuk menelusuri asal setiap atom. Hapus nomor peta dengan
[[Structure > Clear Reaction Map]].

> [!WARNING] Pemetaan otomatis adalah tebakan algoritma, bukan mekanisme reaksi. Periksa
> hasilnya. Nomor peta satu atom bisa diubah lewat [[Structure > Atom Properties]] (kotak
> **Atom map number**) atau dengan alat **Atom-to-Atom Map Number** di fly-out simbol kimia (klik atom untuk memberi atau menghapus nomor peta).

### Langkah 7: menyimpan dan mengekspor gambar untuk naskah

1. Simpan dokumen dengan [[File > Save As]]. Pilih `.rsc` untuk arsip Anda sendiri, atau
   `.cdxml`/`.cdx` jika jurnal atau rekan meminta berkas ChemDraw.
2. Pilih [[File > Export > PNG Image]] (atau TIFF), tentukan nama berkas, lalu pilih
   **Resolution** 600 dpi. Matikan **Transparent background** jika gambar akan dipakai di
   dokumen berlatar putih. Klik **OK**.
3. Untuk gambar vektor yang tetap tajam saat diperbesar, gunakan **SVG Image** atau **PDF**.
   Jika ada objek yang dipilih, hanya pilihan itu yang diekspor.

![chem.ex_export]

> [!TIP] Untuk menempelkan langsung ke Word atau PowerPoint, pilih reaksinya lalu tekan
> ++Ctrl+C++, atau gunakan [[Edit > Copy As > Picture (PNG, 300 dpi)]]. Kebanyakan jurnal
> meminta minimal 300 dpi untuk gambar garis. Ekspor 600 dpi memberi cadangan jika gambar perlu
> diperbesar.

## Style sheet dan pengaturan dokumen

Style sheet menentukan panjang ikatan, tebal garis, jarak ikatan rangkap, serta jenis dan
ukuran huruf. Gaya bawaannya adalah **ACS Document 1996**, dan tersedia pula gaya untuk RSC,
Wiley, Synthesis/Synlett, dan jurnal lain.

- [[File > Open Style Sheets]] membuka dokumen baru dengan gaya tersebut, dan gaya ini menjadi
  bawaan untuk dokumen baru berikutnya.
- [[File > Apply Document Settings from]] mengganti gaya dokumen yang sedang dibuka. Gambar
  diskalakan ke panjang ikatan yang baru.
- [[File > Document Settings]] mengatur setiap nilai secara manual, termasuk **Color atom labels
  by element**.

![chem.styles]

## Objek gambar lain

Selain struktur, Main Toolbar menyediakan objek untuk skema dan ilustrasi:

- **Panah**: panah reaksi, kesetimbangan (seimbang dan tidak seimbang), retrosintesis,
  resonansi, *no-go*, block arrow, busur, dipol, serta panah lengkung dan *fishhook* untuk
  mekanisme. Ujung panah lengkung menempel ke pasangan elektron, atom, atau ikatan dan ikut
  bergerak bersamanya.
- **Kurung**: kurung siku, biasa, dan kurawal. Klik dua kali sebuah kurung untuk membuka
  **Bracket Properties** (jumlah ulang *n*, jenis polimer, pola ulang).
- **Simbol kimia**: muatan, radikal, pasangan elektron bebas, δ+/δ−, ‡.
- **Orbital** s, p, d, dan hibrida, serta **Pen** untuk kurva bebas yang gayanya diatur di
  menu **Curves**.
- **Bentuk** (kotak, lingkaran, busur), **pelat KLT dan gel elektroforesis** (klik lajur untuk
  menambah noda, seret noda untuk mengatur Rf), dan **alat gelas** laboratorium (*Clipware*).

![chem.objects]

Warna objek terpilih diatur di menu **Colors**, dan gaya teks di menu **Text** atau Style
Toolbar. Perintah selengkapnya ada di Referensi Fitur.

## Format berkas

| Keperluan | Format |
|---|---|
| Menyimpan pekerjaan sendiri | `.rsc` (RS-StatChem drawing) |
| Bertukar dengan pengguna ChemDraw | `.cdx`, `.cdxml` |
| Bertukar dengan program kimia lain | `.mol` (V2000/V3000), `.sdf`, `.rxn`, SMILES, InChI |
| Peptida dan asam nukleat | HELM, FASTA |
| Gambar untuk naskah dan presentasi | PNG, TIFF, JPEG, BMP (150 sampai 1200 dpi), SVG, PDF |

Semua ekspor ada di menu [[File > Export]]. Hasilnya bisa juga disalin ke clipboard lewat
[[Edit > Copy As]]. [[File > Send to Output Viewer]] mengirim gambar struktur beserta
sifat-sifatnya ke Output Viewer agar bisa digabung dengan hasil analisis statistik.

## Referensi Fitur

Berikut semua perintah di menu modul Chemical Drawing beserta fungsinya.

!features chem
