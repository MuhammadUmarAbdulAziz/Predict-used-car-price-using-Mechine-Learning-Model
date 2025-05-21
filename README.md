# Predict-used-car-price-using-Mechine-Learning-Model
## Project Capstone Modul 3 Purwadhika : Machine Learning

**By: Risdan Kristori**


## Repository Structure
- <b>README.md:</b> Deskripsi Project
- <b>Predict Saudi Used Cars Prise Using Mechine Learning.ipynb:</b> Notebook Project

## Introduction
# **Business Problem**

**Saudi Arabia Market Opportunity**

Kerajaan Arab Saudi menunjukkan pertumbuhan signifikan dalam industri jual beli mobil bekas, didorong oleh berbagai faktor ekonomi, demografis, dan teknologi. Menurut laporan Ken Research, nilai transaksi penjualan mobil bekas di Arab Saudi diproyeksikan mengalami pertumbuhan tahunan sebesar 4,5% pada periode 2019 hingga 2025 .

Faktor-faktor yang mendorong pertumbuhan ini antara lain meningkatnya pendapatan masyarakat, pertumbuhan populasi ekspatriat, serta peningkatan jumlah pengemudi perempuan setelah pelonggaran regulasi. Selain itu, digitalisasi memainkan peran penting dalam transformasi pasar ini. Platform online seperti Syarah.com, Haraj, dan Motory memudahkan konsumen dalam mencari dan membeli mobil bekas dengan menyediakan informasi detail, opsi pembiayaan, dan layanan inspeksi kendaraan .

Peningkatan kualitas kendaraan bekas melalui program sertifikasi juga meningkatkan kepercayaan konsumen. Program ini menawarkan jaminan kualitas dan layanan purna jual, menjadikan mobil bekas pilihan yang menarik bagi konsumen yang mencari nilai lebih dengan harga terjangkau . Secara keseluruhan, kombinasi dari faktor-faktor tersebut menjadikan pasar mobil bekas di Arab Saudi sebagai sektor yang menjanjikan dengan potensi pertumbuhan yang berkelanjutan.

**Used Car Price Problem**

Salah satu permasalahan utama dalam bisnis jual beli mobil bekas di Arab Saudi adalah ketidakpastian harga jual yang wajar. Dikutip dari mordorintelligence.com (Faisal Al-Qasabi, 2024), harga mobil bekas sangat bergantung pada banyak faktor seperti merek, model, tahun pembuatan, jarak tempuh, jenis bahan bakar, serta kondisi fisik dan kelengkapan dokumen. Baik penjual maupun pembeli yang tidak memiliki informasi yang cukup sering kali mengalami kebingungan dalam menentukan harga yang sesuai. Penjual yang menetapkan harga terlalu tinggi berisiko tidak berhasil menjual kendaraannya, sementara penjual yang memasang harga terlalu rendah akan mengalami kerugian. Sebaliknya, pembeli yang tidak memahami nilai pasar kendaraan bisa saja membeli mobil dengan harga jauh di atas nilai sebenarnya, yang tentu merugikan dari sisi finansial. Dikutip dari marketresearch.com (Ken Research, 2024), kurangnya transparansi harga menjadi hambatan besar dalam pertumbuhan pasar mobil bekas secara digital di Arab Saudi.

**Problem Statement**

Penjual dan pembeli mobil bekas yang tidak memiliki pemahaman cukup mengenai harga pasar mobil, akan kesulitan menentukan nilai jual atau beli yang tepat. Hal ini menyebabkan potensi kerugian finansial atau tidak terjadinya transaksi karena adanya ketidakpastian dalam menentukan harga.

**Goals**

Berdasarkan permasalahan tersebut, dibutuhkan pengembangan sebuah model prediktif berbasis data untuk memperkirakan harga mobil bekas berdasarkan spesifikasi kendaraan. Model ini bertujuan memberikan estimasi harga yang akurat sehingga dapat dijadikan acuan oleh penjual dan pembeli saat bertransaksi. Dengan adanya estimasi harga yang objektif, proses pengambilan keputusan menjadi lebih cepat dan efisien, serta meningkatkan kepercayaan terhadap platform digital jual beli mobil bekas seperti Syarah.com. Pada akhirnya, hal ini diharapkan dapat mendorong peningkatan volume transaksi dan mendukung pertumbuhan industri mobil bekas secara keseluruhan.

**Analytic Approach**

Pendekatan yang digunakan adalah analisis data spesifikasi kendaraan yang mempengaruhi harga mobil bekas, seperti merek, tipe, tahun produksi, kapasitas mesin, dan jumlah kilometer yang telah ditempuh. Setelah variabel-variabel tersebut dianalisis, dilakukan pemodelan regresi menggunakan pendekatan machine learning. Beberapa algoritma seperti regresi linier, decision tree, dan random forest akan dibandingkan untuk menentukan model dengan performa terbaik berdasarkan metrik evaluasi seperti RMSE dan MAE. Model dengan performa terbaik akan dijadikan model akhir yang dapat diintegrasikan ke dalam platform jual beli mobil online untuk membantu pengguna menentukan harga jual atau beli yang optimal.

**Matrix Evalution**

Matrix yang akan digunakan untuk model ini adalah:

- Root Mean Squared Error (rmse), atau
- Mean Absolute Error (MAE)

Matrix yang diperlukan akan ditentukan setelah kita melihat kareakteristik data yang akan digunakan.

**source:**

- Ahmed Al-Mutairi. 2024. Used Car Market in Saudi Arabia Expected to Grow as Buyers Go Digital. Riyadh: Zawya.com.

- Ken Research. 2024. KSA Used Cars Market Outlook to 2028. Riyadh: Ken Research.

- Mordor Intelligence. 2024. Used Car Market in Saudi Arabia - Size, Share & Forecast. Riyadh: Mordor Intelligence.

- Verified Market Research. 2025. Saudi Arabia Used Car Market Size By Vehicle Type, Sales Channel, And Region For 2025–2032. Riyadh: Verified Market Research.

- Faisal Al-Qasabi. 2024. Used Car Market Challenges and Digital Transformation in Saudi Arabia. Riyadh: Mordor Intelligence.

- Bonafide Research. 2024. Saudi Arabia Used Car Market Overview, 2029. Riyadh: Bonafide Research.

- Grand View Research. 2024. Saudi Arabia Used Car Market Size & Outlook, 2024-2030. Riyadh: Grand View Research.

# **DATA Understanding**

**Data Information**

- Data yang digunakan merupakan data spesifikasi/fitur mobil bekas beserta beserta harga jualnya yang telah disediakan oleh Purwadhika. Dilihat dari bentuk dan pola data, diasumsikan bahwa data ini diambil dari website Syarah.com yang menjual mobil bekas di Saudi Arabia.
- Data 'price'/harga untuk mobil bekas yang sifatnya 'negotiable' bernilai 0 (nol). Harga berdasarkan kesepakatan antara penjual dan pembeli.

**Feature**
| Attribute | Data Type| Description |
| --- | --- | --- |
| Type | Object | Type of used car |
| Region | Object | The region in which the used car was offered for sale |
| Make | Object | Name of the car company |
| Gear_Type | Object | Automatic / Manual |
| Origin | Object | Country of importer (Gulf / Saudi / Other) |
| Options | Object | Full Options / Semi-Full / Standard |
| Year | Int | Year of Manufacturing |
| Enginee_Size | Float | The engine size of used car |
| Mileage | Int | The distance that used car have travelled, measured in miles |
| Negotiable | Bool | If True, the price is 0. This means the price is negotiable (not set) |
| Price | int | Price of the used car (in SAR) |

## Exploratory Data Analysis

<img src="Picture/Screenshot 2025-05-20 210134.png" alt="isolated" width="700"/>
Distribusi data untuk variabel Mileage (jarak tempuh) dan Price (harga) pada mobil bekas umumnya tidak mengikuti distribusi normal. Kedua variabel ini cenderung memiliki distribusi miring ke kanan (right-skewed), yang berarti sebagian besar mobil memiliki harga dan jarak tempuh yang relatif rendah, sementara sebagian kecil lainnya memiliki nilai yang sangat tinggi. Hal ini wajar terjadi dalam pasar mobil bekas karena banyak mobil yang masih dalam kondisi baik dengan harga terjangkau beredar lebih banyak, sedangkan hanya sebagian kecil mobil yang berharga sangat mahal atau memiliki jarak tempuh ekstrem. Distribusi semacam ini dapat memengaruhi kinerja model prediksi jika tidak ditangani dengan benar, karena model regresi dan algoritma statistik lainnya sering kali mengasumsikan bahwa data berdistribusi mendekati normal.

Dikutip dari Brownlee (2020), data dengan distribusi miring ke kanan sering muncul pada domain harga dan volume karena sifat alamiah dari nilai yang terbatas di bawah (tidak bisa bernilai negatif) namun tidak terbatas di atas (bisa sangat tinggi). Dalam analisis data, penting untuk mendeteksi dan menangani distribusi semacam ini untuk meningkatkan akurasi dan stabilitas model (machine learning) yang digunakan.

**Referensi**:

- Brownlee, Jason. 2020. Better Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms. Australia: Machine Learning Mastery.

- Shmueli, Galit, et al. 2016. Data Mining for Business Analytics: Concepts, Techniques, and Applications in R. Hoboken: Wile

<img src="Picture/Screenshot 2025-05-20 210825.png" alt="isolated" width="700"/>
Berdasarkan hasil analisis korelasi, terdapat beberapa hubungan penting antara variabel numerik yang memengaruhi harga mobil bekas. Mobil dengan tahun produksi lebih baru cenderung memiliki harga yang lebih tinggi, meskipun hubungan ini tidak terlalu kuat. Sebaliknya, jarak tempuh memiliki korelasi negatif dengan harga, yang berarti semakin besar jarak tempuh mobil, harga jualnya cenderung menurun. Ini mencerminkan kekhawatiran calon pembeli terhadap tingkat keausan kendaraan. Selain itu, mobil dengan ukuran mesin yang lebih besar biasanya dihargai lebih mahal karena umumnya dikaitkan dengan kendaraan tipe premium atau SUV yang menawarkan performa lebih tinggi. Namun, secara keseluruhan, kekuatan korelasi antar variabel terhadap harga tergolong lemah, menunjukkan bahwa ada banyak faktor non-numerik lain seperti kondisi fisik mobil, riwayat perawatan, dan lokasi penjualan yang mungkin lebih berperan dalam menentukan nilai jual mobil bekas.

Dikutip dari Shmueli et al. (2016), korelasi yang rendah antara fitur numerik dan target prediksi seperti harga merupakan indikasi bahwa model prediksi harus mempertimbangkan kombinasi variabel dan pendekatan non-linear untuk mendapatkan hasil yang lebih akurat.

**Referensi**:
- Shmueli, Galit, et al. 2016. Data Mining for Business Analytics: Concepts, Techniques, and Applications in R. Hoboken: Wiley.

- James, Gareth, et al. 2013. An Introduction to Statistical Learning: with Applications in R. New York: Springer.

- Ling, Charles X., and Qiang Yang. 2011. Machine Learning for Prediction and Analysis. San Francisco: Morgan Kaufmann.

**Modeling**

Untuk memprediksi harga mobil bekas (Price), digunakan 9 model machine learning, yang terdiri dari 5 model dasar (base models) dan 4 model gabungan (ensemble models). Model dasar yang digunakan meliputi **Linear Regression**, **KNN Regressor**, **Decision Tree Regressor**, **Ridge Regression**, **Lasso Regression**, dan **Elastic Net**. Sementara itu, model gabungan atau ensembel model yang digunakan adalah **Random Forest Regressor**, **AdaBoost Regressor**, dan **Gradient Boosting Regressor**. Model-model ini dipilih karena mewakili berbagai pendekatan dalam mempelajari hubungan antara fitur kendaraan dan harga, mulai dari model sederhana berbasis linear hingga model kompleks yang mampu menangkap pola non-linear.

Dalam proses pemilihan model terbaik, digunakan metrik evaluasi **Mean Absolute Error (MAE)**. MAE dipilih karena lebih cocok digunakan pada data yang memiliki rentang nilai harga yang sangat lebar dan distribusi yang tidak normal (condong ke kanan atau positively skewed), seperti pada harga mobil bekas. Tidak seperti metrik lain yang sensitif terhadap outlier, MAE memberikan gambaran kesalahan rata-rata dalam satuan yang sama dengan target (harga), sehingga lebih mudah diinterpretasikan oleh pengguna akhir dan tidak terpengaruh secara ekstrem oleh nilai-nilai yang sangat tinggi.

Berikut adalah hasil evaluasi terhadap semua model terhadap **data train**
<img src="Picture/Screenshot 2025-05-20 135536.png" alt="isolated" width="700"/>
<img src="Picture/Screenshot 2025-05-21 191318.png" alt="isolated" width="700"/>
Berdasarkan hasil uji terhadap data test diatas, didapatkan bahwa model Ridge Regression memiliki score matrix yang terbaik, maka model ini yang akan digunakan untuk memprediksi harga mobil sewa.

## Result

Grafik dibawah adalah grafik perbandingan antara nilai aktual Price dengan nilai hasil prediksi dengan menggunakan model Gradient Boosting dengan parameter yang telah di tuning. Dari grafik tersebut beberapa hal yang dapat ditemukan sebagai berikut.

Grafik di atas menampilkan perbandingan antara nilai aktual (Actual Price) dan nilai prediksi (Predicted Price) yang dihasilkan oleh model Gradient Boosting Regressor dengan parameter yang telah dioptimalkan melalui proses hyperparameter tuning. Secara umum, pola persebaran titik menunjukkan hubungan yang cukup linear antara nilai aktual dan prediksi, yang mengindikasikan bahwa model mampu menangkap pola harga mobil bekas dengan cukup baik.

Namun, masih terlihat adanya beberapa titik yang menyimpang dari garis imajiner linear (y = x), terutama pada rentang harga tinggi. Hal ini menunjukkan bahwa model cenderung memiliki tingkat error yang lebih besar saat memprediksi mobil-mobil dengan harga sangat tinggi, kemungkinan disebabkan oleh jumlah data yang lebih sedikit pada segmen harga tersebut (data sparsity). Sementara itu, untuk mobil dengan harga yang lebih rendah dan menengah, prediksi model terlihat lebih rapat dan akurat.

Distribusi data yang tidak merata dan keberadaan outlier pada harga-harga ekstrem memberikan tantangan tersendiri dalam pemodelan, sehingga menjadi bahan evaluasi untuk pengolahan data dan pemilihan fitur ke depannya. Meskipun demikian, berdasarkan hasil visualisasi dan evaluasi metrik sebelumnya (seperti MAE), model ini sudah menunjukkan performa yang cukup baik dalam konteks regresi harga mobil bekas.

<img src="Picture/Screenshot 2025-05-20 142959.png" alt="isolated" width="700"/>

## Conclusion

<img src="Picture/Screenshot 2025-05-20 143938.png" alt="isolated" width="700"/>

Berdasarkan hasil evaluasi model menggunakan metrik Mean Absolute Error (MAE), didapatkan nilai MAE sebesar 18.344 SAR. Artinya, rata-rata selisih antara harga mobil bekas yang diprediksi oleh model dengan harga sebenarnya adalah sekitar 18.344 Saudi Riyal. Untuk mobil dengan harga tinggi, misalnya hingga 850.000 SAR, kesalahan prediksi ini relatif kecil (sekitar 2,2%) dan masih dapat ditoleransi. Namun, untuk mobil dengan harga rendah seperti 500 SAR, nilai kesalahan ini menjadi sangat besar secara persentase, bahkan mencapai lebih dari 3.000%.

Kondisi ini menunjukkan bahwa model memiliki performa yang baik secara umum, namun cenderung kurang akurat pada segmen harga mobil yang sangat rendah. Oleh karena itu, pengembangan lebih lanjut dapat dilakukan dengan memberikan batas nilai minimum harga mobil agar model tidak terdistraksi oleh outlier harga yang sangat kecil.

Secara keseluruhan, model ini cukup efektif untuk diaplikasikan dalam memprediksi harga mobil bekas. Model seperti ini dapat sangat bermanfaat bagi platform jual-beli mobil seperti Syarah.com, karena dapat membantu pengguna memperkirakan harga pasar kendaraan yang mereka incar. Dengan demikian, transparansi harga akan meningkat dan potensi transaksi di platform digital akan bertambah.

**Referensi**:
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.). O'Reilly Media.

- Raschka, S., & Mirjalili, V. (2019). Python Machine Learning (3rd ed.). Packt Publishing.

## Recomendation
Untuk meningkatkan kualitas model prediksi harga mobil bekas di masa mendatang, beberapa rekomendasi berikut dapat dipertimbangkan:

**1. Menerapkan Batasan Minimum Harga**

Berdasarkan hasil visualisasi Actual vs. Predicted Price, terlihat bahwa error paling besar terjadi pada mobil dengan harga sangat rendah. Oleh karena itu, disarankan untuk menetapkan batas minimum harga mobil, misalnya dengan mengecualikan data dengan harga di bawah ambang tertentu. Alternatif lainnya adalah dengan memisahkan model menjadi dua kelompok: satu model khusus untuk mobil dengan harga rendah, dan satu model lain untuk harga menengah hingga tinggi. Pendekatan ini akan meningkatkan akurasi masing-masing model sesuai segmentasi harga.

**2. Evaluasi Kolinearitas Fitur**

Hasil feature importance menunjukkan bahwa fitur-fitur seperti Year, Engine_Size, dan Mileage memberikan kontribusi dominan. Untuk memastikan bahwa tidak terjadi multikolinearitas yang dapat menurunkan stabilitas model, perlu dilakukan analisis Variance Inflation Factor (VIF) pada fitur numerik. Fitur dengan nilai VIF tinggi sebaiknya dievaluasi lebih lanjut atau dieliminasi agar model lebih efisien dan tidak overfit.

**3. Eksplorasi Fitur Non-Linear (Polinomial)**

Mengingat adanya kemungkinan hubungan non-linear antara fitur dengan target Price, disarankan untuk mencoba teknik transformasi polinomial atau feature interaction. Pendekatan ini memungkinkan model menangkap pola kompleks yang mungkin tidak terdeteksi dalam model linear atau boosting biasa. Namun, penambahan kompleksitas ini harus tetap diimbangi dengan evaluasi performa model agar tidak menyebabkan overfitting.

**4. Peningkatan Preprocessing pada Fitur Kategorikal**

Beberapa fitur kategorikal seperti Make, Type, dan Origin masih memiliki pengaruh yang signifikan. Anda dapat mengevaluasi metode encoding yang digunakan, misalnya mencoba target encoding atau frequency encoding sebagai alternatif dari one-hot/binary encoding, untuk mengurangi dimensi dan menangkap pola yang lebih informatif.

**5. Validasi Model Lebih Lanjut**

Terakhir, untuk memastikan model yang dibangun dapat digunakan secara umum, disarankan untuk menggunakan teknik validasi silang (cross-validation) yang lebih ketat. Teknik ini dapat memberikan estimasi performa model yang lebih stabil dan membantu menghindari bias akibat data yang tidak merata.
