// ================================================
// AUTO-GENERATED KNOWLEDGE GRAPH IMPORT SCRIPT
// ================================================
 
// === CREATE NODES ===
CREATE (:Hama {nama: "wereng"});
CREATE (:Hama {nama: "penggerek"});
CREATE (:Hama {nama: "ulat"});
CREATE (:Hama {nama: "wereng batang coklat"});
CREATE (:Hama {nama: "wereng hijau"});
CREATE (:Hama {nama: "penggerek batang"});
CREATE (:Hama {nama: "ulat grayak"});
CREATE (:Hama {nama: "ulat penggulung daun"});
CREATE (:Hama {nama: "hama pelipat daun"});
CREATE (:Hama {nama: "walang sangit"});
CREATE (:Hama {nama: "tikus"});
CREATE (:Hama {nama: "tikus sawah"});
CREATE (:Hama {nama: "keong mas"});
CREATE (:Hama {nama: "kutu putih"});
CREATE (:Hama {nama: "belalang"});
CREATE (:Hama {nama: "nezara viridula"});
CREATE (:Hama {nama: "nilaparvata lugens"});
CREATE (:Hama {nama: "cnaphalocrocis medinalis"});
CREATE (:Hama {nama: "scirpophaga incertulas"});
CREATE (:Penyakit {nama: "blas"});
CREATE (:Penyakit {nama: "penyakit blas"});
CREATE (:Penyakit {nama: "pyricularia oryzae"});
CREATE (:Penyakit {nama: "hawar daun bakteri"});
CREATE (:Penyakit {nama: "xanthomonas oryzae pv. oryzae"});
CREATE (:Penyakit {nama: "cercospora"});
CREATE (:Penyakit {nama: "busuk pelepah"});
CREATE (:Penyakit {nama: "busuk batang"});
CREATE (:Penyakit {nama: "busuk akar"});
CREATE (:Penyakit {nama: "busuk biji"});
CREATE (:Penyakit {nama: "busuk bulir"});
CREATE (:Penyakit {nama: "virus tungro"});
CREATE (:Penyakit {nama: "tungro"});
CREATE (:Penyakit {nama: "virus kerdil rumput"});
CREATE (:Penyakit {nama: "virus kerdil hampa"});
CREATE (:Penyakit {nama: "karat daun"});
CREATE (:Penyakit {nama: "jamur"});
CREATE (:Penyakit {nama: "bakteri"});
CREATE (:Penyakit {nama: "hawar daun"});
CREATE (:Gejala {nama: "klorosis"});
CREATE (:Gejala {nama: "menguning"});
CREATE (:Gejala {nama: "pucat"});
CREATE (:Gejala {nama: "kering"});
CREATE (:Gejala {nama: "layu"});
CREATE (:Gejala {nama: "menggulung"});
CREATE (:Gejala {nama: "bintik putih"});
CREATE (:Gejala {nama: "bintik coklat"});
CREATE (:Gejala {nama: "bercak putih"});
CREATE (:Gejala {nama: "bercak coklat"});
CREATE (:Gejala {nama: "bercak air"});
CREATE (:Gejala {nama: "bercak mata"});
CREATE (:Gejala {nama: "bercak hitam"});
CREATE (:Gejala {nama: "bercak ungu"});
CREATE (:Gejala {nama: "oval"});
CREATE (:Gejala {nama: "gosong"});
CREATE (:Gejala {nama: "patah"});
CREATE (:Gejala {nama: "hampa"});
CREATE (:Gejala {nama: "kosong"});
CREATE (:Gejala {nama: "mati anakan"});
CREATE (:Gejala {nama: "robek"});
CREATE (:Gejala {nama: "membusuk"});
CREATE (:Gejala {nama: "bercendawan"});
CREATE (:Gejala {nama: "keriting"});
CREATE (:Gejala {nama: "terbakar"});
CREATE (:Gejala {nama: "kering sebagian"});
CREATE (:Gejala {nama: "keropos"});
CREATE (:Gejala {nama: "rapuh"});
CREATE (:Gejala {nama: "berlendir"});
CREATE (:Gejala {nama: "kerdil"});
CREATE (:Gejala {nama: "lemah"});
CREATE (:Gejala {nama: "busuk"});
CREATE (:Gejala {nama: "tekuk"});
CREATE (:Gejala {nama: "lubang"});
CREATE (:Gejala {nama: "retak"});
CREATE (:Gejala {nama: "hitam"});
CREATE (:Gejala {nama: "bergelombang"});
CREATE (:Gejala {nama: "bengkok"});
CREATE (:Gejala {nama: "melepuh"});
CREATE (:Gejala {nama: "melengkung"});
CREATE (:Gejala {nama: "kusam"});
CREATE (:Gejala {nama: "memudar"});
CREATE (:Gejala {nama: "mozaik"});
CREATE (:Gejala {nama: "luka"});
CREATE (:BagianTanaman {nama: "daun"});
CREATE (:BagianTanaman {nama: "helai daun"});
CREATE (:BagianTanaman {nama: "tulang daun"});
CREATE (:BagianTanaman {nama: "pelepah"});
CREATE (:BagianTanaman {nama: "batang"});
CREATE (:BagianTanaman {nama: "batang muda"});
CREATE (:BagianTanaman {nama: "pangkal batang"});
CREATE (:BagianTanaman {nama: "malai"});
CREATE (:BagianTanaman {nama: "bulir"});
CREATE (:BagianTanaman {nama: "akar"});
CREATE (:BagianTanaman {nama: "anakan"});
 
// === CREATE RELATIONS ===
// ⚠️ SKIPPED: 'wereng batang coklat' atau 'tanaman' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "klorosis"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "hampa"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'penyakit garis coklat' atau 'bercak' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'penyakit garis coklat' atau 'coklat' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "penggerek batang"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "patah"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:Gejala {nama: "busuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "klorosis"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'cercospora' atau 'pada' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "hama pelipat daun"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "hama pelipat daun"}), (b:Gejala {nama: "menggulung"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:Gejala {nama: "busuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'keong mas' atau 'terpotong' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "tungro"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "tungro"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus sawah"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "tikus sawah"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "bercak air"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "hawar daun"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "bergelombang"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "tungro"}), (b:Gejala {nama: "klorosis"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'penggerek batang' atau 'beluk' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "ulat grayak"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "lubang"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "wereng"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'blas' atau 'ujung' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "tungro"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'tungro' atau 'tidak' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "keong mas"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "kutu putih"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "kutu putih"}), (b:Gejala {nama: "bercak putih"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:Gejala {nama: "patah"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'chlorotic' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "penggerek batang"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'penggerek batang' atau 'bergaris' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'Xanthomonas oryzae pv. oryzae' atau 'bercak' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'Xanthomonas oryzae pv. oryzae' atau 'air' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "walang sangit"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "pucat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:BagianTanaman {nama: "anakan"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'wereng hijau' atau 'tidak' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "keong mas"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:Gejala {nama: "rapuh"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "lubang"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:Gejala {nama: "luka"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "tungro"}), (b:Gejala {nama: "kusam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'ulat grayak' atau 'bergerigi' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'wereng batang coklat' atau 'mati' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "blas"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'blas' atau 'keluar' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "mozaik"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'tikus' atau 'gagal' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "xanthomonas oryzae pv. oryzae"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'Xanthomonas oryzae pv. oryzae' atau 'blight' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:BagianTanaman {nama: "pangkal batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:Gejala {nama: "busuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "kosong"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'tungro' atau 'chlorosis' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'wereng batang coklat' atau 'ujung' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "menggulung"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'cercospora' atau 'berbintik' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "tikus sawah"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "tikus sawah"}), (b:Gejala {nama: "patah"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'berubah' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'blas' atau 'tidak' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "tikus"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'keong mas' atau 'terkelupas' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:BagianTanaman {nama: "pangkal batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:Gejala {nama: "busuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'memucat' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat"}), (b:Gejala {nama: "robek"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "kutu putih"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "kutu putih"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'blas' atau 'menunjukkan' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "hawar daun"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "hawar daun"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:Gejala {nama: "bengkok"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'keong mas' atau 'putus' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'walang sangit' atau 'kurang' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "tikus"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'tikus' atau 'terkelupas' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "penggerek batang"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'wereng batang coklat' atau 'merunduk' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "tungro"}), (b:Gejala {nama: "pucat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'wereng' atau 'oplos' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "hama pelipat daun"}), (b:Gejala {nama: "melengkung"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:Gejala {nama: "patah"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "kosong"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'penggerek batang' atau 'bolong' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "robek"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'wereng' atau 'garis' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus tungro"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'gagal' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "kutu putih"}), (b:Gejala {nama: "lubang"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "karat daun"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'karat daun' atau 'berkarat' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk batang"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'busuk batang' atau 'mengeluarkan' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:Gejala {nama: "bercendawan"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'virus kerdil rumput' atau 'tidak' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'karat daun' atau 'pustula' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "pucat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Gejala {nama: "bercendawan"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'salurannya' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "jamur"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'busuk pelepah' atau 'mudah' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:Gejala {nama: "luka"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'jamur' atau 'berwarna' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk batang"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk batang"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'cercospora' atau 'putih' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus tungro"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk batang"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'penggerek batang' atau 'garis' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:Gejala {nama: "keriting"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "gosong"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'cercospora' atau 'terbelah' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'karat daun' atau 'timbul' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'tumbuh' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'virus tungro' atau 'bercahaya' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "jamur"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:Gejala {nama: "busuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'ulat grayak' atau 'lecek' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'penggerek batang' atau 'kalah' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'karat daun' atau 'bintik' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "hawar daun"}), (b:Gejala {nama: "gosong"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "kutu putih"}), (b:Gejala {nama: "robek"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'cercospora' atau 'koyak' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'walang sangit' atau 'tidak' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'wereng hijau' atau 'timbul' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk pelepah"}), (b:Gejala {nama: "patah"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Gejala {nama: "busuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:Gejala {nama: "keropos"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'walang sangit' atau 'berwarna' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'virus tungro' atau 'kaku' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'cercospora' atau 'noda' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk batang"}), (b:Gejala {nama: "busuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'geser' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "retak"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:Gejala {nama: "keriting"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:Gejala {nama: "kerdil"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'busuk pelepah' atau 'lembek' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "wereng"}), (b:Gejala {nama: "bergelombang"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'kutu putih' atau 'berbintik' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "hampa"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun"}), (b:Gejala {nama: "berlendir"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk batang"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'jamur' atau 'buat' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'penggerek batang' atau 'melemah' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'wereng' atau 'berubah' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:BagianTanaman {nama: "malai"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'virus kerdil rumput' atau 'kurus' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'busuk akar' atau 'lapuk' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'tikus' atau 'hancur' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk akar"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'jamur' atau 'berjamur' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'busuk batang' atau 'benyek' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "oval"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'walang sangit' atau 'merunduk' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "keriting"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'wereng' atau 'encok' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'busuk akar' atau 'gampang' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "luka"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "luka"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'cercospora' atau 'ditumbuhi' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "melengkung"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "keropos"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:Gejala {nama: "hampa"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "bercendawan"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'ulang grayak' atau 'batang' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'ulang grayak' atau 'lubang' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'kutu putih' atau 'sobek' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "hawar daun"}), (b:Gejala {nama: "melepuh"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "hama pelipat daun"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "xanthomonas oryzae pv. oryzae"}), (b:Gejala {nama: "bercak air"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'virus kerdil rumput' atau 'berkerut' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "wereng"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'tikus sawah' atau 'hancur' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "belalang"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "belalang"}), (b:Gejala {nama: "lubang"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
// ⚠️ SKIPPED: 'penggerek batang' atau 'lukanya' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'kutu putih' atau 'titik' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'virus tungro' atau 'merah' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'tikus' atau 'melemah' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "retak"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'virus tungro' atau 'bertepi' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "busuk akar"}), (b:Gejala {nama: "rapuh"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "ulat"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat"}), (b:Gejala {nama: "keriting"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'kutu putih' atau 'bintik' tidak ditemukan di tokens.json
// ⚠️ SKIPPED: 'hawar daun' atau 'berguguran' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "ulat"}), (b:Gejala {nama: "menggulung"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:Gejala {nama: "lemah"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "oval"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "tekuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus sawah"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "tikus sawah"}), (b:Gejala {nama: "lubang"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "rapuh"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "bercak mata"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "pucat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "retak"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:Gejala {nama: "tekuk"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "tikus"}), (b:Gejala {nama: "rapuh"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'cercospora' atau 'bintik' tidak ditemukan di tokens.json
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "bercak putih"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:BagianTanaman {nama: "helai daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus kerdil rumput"}), (b:Gejala {nama: "kosong"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
// ⚠️ SKIPPED: 'busuk akar' atau 'lusuh' tidak ditemukan di tokens.json
MATCH (a:Hama {nama: "ulat"}), (b:Gejala {nama: "keropos"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:BagianTanaman {nama: "pelepah"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "blas"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Penyakit {nama: "hawar daun bakteri"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "busuk akar"}), (b:BagianTanaman {nama: "daun"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "keong mas"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "ulat grayak"}), (b:Gejala {nama: "menguning"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "penggerek batang"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Gejala {nama: "kering"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "hitam"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "virus tungro"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:BagianTanaman {nama: "batang"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:BagianTanaman {nama: "akar"}) CREATE (a)-[:MENYERANG]->(b);
MATCH (a:Hama {nama: "walang sangit"}), (b:Gejala {nama: "bercak coklat"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "cercospora"}), (b:Gejala {nama: "layu"}) CREATE (a)-[:MEMILIKI_GEJALA]->(b);
MATCH (a:Penyakit {nama: "pyricularia oryzae"}), (b:Penyakit {nama: "blas"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Penyakit {nama: "xanthomonas oryzae pv. oryzae"}), (b:Penyakit {nama: "hawar daun bakteri"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:Penyakit {nama: "virus tungro"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Hama {nama: "wereng hijau"}), (b:Penyakit {nama: "virus kerdil rumput"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Hama {nama: "wereng batang coklat"}), (b:Penyakit {nama: "virus kerdil hampa"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Penyakit {nama: "busuk pelepah"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Penyakit {nama: "busuk batang"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Penyakit {nama: "busuk akar"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Penyakit {nama: "cercospora"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Penyakit {nama: "jamur"}), (b:Penyakit {nama: "karat daun"}) CREATE (a)-[:MENYEBABKAN]->(b);
MATCH (a:Hama {nama: "nilaparvata lugens"}), (b:Hama {nama: "wereng batang coklat"}) CREATE (a)-[:NAMA_ILMIAH]->(b);
MATCH (a:Hama {nama: "cnaphalocrocis medinalis"}), (b:Hama {nama: "hama pelipat daun"}) CREATE (a)-[:NAMA_ILMIAH]->(b);
MATCH (a:Hama {nama: "scirpophaga incertulas"}), (b:Hama {nama: "penggerek batang"}) CREATE (a)-[:NAMA_ILMIAH]->(b);
MATCH (a:Hama {nama: "nezara viridula"}), (b:Hama {nama: "walang sangit"}) CREATE (a)-[:NAMA_ILMIAH]->(b);