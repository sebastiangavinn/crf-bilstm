import re
import json
from collections import defaultdict

def extract_relations_from_ner_dataset(file_path):
    """
    Ekstrak relasi dari dataset NER format:
    [BAGIAN_TANAMAN] [GEJALA] oleh/karena/akibat [HAMA/PENYAKIT]
    """
    
    # Dictionary untuk menyimpan relasi
    relations = []
    relation_set = set()  # Untuk menghindari duplikasi
    
    # Pola regex untuk menangkap berbagai format kalimat
    patterns = [
        # Pattern: Bagian gejala oleh/karena/akibat hama/penyakit
        r'(\w+)\s+([\w\s]+?)\s+(?:oleh|karena|akibat|diserang|serangan)\s+([\w\s]+?)(?:\s*\(|\.)',
        # Pattern alternatif
        r'(\w+)\s+(?:mengalami|menjadi|terlihat|tampak|muncul|ada)\s+([\w\s]+?)\s+(?:oleh|karena|akibat|diserang|serangan)\s+([\w\s]+?)(?:\s*\(|\.)',
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Bersihkan nama ilmiah dalam kurung
        line_clean = re.sub(r'\([^)]+\)', '', line)
        
        for pattern in patterns:
            match = re.search(pattern, line_clean, re.IGNORECASE)
            if match:
                bagian = match.group(1).strip().lower()
                gejala = match.group(2).strip().lower()
                penyebab = match.group(3).strip().lower()
                
                # Bersihkan gejala dari kata kerja bantu
                gejala = re.sub(r'^(mengalami|menjadi|terlihat|tampak|muncul|ada)\s+', '', gejala)
                
                # Normalisasi nama bagian tanaman
                bagian_map = {
                    'daun': 'daun',
                    'helai': 'helai daun',
                    'batang': 'batang',
                    'akar': 'akar',
                    'pelepah': 'pelepah',
                    'malai': 'malai',
                    'bulir': 'bulir',
                    'panicle': 'malai',
                    'tangkai': 'malai',
                    'tunas': 'anakan',
                    'pangkal': 'pangkal batang'
                }
                
                bagian_normalized = bagian_map.get(bagian, bagian)
                
                # Normalisasi gejala
                gejala_map = {
                    'klorosis': 'klorosis',
                    'menguning': 'menguning',
                    'mengering': 'kering',
                    'kering': 'kering',
                    'layu': 'layu',
                    'menggulung': 'menggulung',
                    'menghitam': 'hitam',
                    'hitam': 'hitam',
                    'berlubang': 'lubang',
                    'lubang': 'lubang',
                    'busuk': 'busuk',
                    'membusuk': 'busuk',
                    'patah': 'patah',
                    'hampa': 'hampa',
                    'kosong': 'kosong',
                    'bercak coklat': 'bercak coklat',
                    'bercak putih': 'bercak putih',
                    'bercak hitam': 'bercak hitam',
                    'bercak air': 'bercak air',
                    'bercak mata': 'bercak mata',
                    'bercak': 'bercak coklat',
                    'berbercak': 'bercak coklat',
                    'pucat': 'pucat',
                    'nekrosis': 'kering',
                    'bergelombang': 'bergelombang',
                    'melipat': 'menggulung',
                    'terlipat': 'menggulung',
                    'ngelipat': 'menggulung',
                    'robek': 'robek',
                    'rusak': 'luka',
                    'rapuh': 'rapuh',
                    'keriting': 'keriting',
                    'mozaik': 'mozaik',
                    'bengkok': 'bengkok',
                    'kerdil': 'kerdil',
                    'berlendir': 'berlendir',
                    'bercendawan': 'bercendawan',
                    'gosong': 'gosong',
                    'terbakar': 'gosong',
                    'kusam': 'kusam',
                    'lemas': 'layu',
                    'oval': 'oval',
                    'melepuh': 'melepuh'
                }
                
                # Cari gejala yang cocok
                gejala_normalized = None
                for key, value in gejala_map.items():
                    if key in gejala:
                        gejala_normalized = value
                        break
                
                if not gejala_normalized:
                    gejala_normalized = gejala.split()[0] if gejala else None
                
                # Normalisasi penyebab (hama/penyakit)
                penyebab_map = {
                    'wereng batang coklat': 'wereng batang coklat',
                    'wereng coklat': 'wereng batang coklat',
                    'wereng hijau': 'wereng hijau',
                    'wereng': 'wereng',
                    'blas': 'blas',
                    'walang sangit': 'walang sangit',
                    'walang': 'walang sangit',
                    'penggerek batang': 'penggerek batang',
                    'penggerek': 'penggerek batang',
                    'hawar daun bakteri': 'hawar daun bakteri',
                    'hawar daun': 'hawar daun',
                    'busuk akar': 'busuk akar',
                    'virus tungro': 'virus tungro',
                    'tungro': 'tungro',
                    'cercospora': 'cercospora',
                    'pelipat daun': 'hama pelipat daun',
                    'keong mas': 'keong mas',
                    'keong sawah': 'keong mas',
                    'tikus sawah': 'tikus sawah',
                    'tikus': 'tikus',
                    'ulat grayak': 'ulat grayak',
                    'ulat putih': 'kutu putih',
                    'kutu putih': 'kutu putih',
                    'xanthomonas oryzae': 'Xanthomonas oryzae pv. oryzae',
                    'xanthomonas': 'Xanthomonas oryzae pv. oryzae',
                    'busuk pelepah': 'busuk pelepah',
                    'busuk batang': 'busuk batang',
                    'karat daun': 'karat daun',
                    'jamur': 'jamur',
                    'virus kerdil': 'virus kerdil rumput',
                    'virus': 'virus tungro',
                    'bakteri': 'hawar daun bakteri',
                    'ulat': 'ulat',
                    'ulat penggulung daun': 'ulat penggulung daun',
                    'belalang': 'belalang'
                }
                
                penyebab_normalized = None
                for key, value in penyebab_map.items():
                    if key in penyebab:
                        penyebab_normalized = value
                        break
                
                if not penyebab_normalized:
                    penyebab_normalized = penyebab
                
                # Tambahkan relasi
                if bagian_normalized and penyebab_normalized:
                    # Relasi MENYERANG
                    rel1 = (penyebab_normalized, bagian_normalized, "MENYERANG")
                    if rel1 not in relation_set:
                        relations.append({
                            "from": penyebab_normalized,
                            "to": bagian_normalized,
                            "type": "MENYERANG"
                        })
                        relation_set.add(rel1)
                    
                    # Relasi MEMILIKI_GEJALA
                    if gejala_normalized:
                        rel2 = (penyebab_normalized, gejala_normalized, "MEMILIKI_GEJALA")
                        if rel2 not in relation_set:
                            relations.append({
                                "from": penyebab_normalized,
                                "to": gejala_normalized,
                                "type": "MEMILIKI_GEJALA"
                            })
                            relation_set.add(rel2)
                
                break
    
    return relations


def add_additional_relations(relations):
    """
    Tambahkan relasi tambahan berdasarkan pengetahuan domain
    """
    additional = [
        # Relasi penyebab penyakit
        {"from": "Pyricularia oryzae", "to": "blas", "type": "MENYEBABKAN"},
        {"from": "Xanthomonas oryzae pv. oryzae", "to": "hawar daun bakteri", "type": "MENYEBABKAN"},
        {"from": "wereng hijau", "to": "virus tungro", "type": "MENYEBABKAN"},
        {"from": "wereng hijau", "to": "virus kerdil rumput", "type": "MENYEBABKAN"},
        {"from": "wereng batang coklat", "to": "virus kerdil hampa", "type": "MENYEBABKAN"},
        {"from": "jamur", "to": "busuk pelepah", "type": "MENYEBABKAN"},
        {"from": "jamur", "to": "busuk batang", "type": "MENYEBABKAN"},
        {"from": "jamur", "to": "busuk akar", "type": "MENYEBABKAN"},
        {"from": "jamur", "to": "cercospora", "type": "MENYEBABKAN"},
        {"from": "jamur", "to": "karat daun", "type": "MENYEBABKAN"},
        
        # Relasi nama ilmiah
        {"from": "Nilaparvata lugens", "to": "wereng batang coklat", "type": "NAMA_ILMIAH"},
        {"from": "Cnaphalocrocis medinalis", "to": "hama pelipat daun", "type": "NAMA_ILMIAH"},
        {"from": "Scirpophaga incertulas", "to": "penggerek batang", "type": "NAMA_ILMIAH"},
        {"from": "Nezara viridula", "to": "walang sangit", "type": "NAMA_ILMIAH"},
    ]
    
    relations.extend(additional)
    return relations


def main():
    # Ekstrak relasi dari file
    file_path = "./data/ner_labeling/ner_dataset_500.txt"
    relations = extract_relations_from_ner_dataset(file_path)
    
    # Tambahkan relasi tambahan
    relations = add_additional_relations(relations)
    
    # Hapus duplikasi
    unique_relations = []
    seen = set()
    for rel in relations:
        key = (rel['from'], rel['to'], rel['type'])
        if key not in seen:
            unique_relations.append(rel)
            seen.add(key)
    
    # Simpan ke file JSON
    output_file = "relations_extracted.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_relations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Berhasil mengekstrak {len(unique_relations)} relasi unik")
    print(f"📁 Disimpan ke: {output_file}")
    
    # Statistik
    stats = defaultdict(int)
    for rel in unique_relations:
        stats[rel['type']] += 1
    
    print("\n📊 Statistik Relasi:")
    for rel_type, count in sorted(stats.items()):
        print(f"   {rel_type}: {count}")
    
    # Contoh relasi
    print("\n📝 Contoh 10 relasi pertama:")
    for i, rel in enumerate(unique_relations[:10], 1):
        print(f"   {i}. {rel['from']} --[{rel['type']}]--> {rel['to']}")


if __name__ == "__main__":
    main()