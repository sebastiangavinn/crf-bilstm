DATA = {
  "HAMA": [
    "wereng",
    "penggerek",
    "ulat",
    "wereng batang coklat",
    "wereng hijau",
    "penggerek batang",
    "ulat grayak",
    "ulat penggulung daun",
    "hama pelipat daun",
    "walang sangit",
    "tikus",
    "tikus sawah",
    "keong mas",
    "kutu putih",
    "belalang",
    "Nezara viridula",
    "Nilaparvata lugens",
    "Cnaphalocrocis medinalis",
    "Scirpophaga incertulas"
  ],
  "PENYAKIT": [
    "blas",
    "penyakit blas",
    "Pyricularia oryzae",
    "hawar daun bakteri",
    "Xanthomonas oryzae pv. oryzae",
    "cercospora",
    "busuk pelepah",
    "busuk batang",
    "busuk akar",
    "busuk biji",
    "busuk bulir",
    "virus tungro",
    "tungro",
    "virus kerdil rumput",
    "virus kerdil hampa",
    "karat daun",
    "jamur",
    "bakteri",
    "hawar daun"
  ],
  "GEJALA": [
    "klorosis",
    "menguning",
    "pucat",
    "kering",
    "layu",
    "menggulung",
    "bintik putih",
    "bintik coklat",
    "bercak putih",
    "bercak coklat",
    "bercak air",
    "bercak mata",
    "bercak hitam",
    "bercak ungu",
    "oval",
    "gosong",
    "patah",
    "hampa",
    "kosong",
    "mati anakan",
    "robek",
    "membusuk",
    "bercendawan",
    "keriting",
    "terbakar",
    "kering sebagian",
    "keropos",
    "rapuh",
    "berlendir",
    "kerdil",
    "lemah",
    "busuk",
    "tekuk",
    "lubang",
    "retak",
    "hitam",
    "bergelombang",
    "bengkok",
    "melepuh",
    "melengkung",
    "kusam",
    "memudar",
    "mozaik",
    "luka"
  ],
  "BAGIAN_TANAMAN": [
    "daun",
    "helai daun",
    "tulang daun",
    "pelepah",
    "batang",
    "batang muda",
    "pangkal batang",
    "malai",
    "bulir",
    "akar",
    "anakan"
  ]
}


PESTS = [x.lower() for x in DATA["HAMA"]]
DISEASES = [x.lower() for x in DATA["PENYAKIT"]]
SYMPTOM_KEYWORDS = [x.lower() for x in DATA["GEJALA"]]
PLANT_PARTS = [x.lower() for x in DATA["BAGIAN_TANAMAN"]]

import re
import csv
import json

INPUT_FILE = "./data/ner_labeling/ner_dataset_500.txt"
SYMPTOMS_CSV = "symptoms.csv"
CAUSES_CSV = "causes.csv"
PARTS_CSV = "plant_parts.csv"

rows = []
causes = set()
parts = set()

def match_from_list(text, lst):
    return [w for w in lst if w in text.lower()]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
        if not text:
            continue
        
        # Temukan gejala → mulai dari keseluruhan kalimat
        symptom = text
        
        # Temukan cause dari lookup
        cause_matches = match_from_list(text, PESTS + DISEASES)
        cause = cause_matches[0] if cause_matches else None
        
        # Temukan bagian tanaman
        part_matches = match_from_list(text, PLANT_PARTS)
        part = part_matches[0] if part_matches else None
        
        if cause:
            causes.add(cause)
        if part:
            parts.add(part)

        rows.append({
            "symptom": symptom,
            "cause": cause,
            "part": part,
            "type": (
                "Pest" if cause in PESTS else
                "Disease" if cause in DISEASES else
                None
            )
        })

# Tulis CSV
with open(SYMPTOMS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["symptom","cause","type","part"])
    writer.writeheader()
    for r in rows: writer.writerow(r)

with open(CAUSES_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f); writer.writerow(["name","type"])
    for c in sorted(causes):
        t = "Pest" if c in PESTS else "Disease"
        writer.writerow([c,t])

with open(PARTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f); writer.writerow(["name"])
    for p in sorted(parts): writer.writerow([p])

print("Done! CSV siap import 🚀")
