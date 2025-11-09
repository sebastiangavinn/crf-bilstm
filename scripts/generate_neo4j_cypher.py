import json
from pathlib import Path

TOKENS_FILE = Path("data/processed/gazetteer.json")
RELATIONS_FILE = Path("data/processed/relations.json")
OUTPUT_FILE = Path("data/processed/import_graph.cypher")

label_map = {
    "HAMA": "Hama",
    "PENYAKIT": "Penyakit",
    "GEJALA": "Gejala",
    "ORGAN": "OrganTanaman",
    "LOKASI": "Lokasi"
}

with open(TOKENS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

entity_label_map = {}
for category, items in data.items():
    label = label_map.get(category, category.title())
    for name in items:
        entity_label_map[name.lower()] = label 

relations = []
if RELATIONS_FILE.exists():
    with open(RELATIONS_FILE, "r", encoding="utf-8") as f:
        relations = json.load(f)

cypher_lines = [
    "// ================================================",
    "// AUTO-GENERATED KNOWLEDGE GRAPH IMPORT SCRIPT",
    "// ================================================",
    " "
]

cypher_lines.append("// === CREATE NODES ===")
for name, label in entity_label_map.items():
    safe_name = name.replace('"', '\\"')
    cypher_lines.append(f'CREATE (:{label} {{nama: "{safe_name}"}});')

cypher_lines.append(" ")

cypher_lines.append("// === CREATE RELATIONS ===")
for rel in relations:
    f_name = rel["from"].lower()
    t_name = rel["to"].lower()
    rel_type = rel["type"].upper()

    from_label = entity_label_map.get(f_name)
    to_label = entity_label_map.get(t_name)

    if not from_label or not to_label:
        cypher_lines.append(f"// ⚠️ SKIPPED: '{rel['from']}' atau '{rel['to']}' tidak ditemukan di tokens.json")
        continue

    cypher_lines.append(
        f'MATCH (a:{from_label} {{nama: "{f_name}"}}), (b:{to_label} {{nama: "{t_name}"}}) '
        f'CREATE (a)-[:{rel_type}]->(b);'
    )

OUTPUT_FILE.write_text("\n".join(cypher_lines), encoding="utf-8")
print(f"✅ File '{OUTPUT_FILE}' berhasil dibuat!")
