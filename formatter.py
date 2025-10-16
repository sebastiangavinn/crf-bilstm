import re
import json
from collections import defaultdict

class NERDatasetConverter:
    def __init__(self):
        self.entity_map = {
            'hama': 'PEST',
            'penyakit': 'DISEASE',
            'gejala': 'SYMPTOM',
            'organ': 'ORGAN',
            'lokasi': 'LOCATION',
            'kondisi lingkungan': 'CONDITION',
            'hama dewasa': 'PEST',
            'penyakit sekunder': 'DISEASE'
        }
    
    def parse_line(self, line):
        """Parse satu baris teks dengan anotasi dalam kurung"""
        tokens = []
        entities = []
        
        # Hapus semua label dalam kurung terlebih dahulu, tapi simpan informasinya
        # Pattern: kata diikuti (label)
        word_label_pairs = []
        
        # Split berdasarkan spasi
        parts = line.split()
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Cek apakah ada label dalam kurung
            if '(' in part:
                # Cek apakah kurung tutup ada di part yang sama
                if ')' in part:
                    # Format: kata(label) atau kata (label)
                    match = re.match(r'^(.+?)\s*\(([^)]+)\)(.*)$', part)
                    if match:
                        word = match.group(1)
                        label = match.group(2)
                        suffix = match.group(3)  # tanda baca setelah kurung
                        
                        if word:
                            word_label_pairs.append((word, label))
                        if suffix:
                            word_label_pairs.append((suffix, None))
                    else:
                        word_label_pairs.append((part, None))
                else:
                    # Kurung buka tapi tutup di part berikutnya
                    # Ambil kata sebelum kurung
                    word_before = part.split('(')[0]
                    if word_before:
                        # Cari kurung tutup
                        label_parts = [part.split('(')[1]]
                        i += 1
                        while i < len(parts) and ')' not in parts[i]:
                            label_parts.append(parts[i])
                            i += 1
                        if i < len(parts):
                            last_part = parts[i]
                            if ')' in last_part:
                                label_parts.append(last_part.split(')')[0])
                                label = ' '.join(label_parts)
                                suffix = last_part.split(')')[1] if ')' in last_part else ''
                                word_label_pairs.append((word_before, label))
                                if suffix:
                                    word_label_pairs.append((suffix, None))
                    else:
                        word_label_pairs.append((part, None))
            else:
                word_label_pairs.append((part, None))
            
            i += 1
        
        # Proses word_label_pairs menjadi tokens dan entities
        for word, label in word_label_pairs:
            if not word:
                continue
                
            # Pisahkan tanda baca
            # Cek tanda baca di awal
            if word[0] in '.,;:!?':
                tokens.append(word[0])
                entities.append('O')
                word = word[1:]
            
            # Cek tanda baca di akhir
            if word and word[-1] in '.,;:!?':
                main_word = word[:-1]
                punct = word[-1]
                if main_word:
                    tokens.append(main_word)
                    if label:
                        entity_type = self.entity_map.get(label.lower(), label.upper())
                        entities.append(entity_type)
                    else:
                        entities.append('O')
                tokens.append(punct)
                entities.append('O')
            else:
                if word:
                    tokens.append(word)
                    if label:
                        entity_type = self.entity_map.get(label.lower(), label.upper())
                        entities.append(entity_type)
                    else:
                        entities.append('O')
        
        return tokens, entities
    
    def read_file(self, filepath):
        """Baca file input"""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]
    
    def to_conll(self, lines, output_file='output_conll.txt'):
        """Konversi ke format CoNLL"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                tokens, entities = self.parse_line(line)
                for token, entity in zip(tokens, entities):
                    f.write(f"{token}\t{entity}\n")
                f.write("\n")  # Baris kosong sebagai pemisah kalimat
        print(f"✓ File CoNLL tersimpan: {output_file}")
        return output_file
    
    def to_bio(self, lines, output_file='output_bio.txt'):
        """Konversi ke format BIO tagging"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                tokens, entities = self.parse_line(line)
                prev_entity = 'O'
                
                for token, entity in zip(tokens, entities):
                    if entity != 'O':
                        # Tentukan prefix B- atau I-
                        if entity == prev_entity:
                            bio_tag = f"I-{entity}"
                        else:
                            bio_tag = f"B-{entity}"
                    else:
                        bio_tag = 'O'
                    
                    f.write(f"{token}\t{bio_tag}\n")
                    prev_entity = entity
                
                f.write("\n")
        print(f"✓ File BIO tersimpan: {output_file}")
        return output_file
    
    def to_json(self, lines, output_file='output_json.json'):
        """Konversi ke format JSON"""
        dataset = []
        
        for idx, line in enumerate(lines):
            tokens, entities = self.parse_line(line)
            text = ' '.join(tokens)
            
            # Ekstrak entitas dengan posisi karakter
            entity_list = []
            char_pos = 0
            current_entity = None
            
            for token, entity in zip(tokens, entities):
                if entity != 'O':
                    if current_entity and current_entity['label'] == entity:
                        # Lanjutkan entitas yang sama
                        current_entity['end'] = char_pos + len(token)
                        current_entity['text'] += ' ' + token
                    else:
                        # Simpan entitas sebelumnya dan mulai yang baru
                        if current_entity:
                            entity_list.append(current_entity)
                        current_entity = {
                            'start': char_pos,
                            'end': char_pos + len(token),
                            'label': entity,
                            'text': token
                        }
                else:
                    if current_entity:
                        entity_list.append(current_entity)
                        current_entity = None
                
                char_pos += len(token) + 1
            
            if current_entity:
                entity_list.append(current_entity)
            
            dataset.append({
                'id': idx,
                'text': text,
                'entities': entity_list
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"✓ File JSON tersimpan: {output_file}")
        return output_file
    
    def to_spacy(self, lines, output_file='output_spacy.json'):
        """Konversi ke format spaCy"""
        training_data = []
        
        for line in lines:
            tokens, entities = self.parse_line(line)
            text = ' '.join(tokens)
            
            # Ekstrak entitas dengan posisi karakter
            entity_list = []
            char_pos = 0
            
            for token, entity in zip(tokens, entities):
                if entity != 'O':
                    entity_list.append((char_pos, char_pos + len(token), entity))
                char_pos += len(token) + 1
            
            training_data.append((text, {'entities': entity_list}))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"✓ File spaCy tersimpan: {output_file}")
        return output_file
    
    def get_statistics(self, lines):
        """Hitung statistik dataset"""
        total_tokens = 0
        entity_counts = defaultdict(int)
        
        for line in lines:
            tokens, entities = self.parse_line(line)
            total_tokens += len(tokens)
            for entity in entities:
                if entity != 'O':
                    entity_counts[entity] += 1
        
        stats = {
            'total_sentences': len(lines),
            'total_tokens': total_tokens,
            'total_entities': sum(entity_counts.values()),
            'entity_distribution': dict(entity_counts)
        }
        
        return stats
    
    def print_statistics(self, stats):
        """Cetak statistik dataset"""
        print("\n" + "="*50)
        print("STATISTIK DATASET")
        print("="*50)
        print(f"Total Kalimat    : {stats['total_sentences']}")
        print(f"Total Token      : {stats['total_tokens']}")
        print(f"Total Entitas    : {stats['total_entities']}")
        print("\nDistribusi Entitas:")
        for entity, count in sorted(stats['entity_distribution'].items()):
            print(f"  {entity:<15} : {count:>5}")
        print("="*50 + "\n")


# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi converter
    converter = NERDatasetConverter()
    
    # Baca file input
    input_file = 'dataset.txt'  # Ganti dengan nama file Anda
    print(f"Membaca file: {input_file}")
    lines = converter.read_file(input_file)
    
    # Hitung dan tampilkan statistik
    stats = converter.get_statistics(lines)
    converter.print_statistics(stats)
    
    # Konversi ke berbagai format
    print("Mengkonversi dataset...")
    converter.to_conll(lines, 'dataset_conll.txt')
    converter.to_bio(lines, 'dataset_bio.txt')
    converter.to_json(lines, 'dataset_json.json')
    converter.to_spacy(lines, 'dataset_spacy.json')
    
    print("\n✓ Semua konversi selesai!")
    print("\nFile output yang dihasilkan:")
    print("  - dataset_conll.txt   : Format CoNLL")
    print("  - dataset_bio.txt     : Format BIO tagging")
    print("  - dataset_json.json   : Format JSON")
    print("  - dataset_spacy.json  : Format spaCy")