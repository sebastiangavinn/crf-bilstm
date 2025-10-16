import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime

def scrape_article(url):
    """
    Scrape artikel dari URL dan ekstrak informasi penting
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        print(f"[OK] Status code: {resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Gagal mengambil URL: {e}")
        return None
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Ekstrak judul
    title = soup.find("h1")
    title_text = title.get_text().strip() if title else "No Title"
    
    # Cari konten artikel - coba beberapa selector umum
    content_div = None
    selectors = [
        {"name": "div", "class_": "entry-content"},
        {"name": "div", "class_": "article-content"},
        {"name": "div", "class_": "post-content"},
        {"name": "article"},
        {"name": "main"}
    ]
    
    for selector in selectors:
        content_div = soup.find(selector["name"], class_=selector.get("class_"))
        if content_div:
            print(f"[OK] Konten ditemukan dengan selector: {selector}")
            break
    
    # Ekstrak paragraf
    if content_div:
        paras = content_div.find_all("p")
    else:
        print("[INFO] Menggunakan fallback: semua <p> tag")
        paras = soup.find_all("p")
    
    # Bersihkan dan gabungkan teks
    paragraphs = []
    for p in paras:
        text = p.get_text().strip()
        # Filter paragraf yang terlalu pendek atau kosong
        if text and len(text) > 20:
            paragraphs.append(text)
    
    full_text = "\n\n".join(paragraphs)
    
    return {
        "url": url,
        "title": title_text,
        "content": full_text,
        "paragraphs": paragraphs,
        "scraped_at": datetime.now().isoformat()
    }


def save_for_ner_labeling(data, output_dir="data/ner_labeling"):
    """
    Simpan data dalam format yang siap untuk labeling NER
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simpan sebagai text mentah untuk labeling manual
    txt_path = os.path.join(output_dir, "raw_article.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"JUDUL: {data['title']}\n")
        f.write(f"URL: {data['url']}\n")
        f.write(f"{'-'*80}\n\n")
        f.write(data['content'])
    print(f"[OK] Text disimpan: {txt_path}")
    
    # 2. Simpan sebagai JSON dengan metadata
    json_path = os.path.join(output_dir, "article_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON disimpan: {json_path}")
    
    # 3. Simpan per paragraf untuk labeling granular
    para_path = os.path.join(output_dir, "paragraphs.jsonl")
    with open(para_path, "w", encoding="utf-8") as f:
        for idx, para in enumerate(data['paragraphs']):
            entry = {
                "id": idx,
                "text": para,
                "source": data['url']
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[OK] Paragraphs (JSONL) disimpan: {para_path}")
    
    # 4. Template untuk anotasi NER (format CoNLL-like)
    conll_path = os.path.join(output_dir, "template_ner.txt")
    with open(conll_path, "w", encoding="utf-8") as f:
        f.write("# Format: TOKEN\tLABEL\n")
        f.write("# Label contoh: B-HAMA, I-HAMA, B-PENYAKIT, I-PENYAKIT, B-TANAMAN, O\n\n")
        for para in data['paragraphs'][:3]:  # 3 paragraf pertama sebagai contoh
            words = para.split()
            for word in words:
                f.write(f"{word}\tO\n")
            f.write("\n")
    print(f"[OK] Template NER disimpan: {conll_path}")


def main():
    # URL yang ingin di-scrape
    urls = [
        "https://disperta.mojokertokab.go.id/detail-artikel/hama-dan-penyakit-tanaman-padi-1594789787",
        # Tambahkan URL lain jika perlu
    ]
    
    all_data = []
    
    for url in urls:
        print(f"\n{'='*80}")
        print(f"Scraping: {url}")
        print(f"{'='*80}")
        
        article_data = scrape_article(url)
        
        if article_data:
            all_data.append(article_data)
            save_for_ner_labeling(article_data)
            print(f"\n[SUKSES] Artikel berhasil di-scrape dan disimpan")
            print(f"Jumlah paragraf: {len(article_data['paragraphs'])}")
        else:
            print(f"[GAGAL] Tidak dapat scrape artikel dari {url}")
    
    # Simpan semua data gabungan
    if all_data:
        combined_path = "data/ner_labeling/all_articles.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Semua artikel disimpan: {combined_path}")


if __name__ == "__main__":
    main()