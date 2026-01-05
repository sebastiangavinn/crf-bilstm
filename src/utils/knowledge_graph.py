"""
Knowledge Graph handler untuk Neo4j
Menggunakan Neo4j untuk reasoning dan query entitas
"""

import logging
from neo4j import GraphDatabase
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Handler untuk query Neo4j knowledge graph"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j database: {database}")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        logger.info("Neo4j connection closed.")
    
    def query_full_reasoning(self, symptoms: List[str], organs: List[str]) -> List[Dict]:
        """
        Reasoning: cari Hama/Penyakit berdasarkan kecocokan gejala & organ,
        dengan skor berbasis rasio kecocokan (precision-like).
        
        Args:
            symptoms: List of gejala names
            organs: List of bagian tanaman names
            
        Returns:
            List of dictionaries dengan informasi penyakit/hama yang cocok
        """
        query = """
        MATCH (entity)
        WHERE entity:Penyakit OR entity:Hama
        
        OPTIONAL MATCH (entity)-[:MEMILIKI_GEJALA]->(g:Gejala)
        WITH entity, collect(DISTINCT g) AS all_gejala
        
        OPTIONAL MATCH (entity)-[:MENYERANG]->(o:BagianTanaman)
        WITH entity, all_gejala, collect(DISTINCT o) AS all_organ
        
        // Filter gejala & organ yang cocok
        WITH entity,
             [g IN all_gejala WHERE g.nama IN $symptoms] AS gejala_cocok,
             [o IN all_organ WHERE o.nama IN $organs] AS organ_cocok,
             all_gejala, all_organ
        
        WITH entity,
             gejala_cocok, organ_cocok,
             size(gejala_cocok) AS matched_gejala,
             size(organ_cocok) AS matched_organ,
             size(all_gejala) AS total_gejala,
             size(all_organ) AS total_organ
        
        // Hanya ambil yang ada kecocokan
        WHERE matched_gejala > 0 OR matched_organ > 0
        
        WITH entity,
             gejala_cocok,
             organ_cocok,
             CASE 
                WHEN total_gejala = 0 THEN 0.0 
                ELSE 1.0 * matched_gejala / total_gejala 
             END AS score_gejala,
             CASE 
                WHEN total_organ = 0 THEN 0.0
                ELSE 1.0 * matched_organ / total_organ
             END AS score_organ
        
        // hitung skor akhir (70% gejala, 30% organ)
        WITH entity,
             gejala_cocok,
             organ_cocok,
             (0.7 * score_gejala + 0.3 * score_organ) AS skor
        
        OPTIONAL MATCH (penyebab)-[:MENYEBABKAN]->(entity)
        OPTIONAL MATCH (entity)-[:MENYEBABKAN]->(penyakit:Penyakit)
        
        WITH entity,
             gejala_cocok,
             organ_cocok,
             skor,
             collect(DISTINCT penyebab.nama) AS penyebab_list,
             collect(DISTINCT penyakit.nama) AS penyakit_list
        
        RETURN DISTINCT 
            labels(entity)[0] AS tipe,
            entity.nama AS nama,
            [g IN gejala_cocok | g.nama] AS gejala,
            [o IN organ_cocok | o.nama] AS organ,
            penyebab_list AS penyebab,
            penyakit_list AS penyakit_disebabkan,
            skor
        ORDER BY skor DESC, nama ASC
        LIMIT 10
        """
        with self.driver.session(database=self.database) as session:
            results = session.run(query, symptoms=symptoms, organs=organs)
            res = [dict(r) for r in results]
            logger.info(
                "Reasoning query returned %d candidates for symptoms=%s, organs=%s",
                len(res), symptoms, organs
            )
            return res
    
    def query_entity_details(self, name: str) -> Optional[Dict]:
        """
        Query detail entitas berdasarkan nama
        
        Args:
            name: Nama penyakit atau hama
            
        Returns:
            Dictionary dengan detail entitas atau None jika tidak ditemukan
        """
        query = """
        MATCH (e)
        WHERE toLower(e.nama) = toLower($name) AND (e:Penyakit OR e:Hama)
        
        OPTIONAL MATCH (e)-[:MEMILIKI_GEJALA]->(g:Gejala)
        OPTIONAL MATCH (e)-[:MENYERANG]->(o:BagianTanaman)
        OPTIONAL MATCH (penyebab)-[:MENYEBABKAN]->(e)
        OPTIONAL MATCH (e)-[:MENYEBABKAN]->(penyakit:Penyakit)
        OPTIONAL MATCH (e)-[:NAMA_ILMIAH]->(ilmiah)
        
        RETURN labels(e)[0] AS tipe,
               e.nama AS nama,
               collect(DISTINCT g.nama) AS gejala,
               collect(DISTINCT o.nama) AS organ,
               collect(DISTINCT penyebab.nama) AS penyebab,
               collect(DISTINCT penyakit.nama) AS penyakit_disebabkan,
               collect(DISTINCT ilmiah.nama) AS nama_ilmiah
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=name).single()
            if result:
                logger.info("Entity details found for '%s'", name)
            else:
                logger.info("No entity details for '%s'", name)
            return dict(result) if result else None
    
    def query_by_scientific_name(self, name: str) -> Optional[Dict]:
        """
        Query berdasarkan nama ilmiah
        
        Args:
            name: Nama ilmiah
            
        Returns:
            Dictionary dengan detail entitas atau None jika tidak ditemukan
        """
        query = """
        MATCH (ilmiah)-[:NAMA_ILMIAH]->(e)
        WHERE toLower(ilmiah.nama) = toLower($name)
        
        OPTIONAL MATCH (e)-[:MEMILIKI_GEJALA]->(g:Gejala)
        OPTIONAL MATCH (e)-[:MENYERANG]->(o:BagianTanaman)
        OPTIONAL MATCH (penyebab)-[:MENYEBABKAN]->(e)
        OPTIONAL MATCH (e)-[:MENYEBABKAN]->(penyakit:Penyakit)
        
        RETURN labels(e)[0] AS tipe,
               e.nama AS nama,
               ilmiah.nama AS nama_ilmiah,
               collect(DISTINCT g.nama) AS gejala,
               collect(DISTINCT o.nama) AS organ,
               collect(DISTINCT penyebab.nama) AS penyebab,
               collect(DISTINCT penyakit.nama) AS penyakit_disebabkan
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=name).single()
            if result:
                logger.info("Entity found by scientific name '%s'", name)
            else:
                logger.info("No entity by scientific name '%s'", name)
            return dict(result) if result else None

