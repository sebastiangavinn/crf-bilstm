from py2neo import Graph, Node, Relationship

def build_graph():
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    print("[Connected] ke Neo4j database")

    padi = Node("Plant", name="Padi")
    wereng = Node("Pest", name="Wereng Coklat")
    rel = Relationship(padi, "DISERANG_OLEH", wereng)

    graph.create(padi)
    graph.create(wereng)
    graph.create(rel)
    print("[OK] Graph node & relasi ditambahkan!")

if __name__ == "__main__":
    build_graph()
