import heapq
import networkx as nx
import matplotlib.pyplot as plt


def dijkstra(graph, start):
    """
    graph: lista sąsiedztwa, graph[u] = lista (v, w)
    start: wierzchołek startowy
    """
    n = len(graph)
    dist = {v: float('inf') for v in range(n)}
    prev = {v: None for v in range(n)}
    dist[start] = 0

    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue

        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(prev, start, target):
    path = []
    current = target

    if start == target:
        return [start]

    if prev[current] is None:
        return []

    while current is not None:
        path.append(current)
        if current == start:
            break
        current = prev[current]

    path.reverse()
    if path and path[0] == start:
        return path
    else:
        return []


def draw_graph(graph, path=None, directed=True):
    """
    graph: lista sąsiedztwa (u -> (v, w))
    path: lista wierzchołków na najkrótszej ścieżce (może być None)
    directed: True – graf skierowany, False – nieskierowany
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Dodajemy krawędzie do grafu networkx
    for u, neighbours in enumerate(graph):
        for v, w in neighbours:
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)  # pozycje wierzchołków (ładny układ)

    # Rysowanie wszystkich wierzchołków i krawędzi
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_labels(G, pos)

    # Wszystkie krawędzie (jasne)
    nx.draw_networkx_edges(G, pos, arrows=directed, alpha=0.5)

    # Etykiety wag krawędzi
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Jeśli podano ścieżkę, podświetlamy ją grubszą kreską
    if path is not None and len(path) > 1:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            width=3,  # grubsze
            edge_color="red",
            arrows=directed
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    print("Podaj n (liczba wierzchołków) i m (liczba krawędzi):")
    n, m = map(int, input().split())

    graph = [[] for _ in range(n)]

    print("Podaj krawędzie w formacie: u v w (0-indexed, w – waga):")
    for _ in range(m):
        u, v, w = input().split()
        u = int(u)
        v = int(v)
        w = float(w)
        graph[u].append((v, w))
        # jeśli ma być nieskierowany, odkomentuj:+
        graph[v].append((u, w))

    print("Podaj wierzchołek startowy i docelowy:")
    start, target = map(int, input().split())

    dist, prev = dijkstra(graph, start)
    path = reconstruct_path(prev, start, target)

    if not path:
        print(f"Brak ścieżki z {start} do {target}.")
    else:
        print(f"Najkrótsza odległość z {start} do {target}: {dist[target]}")
        print("Najkrótsza ścieżka:", " -> ".join(map(str, path)))

    # Rysowanie grafu z podświetloną ścieżką
    draw_graph(graph, path=path, directed=True)



main()
