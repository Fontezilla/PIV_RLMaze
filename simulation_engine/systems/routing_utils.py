"""
routing_utils.py — Wrappers de routing sobre o FactoryGraph.

Funções utilitárias para o MotionAgent calcular próximos hops, distâncias
e caminhos, sem expor detalhes do NetworkX ao resto do sistema.
"""

from typing import List, Optional, Set

from simulation_engine.core.graph import FactoryGraph


def next_hop(graph: FactoryGraph, start: str, goal: str) -> Optional[str]:
    """
    Retorna um próximo hop no caminho mínimo de start a goal.
    Se existirem vários caminhos mínimos com hops diferentes, escolhe
    o primeiro por ordem arbitrária (set iteration).
    Retorna None se start == goal ou não existe caminho.
    """
    hops = graph.get_next_hops(start, goal)
    if not hops:
        return None
    return next(iter(hops))


def all_next_hops(graph: FactoryGraph, start: str, goal: str) -> Set[str]:
    """
    Retorna o conjunto de todos os possíveis próximos hops para goal
    (todos os caminhos mínimos igualmente curtos).
    """
    return graph.get_next_hops(start, goal)


def path(graph: FactoryGraph, start: str, goal: str) -> List[str]:
    """Retorna um caminho mínimo completo (lista de nós) de start a goal."""
    return graph.shortest_path(start, goal)


def route_distance(graph: FactoryGraph, start: str, goal: str) -> float:
    """Retorna a distância acumulada mínima de start a goal."""
    return graph.shortest_path_length(start, goal)


def hop_count(graph: FactoryGraph, start: str, goal: str) -> int:
    """Retorna o número de arestas (hops) no caminho mínimo."""
    paths = graph.get_all_shortest_paths(start, goal)
    if not paths or not paths[0]:
        return 0
    return len(paths[0]) - 1
