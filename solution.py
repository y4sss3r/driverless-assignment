"""Candidate solution for the assignment "Map, Data Association & Path Planning".

Run it with::

    python3 solution.py

The script executes the full pipeline step by step and stops at the first
step you have not implemented yet, with a reminder of what is missing.
Implement the three functions below (``associate``, ``correct_map``,
``find_path``), keeping their names and signatures unchanged.
"""

import math
from typing import List, Tuple

from map_generation import Map, Point, generate_map, generate_observations
from visualization import (
    visualize_association_n_correction,
    visualize_map_n_observations,
    visualize_path,
)

# ---------------------------------------------------------------------------
# Parameters of the run.
#
# Feel free to experiment with them, but note that your solution will also be
# evaluated with different values (especially with different random seeds).
# ---------------------------------------------------------------------------
NUM_POINTS = 20
X_RANGE = 20.0
Y_RANGE = 20.0
MAP_SEED = 42

NOISE_STD = 0.25
OBSERVATION_RATE = 0.8
OBSERVATIONS_SEED = 7


# ---------------------------------------------------------------------------
# Provided helper
# ---------------------------------------------------------------------------

def path_length(points: List[Point], path: List[int]) -> float:
    """Total Euclidean length of ``path`` over ``points``.

    Useful to measure how short your path is, e.g. when going for Bonus 1.
    The length is the sum of the Euclidean distances between the consecutive
    nodes of the path.
    """
    return sum(math.dist(points[a], points[b]) for a, b in zip(path, path[1:]))


# ---------------------------------------------------------------------------
# Step 3 -- Data association (yours to implement)
# ---------------------------------------------------------------------------

def associate(map_points: Map, observations: List[Point]) -> List[Tuple[int, int]]:
    """Associate every observation with the map point it was generated from.

    Given the map and the observations (noisy, incomplete and shuffled),
    recover the observation-to-map-point correspondence.

    Returns:
        The association couples as a list of ``(map_index, observation_index)``
        tuples, e.g. ``[(0, 12), (3, 5)]`` means that observation 12 was
        generated from map point 0, and observation 5 from map point 3.
    """

    def build_costs_matrix(mappa, osservati):
        # la j-esima colonna corrisponde all' j-esimo index in mappa
        # la i-esima riga corrisponde al i-esimo index in osservati
        # il valore (i, j) corrisponde alla distanza tra il punto osservato alla posizione i, e il punto sulla mappa alla posizione j

        costs = []
        for i, observed_point in enumerate(osservati):
            riga_i = []
            for j, map_point in enumerate(mappa):
                distance = math.dist(observed_point, map_point)
                riga_i.append(distance)
            costs.append(riga_i)
        return costs

    costs=build_costs_matrix(map_points, observations)
    cost_match = [] # "tupla (costo, i, j) ordinata con costo crescente"
    for i in range(len(observations)):
        for j in range(len(map_points)):
            costo = costs[i][j]
            cost_match.append([costo, i, j])

    for i in range(len(cost_match)):
        for j in range(0, len(cost_match) - i - 1):
            if cost_match[j][0] > cost_match[j + 1][0]:
                temp = cost_match[j]
                cost_match[j] = cost_match[j + 1]
                cost_match[j + 1] = temp

    observations_occupate = [False]*len(observations)
    map_points_occupati = [False]*len(map_points)

    associations: List[Tuple[int, int]] = []

    for element in cost_match:
        costo = element[0]
        i = element[1] 
        j = element[2]
        if not observations_occupate[i] and not map_points_occupati[j]:
            associations.append((j, i))
            observations_occupate[i]=True
            map_points_occupati[j]=True

    return associations
    raise NotImplementedError("Step 3 (data association) is yours to implement")


# ---------------------------------------------------------------------------
# Step 4 -- Map correction (yours to implement)
# ---------------------------------------------------------------------------

def correct_map(
    map_points: Map,
    observations: List[Point],
    associations: List[Tuple[int, int]],
) -> Map:
    """Correct the map, given the data association.

    For every association couple ``(i, j)``, the corrected position of map
    point ``i`` is the average (midpoint) of ``map_points[i]`` and
    ``observations[j]``. Map points without an associated observation keep
    their original position.

    Returns:
        The corrected map: a list of ``(x, y)`` tuples with the same length
        and the same order as ``map_points``.
    """
    for i, j in associations:
        map_points[i] = tuple((a+b)/2 for a, b in zip(map_points[i], observations[j]))

    return map_points
    raise NotImplementedError("Step 4 (map correction) is yours to implement")


# ---------------------------------------------------------------------------
# Step 5 -- Path finding (yours to implement)
# ---------------------------------------------------------------------------

def find_path(corrected_map: Map) -> List[int]:
    """Find a path that visits every node of the corrected map exactly once.

    Return the path as a list of node indices in visit order, e.g.
    ``[4, 0, 11, 2, ...]``. A path that includes each node exactly once is
    usually called a *Hamiltonian path*: any permutation of the node indices
    is a formally valid answer, but:

    * **Bonus 1 (+1 point)**: use a heuristic or an algorithm (e.g. nearest
      neighbour, greedy insertion, 2-opt, ...) so that the total length of
      your path is clearly shorter than the worst possible path. You can
      measure it with :func:`path_length`.

    * **Bonus 2 (+1 point)**: no arrow of the path goes over a previous
      arrow (arrows that only touch at a shared node do not count as
      crossings). You can check it visually with ``visualize_path``.
    """
    
    def near_nighbour(corrected_map: Map, init_index: int) -> List[int]: 
        path:List[int]=[init_index]
        for iter in range(1, len(corrected_map)):
            index_node=path[-1]
            node = corrected_map[index_node]
            min_distance=float("inf")
            for index_neighbour in range(0, len(corrected_map)):
                if index_neighbour==index_node or index_neighbour in path:
                    continue
                neighbour=corrected_map[index_neighbour]
                distance=math.dist(node, neighbour)
                if distance<min_distance:
                    min_distance=distance
                    next_index=index_neighbour
            path.append(next_index)
        return path

    def brute_force_first_node(corrected_map: Map) -> List[int]:
        paths_length = []
        for init_index in range(0, len(corrected_map)):
            path_i = near_nighbour(corrected_map, init_index)
            path_i_length = path_length([point for point in corrected_map], path_i)
            paths_length.append(path_i_length)

        path_min = min(paths_length)
        init_node = paths_length.index(path_min)
        return near_nighbour(corrected_map, init_node)

    def two_opt(corrected_map: Map, path:List[int]) -> List[int]:
        def switch_nodes(path:List[int], index_node_C:int, index_node_D:int) -> List[int]:
            # A-B-X-Y-Z-C-D --> A-C-Z-Y-X-B-D  A non init_point; B non end_point
            new_path=path[0:index_node_C]+list(reversed(path[index_node_C:index_node_D+1]))+path[index_node_D+1:]
            return new_path

        optimal_path_found=True
        while optimal_path_found:
            optimal_path_found=False
            opt_path_length=path_length([point for point in corrected_map], path)
            for i in range(1, len(path)-2):
                for j in range(i+1, len(path)-1):
                    new_path=switch_nodes(path, i, j)
                    new_length=path_length([point for point in corrected_map], new_path)
                    if new_length < opt_path_length:
                        optimal_path_found=True
                        path=new_path
                        break
        return path


    return two_opt(corrected_map, brute_force_first_node(corrected_map))
    raise NotImplementedError("Step 5 (path finding) is yours to implement")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1 -- Generate the map (provided) ---------------------------------
    map_points = generate_map(
        num_points=NUM_POINTS,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        seed=MAP_SEED,
    )

    # Step 2 -- Generate the observations (provided) ------------------------
    observations = generate_observations(
        map_points,
        noise_std=NOISE_STD,
        observation_rate=OBSERVATION_RATE,
        seed=OBSERVATIONS_SEED,
    )
    visualize_map_n_observations(map_points, observations, X_RANGE, Y_RANGE)

    # Step 3 -- Data association (yours) -------------------------------------
    try:
        associations = associate(map_points, observations)
    except NotImplementedError:
        print("Step 3 is not implemented yet: fill in associate() in solution.py.")
        return

    # Step 4 -- Map correction (yours) ---------------------------------------
    try:
        corrected_map = correct_map(map_points, observations, associations)
    except NotImplementedError:
        print("Step 4 is not implemented yet: fill in correct_map() in solution.py.")
        return
    visualize_association_n_correction(
        map_points,
        observations,
        corrected_map,
        associations,
        X_RANGE,
        Y_RANGE,
    )

    # Step 5 -- Path finding (yours) -----------------------------------------
    try:
        path = find_path(corrected_map)
    except NotImplementedError:
        print("Step 5 is not implemented yet: fill in find_path() in solution.py.")
        return
    visualize_path(corrected_map, path)

    print(
        f"Pipeline complete: {len(associations)} associations over "
        f"{len(observations)} observations, path over {len(corrected_map)} "
        f"nodes with total length {path_length(corrected_map, path):.2f}."
    )


if __name__ == "__main__":
    main()
