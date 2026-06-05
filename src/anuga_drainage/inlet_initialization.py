import math
import numpy as np
import pandas as pd

# pyswmm and anuga are imported lazily inside initialize_inlets() so that the
# pure helpers below (read_inp_coordinates, n_sided_inlet) can be imported and
# unit-tested without a full ANUGA/SWMM install.


def read_inp_coordinates(inp_path):
    """Read the [COORDINATES] section of a SWMM .inp file.

    Returns a DataFrame indexed by node id with X_Coord/Y_Coord columns,
    e.g. coords.loc[nodeid].X_Coord. Replaces the former hymo dependency
    (node coordinates are map metadata, not exposed by the pyswmm API).
    """
    names, xs, ys = [], [], []
    in_section = False
    with open(inp_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('['):
                in_section = s.upper().startswith('[COORDINATES')
                continue
            if not in_section or not s or s.startswith(';'):
                continue
            name, x, y = s.split()[:3]
            names.append(name); xs.append(float(x)); ys.append(float(y))
    return pd.DataFrame({'X_Coord': xs, 'Y_Coord': ys}, index=names)

def n_sided_inlet(n_sides, area, inlet_coordinate, rotation):
    # Computes the vertex coordinates and side length of a regular polygon with:
    # Number of sides = n_sides
    # Area = area
    if n_sides < 3:
        raise RuntimeError('A polygon should have at least 3 sides')

    one_segment = math.pi * 2 / n_sides
    side_length = math.sqrt(4.0*area*math.tan(math.pi/n_sides)/n_sides)
    
    radius = side_length/(2.0*math.sin(math.pi/n_sides))

    vertex = [
        (math.sin(one_segment * i + rotation) * radius,
        math.cos(one_segment * i + rotation) * radius)
        for i in range(n_sides)]

    vertex = [[sum(pair) for pair in zip(point, inlet_coordinate)]
            for point in vertex]
        
    return vertex, side_length

def initialize_inlets(domain, sim, coordinates, n_sides = 6, manhole_areas = [1], Q_in_0 = [1], rotation = 0):
    # `coordinates` is a DataFrame of node map coordinates indexed by node id
    # (see read_inp_coordinates), with X_Coord/Y_Coord columns.
    from pyswmm import Nodes
    from anuga import Inlet_operator, Region

    if n_sides < 3:
        raise RuntimeError('A polygon should have at least 3 sides')

    if not(isinstance(manhole_areas, int) or isinstance(manhole_areas,float) or isinstance(manhole_areas,list) or isinstance(manhole_areas,np.ndarray)):
        raise RuntimeError('Invalid ')

    inlet_operators = dict()
    elevation_list  = []
    circumferences  = []
    polygons        = []
    in_nodes        = [node for node in Nodes(sim) if node.is_junction()]

    for inlet_idx, node in enumerate(in_nodes):

        if isinstance(manhole_areas,list) or isinstance(manhole_areas,np.ndarray):
            inlet_area = manhole_areas[inlet_idx] 

        inlet_coordinates = [coordinates.loc[node.nodeid].X_Coord, coordinates.loc[node.nodeid].Y_Coord]
        vertices, side_length = n_sided_inlet(n_sides, inlet_area, inlet_coordinates, rotation)
        
        inlet_operators[node.nodeid] = Inlet_operator(domain, Region(domain,polygon = vertices,expand_polygon = True), Q_in_0[inlet_idx], zero_velocity=True)

        elevation_list.append(inlet_operators[node.nodeid].inlet.get_average_elevation())
        circumferences.append(n_sides*side_length)
        polygons.append(vertices)

    polygons         = np.array(polygons)
    inlet_elevations = np.array(elevation_list)
    circumferences   = np.array(circumferences)

    return inlet_operators,inlet_elevations,circumferences,vertices
