from math import sqrt

def get_dodecahedron_vertices():
    phi = (1 + sqrt(5)) / 2  # The golden ratio
    # The inverse of the golden ratio
    phi_inv = 1 / phi
    # Each set of coordinates
    points = [
        (1, 1, 1),
        (0, phi_inv, phi),
        (phi_inv, phi, 0),
        (phi, 0, phi_inv)
    ]
    # Generate all permutations of coordinates with positive and negative signs
    vertices = []
    for x, y, z in points:
        for sign_x in [-1, 1]:
            for sign_y in [-1, 1]:
                for sign_z in [-1, 1]:
                    vertices.append((sign_x * x, sign_y * y, sign_z * z))
    return vertices

# Usage
vertices = get_dodecahedron_vertices()
for vertex in vertices:
    print(vertex)
