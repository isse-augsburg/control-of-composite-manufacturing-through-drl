include("MeshGenerator.jl")
MeshGenerator.refresh()


meshes = [
    [5, 0, 15, 20],
    [5 , 15, 15, 20],
    [5, 30, 15, 20],
    [15, 0, 15, 20],
    [15 , 15, 15, 20],
    [15, 30, 15, 20],
    [25, 0, 15, 20],
    [25 , 15, 15, 20],
    [25, 30, 15, 20]
]

MeshGenerator.set_from_list(meshes, "test9_final")