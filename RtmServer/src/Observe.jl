


using Plots
using HDF5
using FileIO
gr(legend=false,markerstrokewidth=0,markersize=3)


function plot_cells(cellgridid, coordinates, N, color)

    shapes = []
    for cell in range(1, N)
        verts = cellgridid[cell, :]
        poly = Any[]
        for i in range(1, 3)
            vert = verts[i]
            x = coordinates[vert, 1]
            y = coordinates[vert, 2]
            push!(poly, (x, y))
        end
        push!(shapes, poly)
            
    end
    display(Plots.plot!([Shape(shapes[i]) for i in range(1, N)], fill_z=color, show=true, legend=false))
end



f = "X:\\h\\e\\heberleo\\BA\\Tests\\mesh_1.hdf5"

fid = h5open(f, "r")

cellgridid = read(fid["mesh"]["elements"])
N = size(cellgridid)[2]
cellgridid = permutedims(cellgridid)
cellgridid .+= 1
sort!(cellgridid, dims=2) # !important!

coordinates = read(fid["mesh"]["nodes"])
coordinates = permutedims(coordinates)


patch = read(fid["mesh"]["dryspots"])
patch .+= 1

inlets = read(fid["mesh"]["left_edge"])
inlets .+= 1
print(inlets)

color = ones(N)
max_y = zeros(length(inlets))

# get the biggest y value of each inlet cell
for i in range(1, length(inlets))
    cell = inlets[i]
    y_val = 0.
    for j in range(1, 3)
        node = cellgridid[cell, j]
        if coordinates[node, 2] > y_val
            y_val = coordinates[node, 2]
        end
    end
    max_y[i] = y_val
end

perm = sortperm(max_y)

inlet1 = inlets[perm[1:6]]
inlet2 = inlets[perm[7:12]]
inlet3 = inlets[perm[13:18]]
inlet4 = inlets[perm[19:24]]
inlet5 = inlets[perm[25:30]]
inlet6 = inlets[perm[31:36]]
inlet7 = inlets[perm[37:42]]

inlets = [inlet1, inlet2, inlet3, inlet4, inlet5, inlet6, inlet7]

println(inlets)

for i in range(1, 7)
    color[inlets[i]] .= i + 1
end

plot_cells(cellgridid, coordinates, N, color)

