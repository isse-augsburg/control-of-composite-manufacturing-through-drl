module Mesh2Image

export mesh2img_py

#using Pkg

#Pkg.add("Conda")
#Pkg.add("PyCall")

using Conda
using PyCall
# Conda.add("numpy")
Conda.add("scipy=1.9.3")

# @pyimport numpy
@pyimport scipy.interpolate as i

include("Config.jl")

function refresh()
    Config.load_config(Config.config)
end

function mesh2img(data, state)
    N = data["N"]
    cellgridid = data["cellgridid"]
    coordinates = data["coordinates"] * 100.
    cellcenter_x = zeros(Float32, N)
    cellcenter_y = zeros(Float32, N)

    point = zeros(Float32, 2)
    for cell in 1:N
        verts = cellgridid[cell, :]
        point .= 0.
        for j in 1:3
            vert = verts[j]
            point .+= coordinates[vert, 1:2]
        end
        cellcenter_x[cell] = point[1] / 3.
        cellcenter_y[cell] = point[2] / 3.
    end


    values = state["gamma"]
    for i in 1:length(values)
        if values[i] < 0.01
            values[i] = 0.
        end
    end
    # mean value per node
    vals = zeros(Float32, size(coordinates)[1])
    count = zeros(Int16, size(coordinates)[1])
    for i in range(1, N)
        for j in range(1, 3)
            node = cellgridid[i, j]
            vals[node] += values[i]
            count[node] += 1
        end
    end
    for i in range(1, size(coordinates)[1])
        vals[i] /= count[i]
    end

    fig, ax, hm = GLMakie.mesh(coordinates[:, 1:2], cellgridid, color=vals)
    GLMakie.display(fig)
    
    spl = Dierckx.Spline2D(cellcenter_x, cellcenter_y, values, s=1., kx=1, ky=1)

    x = collect(LinRange(0, 100, 100))
    y = collect(LinRange(0, 50, 50))
    

    img = Dierckx.evalgrid(spl, x, y)
    
    
    for i in 1:50
        for j in 1:100
            if img[j, i] >= 1.
                img[j, i] = 1.
            end
            if img[j, i] < 0.
                img[j, i] = 0.
            end
        end
    end

    return transpose(img)
end

function mesh2img_py(data, state)
    return mesh2img_py(data, state, "gamma")
end

function mesh2img_py(data, state, key)
    mesh_x = Config.mesh_x
    mesh_y = Config.mesh_y
    x = LinRange(1, mesh_x - 1, mesh_x)
    y = LinRange(1, mesh_y - 1, mesh_y)

    xv = getindex.(Iterators.product(x, y), 1) 
    yv = getindex.(Iterators.product(x, y), 2)

    cellcenters = data["cellcenters"]

    values = nothing

    if key == "gamma"
        global values
        values = state["gamma"]  
    elseif key == "p"
        global values
        values = copy(state["p"])
        for i in 1:size(values)[1]
            values[i] = log10(values[i])
        end
    elseif key == "fvc"
        global values
        values = data["fvc"]
    end

    grid_z = i.griddata(cellcenters, values, (xv, yv), method="linear")
    
    # filter out grey pixel, that seem to be artifacts from interpolation
    # because they confuse the agent & reward function
    if key == "gamma"
        for i in 1:mesh_x
            for j in 1:mesh_y
                if grid_z[i, j] > 0.8
                    grid_z[i, j] = 1.0
                end
                if isnan(grid_z[i, j])
                    grid_z[i, j] = 0.0
                end
            end
        end
    end

    return transpose(grid_z)
end

end