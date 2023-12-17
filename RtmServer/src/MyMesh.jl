module MyMesh

export load_RtmMesh

using HDF5
using FileIO
using LinearAlgebra


# ASSUMPTIONS
celldirection_val = [1, 0, 0]
cellthickness_val = 0.005
cellalpha_val = 1.0
mu_resin = 0.1
# custom fvc
fvc_normal = 0.3
fvc_patch = 0.4
permeability_normal = 1.00000000e-06 * exp(-1.86478393e+01 * fvc_normal)
permeability_patch = 1.00000000e-06 * exp(-1.86478393e+01 * fvc_patch)
porosity_normal = 1 - fvc_normal
porosity_patch = 1- fvc_patch

# christofs settings
# standard value, patch value
# permeabilitiy_values = [4.0e-9 1.8e-10]
# porosity_values = [0.7320 0.5370]

function load_MarcelMesh(filename::String, normal_fvc, patch_fvc, fvc_noise)
    # Read data from .erfh5 file
    fid = h5open(filename, "r")

    cellgridid = read(fid["mesh"]["elements"])
    N = size(cellgridid)[2]
    cellgridid = permutedims(cellgridid)
    cellgridid .+= 1
    sort!(cellgridid, dims=2) # !important!

    c = read(fid["mesh"]["nodes"])
    c = permutedims(c)
    coordinates = zeros(Float64, (size(c)[1], 3))
    coordinates[:, 1:2] .= c

    patch = read(fid["mesh"]["patch"])
    patch .+= 1 # shift index

    channel = read(fid["mesh"]["channel"])
    channel .+= 1 # shift index

    inlets = read(fid["mesh"]["left_edge"])::Vector
    inlets .+= 1 # shift index
    close(fid)

    # determin correct inlets
    pressure_inlets = []
    yvals = []
    for i in 1:size(inlets)[1]
        cell = inlets[i]
        verts = cellgridid[cell, :]
        x1 = coordinates[verts[1], 1]
        x2 = coordinates[verts[2], 1]
        x3 = coordinates[verts[3], 1]

        y_max = maximum(coordinates[verts, 2])

        if x1 == 0. && x2 == 0. || x1 == 0. && x3 == 0. || x3 == 0. && x2 == 0. 
            push!(pressure_inlets, cell)
            push!(yvals, y_max)
        end
    end

    perm = sortperm(yvals)

    pressure_inlets = pressure_inlets[perm]

    pressure_inlets = [transpose(pressure_inlets[1:5]); transpose(pressure_inlets[6:9]) -1; transpose(pressure_inlets[10:14])]

    mesh = mesh_calculations(N, coordinates, cellgridid)

    # calculate and set fvc & perm & porosity
    fvc = fill(normal_fvc, N) + randn(Float32, N) * fvc_noise
    fvc[patch] .= patch_fvc + randn(Float32) * fvc_noise
    fvc[channel] .= normal_fvc / 2

    cellporosity = ones(N) - fvc
    cellpermeability = 1.00000000e-06 * exp.(-1.86478393e+01 * fvc)
    
    mesh["fvc"] = fvc
    mesh["cellpermeability"] = cellpermeability
    mesh["cellporosity"] = cellporosity
    mesh["pressure_inlets"] = pressure_inlets
    return mesh
end

function load_RtmMesh(filename::String)

    # Read data from .erfh5 file
    fid = h5open(filename, "r")

    # cellgridid: (N, 3) - Matrix
    # contains the indices of the three vertices forming each cell
    cellgridid = read(fid["post"]["constant"]["connectivities"]["SHELL"]["erfblock"]["ic"])
    cellgridid = cellgridid[1:3, :]

    N = size(cellgridid)[2]
    cellgridid = permutedims(cellgridid)
    sort!(cellgridid, dims=2) # !important!

    # coordinates: (#nodes, 3) - Matrix
    # contains the x, y, z of each vertex
    coordinates = read(fid["post"]["constant"]["entityresults"]["NODE"]["COORDINATE"]["ZONE1_set0"]["erfblock"]["res"])
    coordinates = permutedims(coordinates)
    # convert cm to m
    coordinates ./= 100.0

    # fiber fraction
    fiberfraction = read(fid["post"]["constant"]["entityresults"]["SHELL"]["FIBER_FRACTION"]["ZONE1_set1"]["erfblock"]["res"])
    cellpermeability = read(fid["post"]["constant"]["entityresults"]["SHELL"]["PERMEABILITY1"]["ZONE1_set1"]["erfblock"]["res"])
    cellpermeability = Float64.(abs.(cellpermeability[1, :]) / 10000)


    close(fid)
    # Determine perturbations
    patch = Int64[]
    for cell = range(1, N)
        if fiberfraction[cell] > 0.4 # TODO find a better way to do this
            push!(patch, cell)
        end
    end
    # Determine celltype by coordinates            
    # celltype: (N,) - Vector
    # determining the type of a cell based on the coordinates of the vertices forming this cell
    # if two vertices of a cell lie on a border (e.g. have the same max/ min coordinate value in x/ y direction),
    # the cell is considered a wall or pressure boundary cell.
    # celltypes
    # 1 :: normal
    # -1 :: pressure boundary
    # -3 :: wall

    left = 0.0 # x
    right = 1.0 # x
    upper = 0.5 # y
    lower = 0.0 # y

    celltype = ones(Int64, N)

    for i in range(1, N)
        verts = cellgridid[i, :]

        # left border => pressure inlet cell
        if (coordinates[verts[1], 1] == left && coordinates[verts[2], 1] == left) || (coordinates[verts[2], 1] == left && coordinates[verts[3], 1] == left) || (coordinates[verts[3], 1] == left && coordinates[verts[1], 1] == left)
            celltype[i] = -1

            # lower border => wall cell
        elseif (coordinates[verts[1], 2] == lower && coordinates[verts[2], 2] == lower) || (coordinates[verts[2], 2] == lower && coordinates[verts[3], 2] == lower) || (coordinates[verts[3], 2] == lower && coordinates[verts[1], 2] == lower)
            celltype[i] = -3

            # lower border => wall cell
        elseif (coordinates[verts[1], 2] == upper && coordinates[verts[2], 2] == upper) || (coordinates[verts[2], 2] == upper && coordinates[verts[3], 2] == upper) || (coordinates[verts[3], 2] == upper && coordinates[verts[1], 2] == upper)
            celltype[i] = -3

            # right border => wall cell
        elseif (coordinates[verts[1], 1] == right && coordinates[verts[2], 1] == right) || (coordinates[verts[2], 1] == right && coordinates[verts[3], 1] == right) || (coordinates[verts[3], 1] == right && coordinates[verts[1], 1] == right)
            celltype[i] = -3
        end
    end

    # Assumptions
    # CAUTION!
    # hardcode & assumption: cellthickness & celldirection are the same for all cells
    cellthickness = fill(cellthickness_val, N)
    celldirection = celldirection_val
    cellviscosity = fill(mu_resin, N)
    cellalpha = fill(cellalpha_val, N)

    fvc = fill(fvc_normal, N)
    cellpermeability = fill(permeability_normal, N)
    cellporosity = fill(porosity_normal, N)

    fvc[patch] .= fvc_patch
    cellpermeability[patch] .= permeability_patch
    cellporosity[patch] .= porosity_patch

    # indices of inlet cells organized to 7 separate inlets
    # -1 marks invalid index, needs to be tested for later on
    pressure_inlets = [31 33 35; 25 27 29; 19 21 23; 15 17 -1; 9 11 13; 3 5 7; 37 39 1]
    # CAUTION!

    # Calculations
    cellneighbours, num_neighbours, cellneighbours_array = find_cellneighbours(cellgridid, coordinates, N)
    cellvolume, cellcentertocellcenter, cellfacenormal, cellfacearea, Tmats = createCoordinateSystems(N, cellgridid, coordinates, cellneighbours, celldirection, cellthickness)
    #Amats = []
    Amats = zeros(Float64, (6, N))

    for cell = range(1, N)
        for i_neighbour = range(1, num_neighbours[cell])
            # Add (1, 2) lines to the A matrices of the linear system
            Amats[i_neighbour * 2 - 1: i_neighbour * 2, cell] = [cellcentertocellcenter[cell, i_neighbour, 1] cellcentertocellcenter[cell, i_neighbour, 2]]
        end

    end

    # cellcenters for image creation
    cellcenters = zeros(Float64, (N, 2))
    point = zeros(Float64, 2)
    for cell in range(1, N)
        verts = cellgridid[cell, :]
        point .= 0.
        for j in 1:3
            vert = verts[j]
            # multiply by 100 to match pixels
            point .+= coordinates[vert, 1:2] .* 100
        end
        cellcenters[cell, 1] = point[1] / 3.
        cellcenters[cell, 2] = point[2] / 3.
    end

    output_dict = Dict([
        ("N", N),
        ("cellgridid", cellgridid),
        ("coordinates", coordinates),
        ("celltype", celltype),
        ("pressure_inlets", pressure_inlets),
        ("cellthickness", cellthickness),
        ("cellneighbours", cellneighbours),
        ("cellneighboursarray", cellneighbours_array),
        ("num_neighbours", num_neighbours),
        ("cellvolume", cellvolume),
        ("cellviscosity", cellviscosity),
        ("cellpermeability", cellpermeability),
        ("cellporosity", cellporosity),
        ("cellalpha", cellalpha),
        ("Tmats", Tmats),
        ("cellfacenormal", cellfacenormal),
        ("cellfacearea", cellfacearea),
        ("Amats", Amats),
        ("fvc", fvc),
        ("cellcenters", cellcenters)
    ])
    return output_dict
end

function load_base_mesh(filename)
    # loads a basic mesh without fvc info
    # cellpermeability, cellporosity and fvc need to be added

    # Read data from .erfh5 file
    fid = h5open(filename, "r")

    # cellgridid: (N, 3) - Matrix
    # contains the indices of the three vertices forming each cell
    cellgridid = read(fid["post"]["constant"]["connectivities"]["SHELL"]["erfblock"]["ic"])
    cellgridid = cellgridid[1:3, :]

    N = size(cellgridid)[2]
    cellgridid = permutedims(cellgridid)
    sort!(cellgridid, dims=2) # !important!

    # coordinates: (#nodes, 3) - Matrix
    # contains the x, y, z of each vertex
    coordinates = read(fid["post"]["constant"]["entityresults"]["NODE"]["COORDINATE"]["ZONE1_set0"]["erfblock"]["res"])
    coordinates = permutedims(coordinates)
    # convert cm to m
    coordinates ./= 100.0

    close(fid)
    
    mesh = mesh_calculations(N, coordinates, cellgridid)
    return mesh
end

function mesh_calculations(N, coordinates, cellgridid)
    left = 0.0 # x
    right = maximum(coordinates[:, 1])
    upper = 0.5 # y
    lower = 0.0 # y

    celltype = ones(Int64, N)

    for i in range(1, N)
        verts = cellgridid[i, :]

        # left border => pressure inlet cell
        if (coordinates[verts[1], 1] == left && coordinates[verts[2], 1] == left) || (coordinates[verts[2], 1] == left && coordinates[verts[3], 1] == left) || (coordinates[verts[3], 1] == left && coordinates[verts[1], 1] == left)
            celltype[i] = -1

            # lower border => wall cell
        elseif (coordinates[verts[1], 2] == lower && coordinates[verts[2], 2] == lower) || (coordinates[verts[2], 2] == lower && coordinates[verts[3], 2] == lower) || (coordinates[verts[3], 2] == lower && coordinates[verts[1], 2] == lower)
            celltype[i] = -3

            # lower border => wall cell
        elseif (coordinates[verts[1], 2] == upper && coordinates[verts[2], 2] == upper) || (coordinates[verts[2], 2] == upper && coordinates[verts[3], 2] == upper) || (coordinates[verts[3], 2] == upper && coordinates[verts[1], 2] == upper)
            celltype[i] = -3

            # right border => wall cell
        elseif (coordinates[verts[1], 1] == right && coordinates[verts[2], 1] == right) || (coordinates[verts[2], 1] == right && coordinates[verts[3], 1] == right) || (coordinates[verts[3], 1] == right && coordinates[verts[1], 1] == right)
            celltype[i] = -3
        end
    end

    # Assumptions
    # CAUTION!
    # hardcode & assumption: cellthickness & celldirection are the same for all cells
    cellthickness = fill(cellthickness_val, N)
    celldirection = celldirection_val
    cellviscosity = fill(mu_resin, N)
    cellalpha = fill(cellalpha_val, N)

    # indices of inlet cells organized to 7 separate inlets
    # -1 marks invalid index, needs to be tested for later on
    # pressure_inlets = [31 33 35; 25 27 29; 19 21 23; 15 17 -1; 9 11 13; 3 5 7; 37 39 1]
    # CAUTION!

    # Calculations
    cellneighbours, num_neighbours, cellneighbours_array = find_cellneighbours(cellgridid, coordinates, N)
    cellvolume, cellcentertocellcenter, cellfacenormal, cellfacearea, Tmats = createCoordinateSystems(N, cellgridid, coordinates, cellneighbours, celldirection, cellthickness)
    #Amats = []
    Amats = zeros(Float64, (6, N))

    for cell = range(1, N)
        for i_neighbour = range(1, num_neighbours[cell])
            # Add (1, 2) lines to the A matrices of the linear system
            Amats[i_neighbour * 2 - 1: i_neighbour * 2, cell] = [cellcentertocellcenter[cell, i_neighbour, 1] cellcentertocellcenter[cell, i_neighbour, 2]]
        end

    end

    # cellcenters for image creation
    cellcenters = zeros(Float64, (N, 2))
    point = zeros(Float64, 2)
    for cell in range(1, N)
        verts = cellgridid[cell, :]
        point .= 0.
        for j in 1:3
            vert = verts[j]
            # multiply by 100 to match pixels
            point .+= coordinates[vert, 1:2] .* 100
        end
        cellcenters[cell, 1] = point[1] / 3.
        cellcenters[cell, 2] = point[2] / 3.
    end

    output_dict = Dict([
        ("N", N),
        ("cellgridid", cellgridid),
        ("coordinates", coordinates),
        ("celltype", celltype),
        ("cellthickness", cellthickness),
        ("cellneighbours", cellneighbours),
        ("cellneighboursarray", cellneighbours_array),
        ("num_neighbours", num_neighbours),
        ("cellvolume", cellvolume),
        ("cellviscosity", cellviscosity),
        ("cellalpha", cellalpha),
        ("Tmats", Tmats),
        ("cellfacenormal", cellfacenormal),
        ("cellfacearea", cellfacearea),
        ("Amats", Amats),
        ("cellcenters", cellcenters)
    ])
    return output_dict
end

function create_mesh(base, normal_fvc, patch_fvc, x, y, height, width, pressure_inlets, fvc_noise)
    mesh = load_base_mesh(base)
    N = mesh["N"]
    cellcenters = mesh["cellcenters"]

    # identify ids of elements that correspond to patch
    patch = Vector{Int32}(undef, 0)
    for i in 1:N
        (ccx, ccy) = cellcenters[i, :]
        if ccx >= x && ccx <= x + width && ccy >= y && ccy <= y + height
            push!(patch, i)
        end    
    end

    # calculate and set fvc & perm & porosity
    fvc = fill(normal_fvc, N)
    fvc[patch] .= patch_fvc

    fvc = fvc + (2 * randn(Float32, N) .- 1.) * fvc_noise

    cellporosity = ones(N) - fvc
    cellpermeability = 1.00000000e-06 * exp.(-1.86478393e+01 * fvc)


    mesh["fvc"] = fvc
    mesh["cellpermeability"] = cellpermeability
    mesh["cellporosity"] = cellporosity
    mesh["pressure_inlets"] = pressure_inlets
    return mesh
end

# Helper functions
function find_cellneighbours(cellgridid::Matrix, coordinates::Matrix, N::Int64)
    num_vertices = length(coordinates)[1]        

    # cells_by_vertices[vertex] contains the indices of all cells, 
    # that are connect to this vertex
    cells_by_vertices = Vector{Any}(undef, num_vertices)
    for i in range(1, num_vertices)
        cells_by_vertices[i] = Int64[]
    end

    for cell in range(1, N)
        for j in range(1, 3)
            vertex = cellgridid[cell, j]
            push!(cells_by_vertices[vertex], cell)
        end
    end

    for i in range(1, num_vertices)
        unique!(cells_by_vertices[i])
    end

    # identify cell neighbours
    cellneighbours = Vector{Any}(undef, N)
    for i in range(1, N)
        cellneighbours[i] = Int64[]
    end

    for vertex in range(1, num_vertices)
        # for each 'main' vertex
        for i in range(1, length(cells_by_vertices[vertex]))
            # for each cell connected to this vertex
            cell = cells_by_vertices[vertex][i]
            vertices = cellgridid[cell, :]
            for j in range(1, 3)
                # for each vertex_test connected to this cell
                vertex_test = vertices[j]
                if vertex_test != vertex
                    # excluding the 'main' vertex of this iteration
                    possible_neighbour_cells = cells_by_vertices[vertex_test]
                    for k in range(1, length(possible_neighbour_cells))
                        # for each possible neighbour cell to the current cell (e.g. cells sharing a vertex with the current cell)
                        possible_neighbour = possible_neighbour_cells[k]
                        if (possible_neighbour != cell) && (vertex in cellgridid[possible_neighbour, :]) && (vertex in cellgridid[possible_neighbour, :])
                            # if a cell shares two vertices (the test one and the 'main' one), it as an actual neighbour.
                            # Not all neighbours of a cell will be found in the pass for one 'main' vertex,
                            # but as every vertex will be 'main' vertex once, eventually all neighbours will be found (multiple times).
                            push!(cellneighbours[cell], possible_neighbour)
                        end
                    end
                end
            end
        end
    end

    # delete multiple entries
    num_neighbours = zeros(Int64, N)
    cellneighbours_array = zeros(Int64, (N, 3))
    for i in range(1, N)
        unique!(cellneighbours[i])
        sort!(cellneighbours[i])
        num_neighbours[i] = size(cellneighbours[i])[1]
        for j in range(1, num_neighbours[i])
            cellneighbours_array[i, j] = cellneighbours[i][j]
        end 
    end

    return cellneighbours, num_neighbours, cellneighbours_array
end

function createCoordinateSystems(N, cellgridid, coordinates, cellneighbours, celldirection, cellthickness)

    cellcenters = zeros(Float64, (N, 3))
    local_coordinates = Array{Float64}(undef, N, 3, 3) # current_cell, neighbourcell, dimension
    b1 = zeros(N, 3)
    b2 = zeros(N, 3)
    b3 = zeros(N, 3)
    theta = zeros(N)
    cellvolume = zeros(Float64, N)

    T11 = zeros(Float64, N, 3)
    T21 = zeros(Float64, N, 3)
    T12 = zeros(Float64, N, 3)
    T22 = zeros(Float64, N, 3)

    Tmats = zeros(Float64, N, 3, 2, 2)
    cellfacearea = zeros(Float64, N, 3)
    cellcenter_to_cellcenter = zeros(Float64, N, 3, 2)
    cell_face_normal = zeros(Float64, N, 3, 2)


    #loop to define cell center coordinates in global CS
    for ind = 1:N
        i1 = cellgridid[ind, 1]
        i2 = cellgridid[ind, 2]
        i3 = cellgridid[ind, 3]
        # could be written as one line, kept for readability
        cellcenters[ind, :] = (coordinates[i1, :] + coordinates[i2, :] + coordinates[i3, :]) / 3
    end

    for ind in range(1, N)

        i1 = cellgridid[ind, 1]
        i2 = cellgridid[ind, 2]
        i3 = cellgridid[ind, 3]

        b1[ind, :] = coordinates[i2, :] - coordinates[i1, :]
        b1[ind, :] = b1[ind, :] / norm(b1[ind, :])
        a2 = coordinates[i3, :] - coordinates[i1, :]
        a2 = a2 / norm(a2)
        b2[ind, :] = a2 - dot(b1[ind, :], a2) / dot(b1[ind, :], b1[ind, :]) * b1[ind, :]
        b2[ind, :] = b2[ind, :] / norm(b2[ind, :])
        b3[ind, :] = cross(b1[ind, :], b2[ind, :])

        #new cell CS is given by the projection of the primary direction in local CS

        Tmat = [b1[ind, :] b2[ind, :] b3[ind, :]]

        xvec = celldirection # reference_direction, fixed in our case
        r1 = Tmat \ xvec # ref dir in local CS

        #Calculate the angle by which b1 must be rotated about the b3-axis to
        #match r1 via relation rotation matrix Rz[theta]*[1 0 0]'=r1, i.e.
        #cos(theta)=r1[1] & sin(theta)=r1[2]
        theta[ind] = atan(r1[2], r1[1]) # atan with 2 arguments equals atan2 in julia
        #Rotation of theta about nvec=b3 to get c1 & c2
        nvec = b3[ind, :]
        xvec = b1[ind, :]
        c1 = nvec * dot(nvec, xvec) + cos(theta[ind]) * cross(cross(nvec, xvec), nvec) + sin(theta[ind]) * cross(nvec, xvec)
        xvec = b2[ind, :]
        c2 = nvec * dot(nvec, xvec) + cos(theta[ind]) * cross(cross(nvec, xvec), nvec) + sin(theta[ind]) * cross(nvec, xvec)
        xvec = b3[ind, :]
        c3 = nvec * dot(nvec, xvec) + cos(theta[ind]) * cross(cross(nvec, xvec), nvec) + sin(theta[ind]) * cross(nvec, xvec)
        b1[ind, :] = c1
        b2[ind, :] = c2
        b3[ind, :] = c3

        #transformation of vertices into local CS

        # could be written as one line, kept for readability
        local_coordinates[ind, 1, :] = coordinates[i1, :] - cellcenters[ind, :]
        local_coordinates[ind, 2, :] = coordinates[i2, :] - cellcenters[ind, :]
        local_coordinates[ind, 3, :] = coordinates[i3, :] - cellcenters[ind, :]

        Tmat = [b1[ind, :] b2[ind, :] b3[ind, :]]

        for i in range(1, 3)
            xvec = local_coordinates[ind, i, :]
            bvec = Tmat \ xvec
            local_coordinates[ind, i, :] = bvec
        end

        #cell centers of neighbouring cells in local coordinate system()
        #1) projection of cell center P=(0,0) onto straigth line through
        #i1 and i2 to get point Q1 & calculation of length l1 of line()
        #segment PQ1
        #2)projection of neighbouring cell center A onto straight line()
        #through i1 & i2 to get point Q2 in global coordinate system()
        #calculatin of length l2 of segment AQ2 & then
        #transformation of Q2 into local coordinate system & then
        #cellcentertocellcenterx/y[ind,1] is given by vector addition
        #PQ1+Q1Q2+l2/l1*PQ1
    end

    for ind = 1:N

        for i_neighbour = 1:length(cellneighbours[ind])

            neighbour = cellneighbours[ind][i_neighbour]

            #for every neighbour find the two indices belonging to the boundary
            #face in between; face direction is from smaller to larger index
            #x0..local coordinates of smaller index
            #r0..vector from smaller to larger index in LCS
            #ind, cellneighboursarray[ind]

            verts_ind = cellgridid[ind, :]
            verts_neighbour = cellgridid[neighbour, :]

            common_verts = zeros(Int64, 2)
            common_verts_index = zeros(Int64, 2, 2) # index in cellgridid & local_coordinates

            k = 1
            for i in range(1, 3)
                for j in range(1, 3)
                    if verts_ind[i] == verts_neighbour[j]
                        common_verts[k] = verts_ind[i]
                        common_verts_index[k, :] = [i j] # Not sure about the ordering
                        k += 1
                    end
                end
            end

            ia = common_verts_index[1, 1] # ∈ {1, 2, 3} index of common vertex 1 in cellgridid[ind] & local_coordinates[ind]
            ib = common_verts_index[2, 1] # ∈ {1, 2, 3} index of common vertex 2 in cellgridid[ind] & local_coordinates[ind]

            x0 = local_coordinates[ind, ia, :]
            r0 = local_coordinates[ind, ib, :] - local_coordinates[ind, ia, :]

            #define xvec as the vector between cell centers ind &
            #neighbouring cell center [A]
            #(in GCS) & transform xvec in local coordinates bvec, this gives
            #A in LCS
            #find normal distance from A in LCS to the cell boundary
            #with that cell center A in flat geometry & face normal vector
            #can be defined
            #Fill the cell arrays

            x = [0, 0, 0] # column vector, P at origin of local CS
            P = x
            lambda = dot(x - x0, r0) / dot(r0, r0)
            Q1 = x0 + lambda * r0
            l1 = norm(P - Q1)
            nvec = Q1 - P
            nvec = nvec / norm(nvec)
            # replace dict
            cell_face_normal[ind, i_neighbour, 1:2] = nvec[1:2]

            Tmat = [b1[ind, :] b2[ind, :] b3[ind, :]]

            xvec = cellcenters[neighbour, :] - cellcenters[ind, :] # A in global CS
            x = Tmat \ xvec # A in local CS
            A = x
            lambda = dot(x - x0, r0) / dot(r0, r0)
            Q2 = x0 + lambda * r0
            l2 = norm(A - Q2)

            cellcenter_to_cellcenter[ind, i_neighbour, 1:2] = P[1:2] + (Q1[1:2] - P[1:2]) + (Q2[1:2] - Q1[1:2]) + l2 / l1 * (Q1[1:2] - P[1:2])
            cellfacearea[ind, i_neighbour] = 0.5 * (cellthickness[ind] + cellthickness[neighbour]) * norm(local_coordinates[ind, ib, :] - local_coordinates[ind, ia, :])

            #transfromation matrix for (u,v) of neighrouring cells to local
            #coordinate system()

            #construction of the third one in outside normal direction
            #based on the length of the two non-common edges

            local_coordinates_neighbour = zeros(Float64, 3, 3)

            local_coordinates_neighbour[2, 1:2] = local_coordinates[ind, ia, 1:2] # x and y components
            local_coordinates_neighbour[2, 3] = 0 # z component
            local_coordinates_neighbour[3, 1:2] = local_coordinates[ind, ib, 1:2] # x and y components
            local_coordinates_neighbour[3, 3] = 0 # z component

            # find third vertex of neighbouring cell, that isn't shared with cell ind
            ind3 = 0
            for i in range(1, 3)
                if cellgridid[neighbour, i] != common_verts[1] && cellgridid[neighbour, i] != common_verts[2]
                    ind3 = cellgridid[neighbour, i]
                end
            end

            Tmat = [b1[ind, :] b2[ind, :] b3[ind, :]]

            xvec = coordinates[ind3, :] - cellcenters[ind, :] # cell ind3 position relative to cell ind

            x = Tmat \ xvec
            A = x # A in local CS
            lambda = dot(x - x0, r0) / dot(r0, r0)
            Q2 = x0 + lambda * r0
            l2 = norm(A - Q2)
            local_coordinates_neighbour[1, :] = P + (Q1 - P) + (Q2 - Q1) + l2 / l1 * (Q1 - P)
            local_coordinates_neighbour[1, 3] = 0 # z, kept for safety, not sure yet if necessary

            #construction LCS f1;f2;f3 according to procedure from
            #above from points
            #gridxlocal_neighbour[j],gridylocal_neighbour[j]
            #Procedure:
            #create local cell coordinate system: x-axis with unit vector b1 is()
            #from i1 to i2; y-axis is the orthogonal component from i1 to i3; z-axis()
            #is the cross-product

            # local_coordinates_neighbour[1] entspricht ind3, 
            # local_coordinates_neighbour[2] entspricht ind4
            # local_coordinates_neighbour[3] entspricht ind5,
            ind4 = cellgridid[ind, ia]
            ind5 = cellgridid[ind, ib]
            # k3 ∈ {1, 2, 3} soll die Position des groessten davon sein, usw. fuer k2, k1
            # damit in der folgenden Berechnung die Ausrichtung der Koordinatensystem korrekt ist

            k1 = 0
            k2 = 0
            k3 = 0

            if ind3 > ind4
                if ind4 > ind5
                    k3 = 1
                    k2 = 2
                    k1 = 3
                elseif ind3 > ind5
                    k3 = 1
                    k2 = 3
                    k1 = 2
                else
                    k3 = 3
                    k2 = 1
                    k1 = 2
                end
            else # ind4 >ind3
                if ind3 > ind5
                    k3 = 2
                    k2 = 1
                    k1 = 3
                elseif ind4 > ind5
                    k3 = 2
                    k2 = 3
                    k1 = 1
                else
                    k3 = 3
                    k2 = 2
                    k1 = 1
                end
            end


            f1 = local_coordinates_neighbour[k2, :] - local_coordinates_neighbour[k1, :]
            f1 = f1 / norm(f1)
            a2 = local_coordinates_neighbour[k3, :] - local_coordinates_neighbour[k1, :]
            a2 = a2 / norm(a2)
            f2 = a2 - dot(f1, a2) / dot(f1, f1) * f1
            f2 = f2 / norm(f2)
            f3 = cross(f1, f2)

            nvec = f3
            xvec = f1
            c1 = nvec * dot(nvec, xvec) + cos(theta[neighbour]) * cross(cross(nvec, xvec), nvec) + sin(theta[neighbour]) * cross(nvec, xvec)
            xvec = f2
            c2 = nvec * dot(nvec, xvec) + cos(theta[neighbour]) * cross(cross(nvec, xvec), nvec) + sin(theta[neighbour]) * cross(nvec, xvec)
            xvec = f3
            c3 = nvec * dot(nvec, xvec) + cos(theta[neighbour]) * cross(cross(nvec, xvec), nvec) + sin(theta[neighbour]) * cross(nvec, xvec)
            f1[:] = c1
            f2[:] = c2
            f3[:] = c3

            Tmat = [f1'; f2'; f3']'

            if abs(Tmat[1, 1]) < 1e-15
                Tmat[1, 1] = 0.0
            end
            if abs(Tmat[2, 1]) < 1e-15
                Tmat[2, 1] = 0.0
            end
            if abs(Tmat[1, 2]) < 1e-15
                Tmat[1, 2] = 0.0
            end
            if abs(Tmat[2, 2]) < 1e-15
                Tmat[2, 2] = 0.0
            end

            #(u,v)_e=T*(u,v)_f
            T11[ind, i_neighbour] = Tmat[1, 1]
            T12[ind, i_neighbour] = Tmat[1, 2]
            T21[ind, i_neighbour] = Tmat[2, 1]
            T22[ind, i_neighbour] = Tmat[2, 2]

            Tmats[ind, i_neighbour, :, :] = Tmat[1:2, 1:2]
        end

        # cell volume
        # could maybe be moved out of the loop
        cellvolume[ind] = cellthickness[ind] * 0.5 * norm(cross(local_coordinates[ind, 2, :] - local_coordinates[ind, 1, :], local_coordinates[ind, 3, :] - local_coordinates[ind, 1, :]))
    end
    return cellvolume, cellcenter_to_cellcenter, cell_face_normal, cellfacearea, Tmats
end
end

