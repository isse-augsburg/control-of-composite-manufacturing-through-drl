using FileIO
using ImageIO
using ImageCore
using ImageMagick
using Plots
using Makie
using CairoMakie
using ColorSchemes

include("Config.jl")
include("RtmSimulation.jl")
include("X:\\h\\e\\heberleo\\BA\\RtmServer\\src\\Mesh2Image.jl")
Config.load_config(Config.config)

function plot_border(data)
    verts = data["coordinates"]
    inds = data["cellgridid"]
    N = data["N"]

    i1 = []
    i2 = []
    for i in 1:N
        v1 = inds[i, 1]
        v2 = inds[i, 2]
        v3 = inds[i, 3]

        x1 = verts[v1, 1]
        x2 = verts[v2, 1]
        x3 = verts[v3, 1]

        if x1 == 0. && x2 == 0. || x1 == 0. && x3 == 0. || x2 == 0. && x3 == 0.
            push!(i2, i)
        else 
            push!(i1, i)
        end
    end
    unique!(i1)
    unique!(i2)
    println("Border cells:")
    println(i2)

    f = Figure()
    ax = Axis(f[1, 1])
    Makie.poly!(ax, verts, inds[i1, :], strokecolor = :black, strokewidth = 1, color = RGB{Float32}(0, 0, 255))
    Makie.poly!(ax, verts, inds[i2, :], strokecolor = :black, strokewidth = 1, color = RGB{Float32}(255, 0, 0))
    
    tightlimits!(fig.axis)
    fig.axis.scene
end

function plot_inlets(data, inlets)
    verts = data["coordinates"]
    inds = data["cellgridid"]
    N = data["N"]

    n = length(inlets[:, 1])
    f = Figure()
    ax = Axis(
        f[1, 1], 
        aspect=DataAspect(), 
        bottomspinevisible=false,
        leftspinevisible=false,
        rightspinevisible=false,
        topspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
        xlabelvisible=false,
        ylabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        xautolimitmargin = (0.0,0.0),
        yautolimitmargin = (0.0,0.0),
        backgroundcolor = :transparent
    )

    Makie.poly!(ax, verts, inds, strokecolor = :black, strokewidth = 1, color = RGB{Float32}(1, 1, 1))

    set = inlets[1, :]
    deleteat!(set, findall(x -> (x == -1), set))
    Makie.poly!(ax, verts, inds[set, :], strokecolor = :black, strokewidth = 1, color = RGB{Float32}(1, 0, 0))

    set = inlets[2, :]
    deleteat!(set, findall(x -> (x == -1), set))
    Makie.poly!(ax, verts, inds[set, :], strokecolor = :black, strokewidth = 1, color = RGB{Float32}(0, 1, 0))

    set = inlets[3, :]
    deleteat!(set, findall(x -> (x == -1), set))
    Makie.poly!(ax, verts, inds[set, :], strokecolor = :black, strokewidth = 1, color = RGB{Float32}(0, 0, 1))

    tightlimits!(ax)
    ax.scene
end

function plot_fvc(data, state)
    verts = data["coordinates"]
    inds = data["cellgridid"]
    N = data["N"]
    fvc = data["fvc"]
    fvc = state["gamma"]
    diff = maximum(fvc) - minimum(fvc)
    mini = minimum(fvc)

    f = Figure()
    ax = Axis(
        f[1, 1], 
        aspect=DataAspect(), 
        bottomspinevisible=false,
        leftspinevisible=false,
        rightspinevisible=false,
        topspinevisible=false,
        xgridvisible=false,
        ygridvisible=false,
        xlabelvisible=false,
        ylabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        xautolimitmargin = (0.0,0.0),
        yautolimitmargin = (0.0,0.0),
        backgroundcolor = :transparent
    )

    cscheme = ColorSchemes.gray1
    for i in range(1, N)
        cell = inds[i, :]
        v = transpose(verts[cell, 1:2])
        c = (fvc[i] - mini) / diff
        # c = c * 0.6 + 0.2
        # Makie.poly!(ax, v, strokecolor = :black, strokewidth = 1, color = RGB{Float32}(c, c, c))
        Makie.poly!(ax, v, strokecolor = :black, strokewidth = 1, color = get(cscheme, c))

    end

    tightlimits!(ax)
    ax.scene 
end

include("MeshGenerator.jl")
MeshGenerator.refresh()
RtmSimulation.refresh()
Mesh2Image.refresh()

data, state = MeshGenerator.generate_base_case()
img = Mesh2Image.mesh2img_py(data, state, "fvc")
# display(Plots.heatmap(img))#, clim=(0., 1.)))
p = "X:\\h\\e\\heberleo\\Presi_simon\\gifs\\undisturbed\\fvc.png"
img = reverse(img, dims = 1)
save(p, colorview(Gray, img))

img = Mesh2Image.mesh2img_py(data, state, "gamma")
img = clamp.(img, 0., 1.)
# display(Plots.heatmap(img))#, clim=(0., 1.)))
p = "X:\\h\\e\\heberleo\\Presi_simon\\gifs\\undisturbed\\img0.png"
img = reverse(img, dims = 1)
save(p, colorview(Gray, img))

for i in 1:1
    action = [1., 1., 1.]
    for i in 1:30
        f, s = RtmSimulation.step(action, data, state)
        global state = s
        img = Mesh2Image.mesh2img_py(data, state, "gamma")
        # display(Plots.heatmap(img))#, clim=(0., 1.)))
        p = "X:\\h\\e\\heberleo\\Presi_simon\\gifs\\undisturbed\\img" * string(i) * ".png"
        img = reverse(img, dims = 1)
        save(p, colorview(Gray, img))
    end

end

"""f = plot_fvc(data, state)
Makie.save("X:\\h\\e\\heberleo\\Presi_simon\\filling_mesh.png", f, px_per_unit = 20)

img = Mesh2Image.mesh2img_py(data, state, "gamma")
p = "X:\\h\\e\\heberleo\\Presi_simon\\filling_interp.png"

img = reverse(img, dims = 1)
save(p, colorview(Gray, img))

f = plot_inlets(data, Config.pressure_inlets)
Makie.save("X:\\h\\e\\heberleo\\Presi_simon\\mesh.png", f, px_per_unit = 20)"""




