using FileIO
using ImageIO
using ImageCore
using ImageMagick
using Plots
include("MeshGenerator.jl")
include("Mesh2Image.jl")
MeshGenerator.refresh()
Mesh2Image.refresh()

files = []
for i in 1:100
    data, state = MeshGenerator.generate_random_case()
    p = raw"X:\h\e\heberleo\RL4RTM_paper\slight\testset" * "\\"
    p_linux = "/cfs/home/h/e/heberleo/RL4RTM_paper/slight/testset/"
    f_linux = p_linux * "mesh" * string(i) * ".jld2"
    f = p * "mesh" * string(i) * ".jld2"
    # save fvc map
    img = Mesh2Image.mesh2img_py(data, state, "fvc")
    p = p * "mesh" * string(i) * "fvc.png"
    save(p, colorview(Gray, img))
    push!(files, f_linux)
    save(f, "data", data, "state", state)
end

fp = raw"X:\h\e\heberleo\RL4RTM_paper\slight\testset.jld2"
save(fp, "filelist", files)