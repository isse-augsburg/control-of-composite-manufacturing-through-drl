using FileIO
# using ImageIO
# using ImageCore
# using ImageMagick
# using Plots
# include("MeshGenerator.jl")
# include("Mesh2Image.jl")
# MeshGenerator.refresh()
# Mesh2Image.refresh()

files = []
for i in 1:100
    # data, state = MeshGenerator.generate_random_case()
    p_linux = "/cfs/testsets/testset_example/"
    f_linux = p_linux * "mesh" * string(i) * ".jld2"
    # save fvc map
    # img = Mesh2Image.mesh2img_py(data, state, "fvc")
    # p = p * "mesh" * string(i) * "fvc.png"
    # save(p, colorview(Gray, img))
    push!(files, f_linux)
    # save(f, "data", data, "state", state)
end

fp = raw"/Users/leoheber/repos/control-of-composite-manufacturing-through-drl/simulation_storage/testsets/testset_example.jld2"
save(fp, "filelist", files)