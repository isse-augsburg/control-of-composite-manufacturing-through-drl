using FileIO
using ImageIO
using ImageCore
using ImageMagick

include("RtmSimulation.jl")
include("MyMesh.jl")
include("Mesh2Image.jl")

#const save_path = "X:\\h\\e\\heberleo\\BA\\RtmServer\\ressources\\sim"
save_path = "/cfs/home/h/e/heberleo/BA/RtmServer/ressources/sim"

function prepare_erfh5(count::Int, filename::String, newname)
    mesh = MyMesh.load_RtmMesh(filename)
    data = RtmSimulation.prepare(mesh)
    state = RtmSimulation.init(data)

    fvc_img = Mesh2Image.mesh2img_py(data, state, "fvc")
    fvc_img = map(clamp01nan, colorview(Gray, fvc_img))

    save(save_path * "/" * newname * "_fvc.png", fvc_img)

    file = save_path * "/" * newname * ".jld2"
    save(file, "data", data, "state", state)

    return file::String
end



##################################################################
# DATA PREPROCESSING SCRIPT
##################################################################
# sourcepath = "Y:\\data\\RTM\\LinearInjection\\sim_out\\output\\with_shapes\\2021-12-22_10-44-17_100p"
sourcepath = "/cfs/share/data/RTM/LinearInjection/sim_out/output/with_shapes/2022-06-03_15-38-06_1000p"

subfolders = readdir(sourcepath)
filelist = Vector{String}(undef, 0)
ids = Vector{String}(undef, 0)
for f in subfolders
    println("Searching " * f)
    folder = sourcepath * "/" * f
    if isdir(folder)
        files = readdir(folder)
        for file in files
            if occursin("_RESULT.erfh5", file)
                println("Hit! " * file)
                push!(filelist, folder * "/" * file)
                push!(ids, "2022-06-03_15-38-06_" * string(f))
            end
        end
    end
end
println("Done searching.")

prepared = Vector{String}(undef, 0)
for (count, filename) in enumerate(filelist)
    meshfile = prepare_erfh5(count, filename, ids[count])
    println("Finished " * string(count) * ": " * ids[count])
    push!(prepared, meshfile)
end

p = save_path * "/prepared_2022-06-03_15-38-06_1000p.jld2"
save(p, "filelist", prepared)