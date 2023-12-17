module MeshGenerator
using ImageCore
using FileIO
using Random

include("MyMesh.jl")
include("RtmSimulation.jl")
include("Config.jl")
include("Mesh2Image.jl")
include("Log.jl")

function refresh()
    Config.load_config(Config.config)
    RtmSimulation.refresh()
    Mesh2Image.refresh()
end

function seed(seed::Integer)
    Random.seed!(seed)
    Log.log("MeshGenerator: Set seed " * string(seed))
end

function generate_random_case()
    # # patch placement
    width = rand() * 2. + 14
    height = rand() * 2. + 14

    # # patch placement
    x = rand() * 16 + 5
    y = rand() * 27 + 5
    # # fvc 0.35 / 0.45
    
    # fvc noise, set fvc values in configfile
    fvc_noise = 0.005

    ##########################################
    # Don't change things below
    base = Config.origin_mesh
    pressure_inlets = Config.pressure_inlets
    fvc_n = Config.fvc_normal
    fvc_p = Config.fvc_patch
    mesh = MyMesh.create_mesh(base, fvc_n, fvc_p, x, y, height, width, pressure_inlets, fvc_noise)
    data = RtmSimulation.prepare(mesh)
    state = RtmSimulation.init(data)
    return data, state
end

function generate_base_case()
    x = 6
    y = 17.5
    width = 15
    height = 13
    width = 0
    height = 0
    fvc_noise = 0.005
    base = Config.origin_mesh
    pressure_inlets = Config.pressure_inlets
    mesh = MyMesh.create_mesh(base, Config.fvc_normal, Config.fvc_patch, x, y, height, width, pressure_inlets, fvc_noise)
    data = RtmSimulation.prepare(mesh)
    state = RtmSimulation.init(data)
    return data, state
end

function set_from_list(configs, name::String, fvc_noise = 0.005)
    # configs shall be of format
    # [
    #   [x1, y1, width1, height1],
    #   ...
    #]
    # other parameters will be taken according to Config.jl
    fp = raw"X:\h\e\heberleo\BA\RtmServer\ressources\filelists\\" * name * ".jld2"
    cluster = "/cfs/home/h/e/heberleo/BA/RtmServer/ressources/sets/" * name * "/"
    win = raw"X:\h\e\heberleo\BA\RtmServer\ressources\sets\\" * name * "\\"
    
    i = 1

    files = []
    for m in configs
        x = m[1]
        y = m[2]
        width = m[3]
        height = m[4]

        base = Config.origin_mesh
        pressure_inlets = Config.pressure_inlets
        mesh = MyMesh.create_mesh(base, Config.fvc_normal, Config.fvc_patch, x, y, height, width, pressure_inlets, fvc_noise)
        data = RtmSimulation.prepare(mesh)
        state = RtmSimulation.init(data)

        cluster_path = cluster * "mesh" * string(i) * ".jld2"
        f = win * "mesh" * string(i) * ".jld2"
        
        # save fvc map
        img = Mesh2Image.mesh2img_py(data, state, "fvc")
        p = win * "mesh" * string(i) * "fvc.png"
        save(p, colorview(Gray, img))

        push!(files, cluster_path)
        save(f, "data", data, "state", state)
        i = i + 1   
    end

    save(fp, "filelist", files)
    
end

end