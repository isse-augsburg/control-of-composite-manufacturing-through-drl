###############################################################
# IMPORTS
###############################################################

using HTTP
using Sockets
using FileIO
using JSON3
using StatsBase
using Distributed
include("RtmSimulation.jl")
include("Log.jl")
include("Mesh2Image.jl")
include("Paths.jl")
include("Config.jl")
include("MeshGenerator.jl")
@everywhere include("ParallelSimulation.jl")


###############################################################
# CONSTANTS
###############################################################
const nenvs = nworkers()

Log.log("Listening on " * string(Paths.ip) * "\nServing " * string(nenvs) * " workers.\nIdentifier: " * Paths.addr)

const jobs = RemoteChannel(()->Channel{Any}(nenvs))
const results = RemoteChannel(()->Channel{Any}(nenvs))

const seed = 1337 + parse(Int, Paths.addr) 

mode = "training"
filelist = ["random"]
config_path = ARGS[1]

###############################################################
# SERVER FUNCTIONS
###############################################################
function setup(req::HTTP.Request)
    Log.log("SETUP: LOAD CONFIG")
    # read mode from HTTP request
    mode = String(req.body)
    mode = mode[2:length(mode) - 1]

    Config.load_config(config_path)
    ParallelSimulation.refresh(config_path) # trigger all workers to reload the config
    MeshGenerator.refresh(config_path)
    Mesh2Image.refresh(config_path)
    RtmSimulation.refresh(config_path)

    MeshGenerator.seed(seed)

    if mode == "training"
        msg = "TRAINING MODE\n" *
        "\tpressure: " * string(Config.p_min) * " - " * string(Config.p_max) * "\n" *
        "\tt_step: " * string(Config.t_step) * "s\n" *
        "\tfvc: " * string(Config.fvc_normal) * " / " * string(Config.fvc_patch) * "\n" *
        "\tinlets: " * string(Config.num_inlets) * "\n" *
        "\tdata source path: " * Config.data_source
        Log.log(msg)
        global filelist = ["random"] # pseudo filename "random" will trigger random mesh creation on reset 
    elseif mode == "eval"
        msg = "VALIDATION MODE\n" *
        "\tpressure: " * string(Config.p_min) * " - " * string(Config.p_max) * "\n" *
        "\tt_step: " * string(Config.t_step) * "s\n" *
        "\tinlets: " * string(Config.num_inlets) * "\n" *
        "\tfilelist: " * Config.filelist_path
        Log.log(msg)
        global filelist = load(Config.filelist_path)["filelist"]
        Log.log("Files:")
        Log.log(filelist)
    end

    Log.log("SETUP: DONE")
    response = Dict("id" => Paths.addr)
    HTTP.Response(200, JSON3.write(response))
end

function step(req::HTTP.Request)
    # get actions from HTTP Request
    Log.log("Step: START")
    json = String(req.body)
    json = json[2:length(json) - 1]
    Log.log(json)
    Log.log("Try to read actions.")
    actions = JSON3.read(json, Vector{Vector{Float64}})
    Log.log("Received actions. " * string(size(actions)))

    # prepare info for the workers
    for (id, action) in enumerate(actions)
        ParallelSimulation.put_job(jobs, id, action)
    end
    Log.log("\tStart workers...")
    # let workers execute a step
    ParallelSimulation.step(jobs, results)

    finisheds = Vector{Bool}(undef, nenvs)
    ts = Vector{Float64}(undef, nenvs)

    for i in range(1, nenvs)
        id, finished, t = ParallelSimulation.get_result(results)
        msg = "\tWorker " * string(id) * " is done. Simulation time: " * string(t)
        Log.log(msg)
        finisheds[id] = finished
        ts[id] = t
    end

    Log.log("Step: DONE")    

    # send finished, t via HTTP Response
    response = Dict("finished"=> finisheds, "t"=> ts, "addr" => Paths.addr)
    resp = JSON3.write(response)
    HTTP.Response(200, resp)
end

function reset_selected(req::HTTP.Request)
    Log.log("SelectedReset: START")
    json = String(req.body)
    json = json[2:length(json) - 1]
    selection = JSON3.read(json, Vector{Int})

    for (id, sel) in enumerate(selection)
        if sel == 1 # reset flag
            # get random file from const filelist
            filename = StatsBase.sample(filelist)
            # Reset to random file
            reset_env(id, filename)
        end
    end

    Log.log("SelectedReset: DONE")
    HTTP.Response(200)
end

function reset_all(req::HTTP.Request)
    Log.log("Reset: START")
    # take random samples from const filelist
    if filelist[1] == "random"
        selected_files = StatsBase.sample(filelist, nenvs, replace=true)
        Log.log("Reset all envs to random generated cases.")
    else
        selected_files = filelist[1:nenvs]
        Log.log("Reset all envs. filelist: " * Config.filelist_path)
    end

    for (id, filename) in enumerate(selected_files)
        # Reset to file
        reset_env(id, filename)
    end

    Log.log("Reset: DONE")
    HTTP.Response(200)
end

function file_select(req::HTTP.Request)
    # To use this function, a valid data_source path has to be set in Config.jl
    # Then, filenames found under this path can be selected via a HTTP request.
    # This functionality is meant to be used for evaluation on specific files.

    Log.log("FileSelect: START")
    json = String(req.body)
    json = replace(json, "\\" => "")
    json = json[2:length(json) - 1]
    tuples = JSON3.read(json, Vector{Tuple{Int, String}})
    for t in tuples
        env_id = t[1]
        file = t[2]
        path = Config.data_source * file
        reset_env(env_id, path)
    end
    Log.log("FileSelect: DONE")
    HTTP.Response(200)
end

function get_fvc_map(req::HTTP.Request)
    Log.log("FvcMaps: START")
    for id in 1:nenvs
        # load data
        p = Paths.sim_storage * "server" * Paths.addr * "env" * string(id) * "_data.jld2"
        data = load(p)["data"]
        # load state
        p = Paths.sim_storage * "server" * Paths.addr * "env" * string(id) * "_state.jld2"
        state = load(p)["state"]

        img = Mesh2Image.mesh2img_py(data, state, "fvc")
        p = Paths.shared_fs * "server" * Paths.addr * "env" * string(id) * "fvc.png"
        save(p, colorview(Gray, img))
    end
    Log.log("FvcMaps: DONE")
    HTTP.Response(200)
end

###############################################################
# AUXILIARY FUNCTIONS / IO
###############################################################

function reset_env(id, filename)
    data = 0
    state = 0

    # if the filename equals 'random', a new, random case shall be generated
    if filename != "random"
    # load from prepared file
        Log.log("Reset env " * string(id) * " to: " * filename) 
        d = load(filename)
        data = d["data"]
        state = d["state"]
    else
        Log.log("Reset env " * string(id) * " to random generated mesh. See config for details.")
        data, state = MeshGenerator.generate_random_case()
    end

    # save with naming convention, so that the workers can find the correct data
    p = Paths.sim_storage * "server" * Paths.addr * "env" * string(id) * "_state.jld2"
    save(p, "state", state)
    p = Paths.sim_storage * "server" * Paths.addr * "env" * string(id) * "_data.jld2"
    save(p, "data", data)

    # save fvc map
    img = Mesh2Image.mesh2img_py(data, state, "fvc")
    p = Paths.shared_fs * "server" * Paths.addr * "env" * string(id) * "fvc.png"
    save(p, colorview(Gray, img))
end

###############################################################
# SERVER SETUP
###############################################################
# define endpoints to dispatch to "service" functions
const ROUTER = HTTP.Router()

HTTP.register!(ROUTER, "GET", "/step", step)
HTTP.register!(ROUTER, "GET", "/reset", reset_all)
HTTP.register!(ROUTER, "GET", "/reset_selected", reset_selected)
HTTP.register!(ROUTER, "GET", "/file_select", file_select)
HTTP.register!(ROUTER, "GET", "/setup", setup)
HTTP.register!(ROUTER, "GET", "/fvc_map", get_fvc_map)

server = HTTP.serve(ROUTER, Paths.ip, 8080, verbose=false)
