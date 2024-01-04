module ParallelSimulation
using Distributed
@everywhere using FileIO
@everywhere using ImageCore
@everywhere using ImageIO
@everywhere using Sockets
@everywhere include("RtmSimulation.jl")
@everywhere include("Mesh2Image.jl")
@everywhere include("Paths.jl")
@everywhere include("Log.jl")

export step, put_job, get_result

###########################################################
# WORKER FUNCTIONS & ATTRIBUTES
###########################################################

# main worker function
@everywhere function do_step(jobs, results)
    job = take!(jobs)
    id = job[1]
    action = job[2]

    # load prepare_erfh5 data and state of the last step
    data, state = load_ds(id)

    msg = "Job " * string(id) * ", old iter: " * string(state["iter"])
    Log.log(msg)

    # execute solver step
    finished, state = RtmSimulation.step(action, data, state)

    msg = "Job " * string(id) * ", new iter: " * string(state["iter"])
    Log.log(msg)

    # save state for next step
    save_s(id, state)

    msg = "Saved state."
    Log.log(msg)

    # observe (create img, read t)
    
    t = state["t"]

    # create and save imgs
    save_i(id, data, state)
    msg = "Saved images."
    Log.log(msg)

    # send results (except the image) to the master process
    put!(results, (id, finished, t))
end

# worker function to load simulation data & state
@everywhere function load_ds(id)
    p = Paths.sim_storage * "server" * Paths.addr * "env" * string(id) * "_state.jld2"
    d = load(p)
    state = d["state"]

    p = Paths.sim_storage * "server" * Paths.addr * "env" * string(id) * "_data.jld2"
    d = load(p)
    data = d["data"]

    return data, state
end

# worker function to reload the config
@everywhere function reload_config(config_path::String)
    RtmSimulation.refresh(config_path)
    Mesh2Image.refresh(config_path)
    msg = "WORKER LOAD CONFIG\n"
    Log.log(msg)
end

# worker function to save updated state
@everywhere function save_s(id, state)
    p = Paths.sim_storage * "server" * Paths.addr * "env" * string(id) * "_state.jld2"
    save(p, "state", state)
end

# worker function to save the image
@everywhere function save_i(id, data, state)
    # save flowfront imgage
    img = Mesh2Image.mesh2img_py(data, state)
    p = Paths.shared_fs * "server" * Paths.addr * "env" * string(id) * "image.png"
    save(p, colorview(Gray, img))

    # save pressure image
    img = Mesh2Image.mesh2img_py(data, state, "p")
    p = Paths.shared_fs * "server" * Paths.addr * "env" * string(id) * "pressure.png"
    img = img / maximum(img)
    save(p, colorview(Gray, img))
end

###########################################################
# MASTER FUNCTIONS
###########################################################

function step(jobs, results)   
    for p in workers()
        remote_do(Main.do_step, p, jobs, results)
    end
end 

function put_job(jobs, id, action)
    put!(jobs, (id, action))
end

function get_result(results)
    r = take!(results)
    id = r[1]
    finished = r[2]
    t = r[3]
    
    return id, finished, t
end

function refresh(config_path::String)

    function f() 
        Main.reload_config(config_path)
    end

    for p in workers()
        remotecall_wait(f, p)
    end
end

end

