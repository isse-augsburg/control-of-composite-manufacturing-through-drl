module RtmSimulation

export prepare, init, step

include("MySolver_experimental.jl")
include("Config.jl")
include("Log.jl")

function refresh()
    Config.load_config(Config.config)
end

function prepare(data::Dict)
    # mesh properties
    N = data["N"]
    celltype = data["celltype"]
    num_neighbours = data["num_neighbours"]
    cellneighbours = data["cellneighbours"]
    cellneighboursarray = data["cellneighboursarray"]
    Amats = data["Amats"]
    Tmats = data["Tmats"]
    cellfacearea = data["cellfacearea"]
    cellfacenormal = data["cellfacenormal"]
    cellviscosity = data["cellviscosity"]
    cellpermeability = data["cellpermeability"]
    cellporosity = data["cellporosity"]
    cellalpha = data["cellalpha"]
    cellvolume = data["cellvolume"]
    cellthickness = data["cellthickness"]
    pressure_inlets = data["pressure_inlets"]
    cellgridid = data["cellgridid"]
    coordinates = data["coordinates"]
    fvc = data["fvc"]
    cellcenters = data["cellcenters"]

    # Calculations
    # helper values only used for calculations in constructor
    p_a_val = 1.5e5
    p_init_val = 1e5
    p_ref = 1.0e5
    rho_ref = 1.205
    p_eps = 0.001e5

    # time step calculation and simulation time
    deltatmax = 0.2
    # TODO
    area = minimum(cellvolume ./ cellthickness)

    #area = min(cellvolume ./ cellthickness)
    maxspeed = max(maximum(cellpermeability ./ cellviscosity), maximum(cellalpha .* cellpermeability ./ cellviscosity)) * (p_a_val - p_init_val) / area
    #maxspeed = max(max(cellpermeability ./ cellviscosity), max(cellalpha .* cellpermeability ./ cellviscosity)) * (p_a_val - p_init_val) / min(cellvolume ./ cellthickness) #sqrt(area);
    betat1 = 1
    deltat = betat1 * sqrt(area) / maxspeed
    deltat = min(deltat, deltatmax)
    neutral_inds = Int64[]
    for i = range(1, N)
        if celltype[i] != -1
            push!(neutral_inds, i)
        end
    end

    # STEP TIME
    t_min = 4 * deltat
    t_max = 400.0
    t_max = max(t_min, t_max)

    # constant parameters
    gamma_val = 1.4
    kappa = p_ref / (rho_ref^gamma_val)

    # boundary conditions
    p_a = p_a_val - p_init_val + p_eps # Normalization
    u_a = 0
    v_a = 0
    gamma_a = 1.0
    rho_a = (p_a / kappa)^(1 / gamma_val)

    # polynom parameters for gamma calculation
    p_int1 = 0.0e5
    rho_int1 = (p_int1 / kappa)^(1 / gamma_val)
    p_int2 = 0.1e5
    rho_int2 = (p_int2 / kappa)^(1 / gamma_val)
    p_int3 = 0.5e5
    rho_int3 = (p_int3 / kappa)^(1 / gamma_val)
    p_int4 = 1.0e5
    rho_int4 = (p_int4 / kappa)^(1 / gamma_val)
    p_int5 = 10.0e5
    rho_int5 = (p_int5 / kappa)^(1 / gamma_val)
    p_int6 = 100.0e5
    rho_int6 = (p_int6 / kappa)^(1 / gamma_val)

    A = [rho_int1^2 rho_int1 1; rho_int3^2 rho_int3 1; rho_int4^2 rho_int4 1]
    b = [p_int1; p_int3; p_int4]
    apvals = A \ b
    ap1 = apvals[1]
    ap2 = apvals[2]
    ap3 = apvals[3]

    # Initial values, watch out for dots, so the values are Float64
    u_init = 0.0
    v_init = 0.0
    p_init = p_init_val - p_init_val + p_eps
    gamma_init = -1.0 # see source, volume of fluid method
    rho_init = (p_init / kappa)^(1 / gamma_val)

    output_dict = Dict([
        # mesh properties
        ("N", N),
        ("celltype", celltype),
        ("pressure_inlets", pressure_inlets),
        ("cellthickness", cellthickness),
        ("cellneighbours", cellneighbours),
        ("cellneighboursarray", cellneighboursarray),
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
        ("cellgridid", cellgridid),
        ("coordinates", coordinates),
        ("fvc", fvc),
        ("cellcenters", cellcenters),

        # time parameters
        ("t_max", t_max),
        ("deltat", deltat),
        ("neutral_inds", neutral_inds),
        
        # constants etc.
        ("ap1", ap1),
        ("ap2", ap2),
        ("ap3", ap3),
        ("kappa", kappa),
        ("gamma_val", gamma_val),
        ("p_a", p_a),
        ("rho_a", rho_a),
        ("u_a", u_a),
        ("v_a", v_a),
        ("gamma_a", gamma_a),
        ("p_init", p_init),
        ("rho_init", rho_init),
        ("u_init", u_init),
        ("v_init", v_init),
        ("gamma_init", gamma_init)
    ])
    return output_dict
end

function init(data::Dict)
    N = data["N"]
    pressure_inlets = data["pressure_inlets"]

    u_init = data["u_init"]
    v_init = data["v_init"]
    rho_init = data["rho_init"]
    p_init = data["p_init"]
    gamma_init = data["gamma_init"]

    u_a = data["u_a"]
    v_a = data["v_a"]
    rho_a = data["rho_a"]
    p_a = data["p_a"]
    gamma_a = data["gamma_a"]

    deltat = data["deltat"]

    # allocate & initialize arrays


    # state of simulation is contained in the following variables
    u = fill(u_init, N)
    v = fill(v_init, N)
    rho = fill(rho_init, N)
    p = fill(p_init, N)
    gamma = fill(gamma_init, N)

    t = 0.0
    t_next = 0.0
    iter = 0

    for inlet in range(1, Config.num_inlets)
        for ind in range(1, length(pressure_inlets[inlet, :]))
            cell = pressure_inlets[inlet, ind]
            if cell != -1
                u[cell] = u_a
                v[cell] = v_a
                rho[cell] = rho_a
                p[cell] = p_a
                gamma[cell] = gamma_a
            end
        end
    end

    old_action = zeros(Float32, Config.num_inlets)

    # Allocate arrays
    # will be overwritten every step, just preallocate here
    Δp_old = zeros(Float64, 2, N)
    F_rho_num = zeros(Float64, N)
    F_u_num = zeros(Float64, N)
    F_v_num = zeros(Float64, N)
    F_gamma_num = zeros(Float64, N)
    F_gamma_num1 = zeros(Float64, N)

    uvec = zeros(Float64, 2)
    bvec = zeros(Float64, 3)
    temp = zeros(Float64, N)
    temp2 = zeros(Float64, N)
    temp_mat = zeros(Float64, (3, 2))

    sim_state_dict = Dict([
        ("t", t),
        ("t_next", t_next),
        ("deltat", deltat),
        ("iter", iter),
        ("u", u),
        ("v", v),
        ("p", p),
        ("gamma", gamma),
        ("rho", rho),
        ("old_action", old_action),
        ("Δp_old", Δp_old),
        ("F_rho_num", F_rho_num),
        ("F_u_num", F_u_num),
        ("F_v_num", F_v_num),
        ("F_gamma_num", F_gamma_num),
        ("F_gamma_num1", F_gamma_num1),
        ("uvec", uvec),
        ("bvec", bvec),
        ("temp", temp),
        ("temp2", temp2),
        ("temp_mat", temp_mat)  
        ])
    return sim_state_dict
end

function step(pressure_vals::Vector{Float64}, data::Dict, state::Dict)
    # mesh properties
    N = data["N"]
    celltype = data["celltype"]
    num_neighbours = data["num_neighbours"]
    cellneighbours = data["cellneighbours"]
    cellneighboursarray = data["cellneighboursarray"]
    Amats = data["Amats"]
    Tmats = data["Tmats"]
    cellfacearea = data["cellfacearea"]
    cellfacenormal = data["cellfacenormal"]
    cellviscosity = data["cellviscosity"]
    cellpermeability = data["cellpermeability"]
    cellporosity = data["cellporosity"]
    cellalpha = data["cellalpha"]
    cellvolume = data["cellvolume"]
    cellthickness = data["cellthickness"]
    pressure_inlets::Matrix{Int64} = data["pressure_inlets"]

    # time parameters
    t_max = data["t_max"]
    neutral_inds = data["neutral_inds"]

    # constants etc.
    ap1 = data["ap1"]
    ap2 = data["ap2"]
    ap3 = data["ap3"]
    kappa = data["kappa"]
    gamma_val = data["gamma_val"]
    p_a = data["p_a"]
    u_a = data["u_a"]
    v_a = data["v_a"]
    gamma_a = data["gamma_a"]

    # load sim state
    t = state["t"]
    t_next = state["t_next"]
    deltat = state["deltat"]
    iter = state["iter"]
    u = state["u"]
    v = state["v"]
    p = state["p"]
    gamma = state["gamma"]
    rho = state["rho"]
    old_action = state["old_action"]

    # preallocated arrays for temporary variables
    Δp_old = state["Δp_old"]
    F_rho_num = state["F_rho_num"]
    F_u_num = state["F_u_num"]
    F_v_num = state["F_v_num"]
    F_gamma_num = state["F_gamma_num"]
    F_gamma_num1 = state["F_gamma_num1"]
    uvec = state["uvec"]
    bvec = state["bvec"]
    temp = state["temp"]
    temp2 = state["temp2"]
    temp_mat = state["temp_mat"]

    # set pressure inputs
    # ASSUMPTIONS
    min_pressure = Config.p_min
    max_pressure = Config.p_max

    for inlet in range(1, Config.num_inlets)
        for ind in range(1, length(pressure_inlets[inlet, :]))
            cell = pressure_inlets[inlet, ind]
            if cell != -1
                u[cell] = u_a
                v[cell] = v_a
                p[cell] = (pressure_vals[inlet] * (max_pressure - min_pressure) + min_pressure)
                rho[cell] = (p[cell] / kappa)^(1 / gamma_val)
                gamma[cell] = gamma_a
            end
        end
    end

    # Adjust/reduce time step since locally pressure gradient will increase
    old_p = old_action * (max_pressure - min_pressure) .+ min_pressure
    new_p = pressure_vals * (max_pressure - min_pressure) .+ min_pressure
    deltat = min(deltat / maximum(new_p ./ old_p), deltat)
    old_action = pressure_vals

    # update target time for this step
    t_next += Config.t_step

    # call solver
    finished, t, iter, deltat = MySolver.step2next!(
        t_next,
        Config.t_step,
        t,
        t_max,
        iter,
        deltat,
        neutral_inds,

        # mesh properties
        N,
        celltype,
        num_neighbours,
        cellneighbours,
        cellneighboursarray,
        Amats,
        Tmats,
        cellfacearea,
        cellfacenormal,
        cellviscosity,
        cellpermeability,
        cellporosity,
        cellalpha,
        cellvolume,
        cellthickness,

        # simulation state
        p,
        gamma,
        u,
        v,
        rho,

        # additional parameters
        ap1,
        ap2,
        ap3,
        gamma_val,
        gamma_a, # boundary filling factor

        # preallocations for temporary values
        Δp_old,
        F_rho_num,
        F_u_num,
        F_v_num,
        F_gamma_num,
        F_gamma_num1,
        uvec,
        bvec, 
        temp, 
        temp2,
        temp_mat
    )

    # save state
    # Note that the field values (gamma_old etc.) were already overwritten in the function call before
    state["t"] = t
    state["t_next"] = t_next
    state["deltat"] = deltat
    state["iter"] = iter
    state["old_action"] = old_action

    return finished, state
end

end