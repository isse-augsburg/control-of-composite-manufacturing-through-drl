module Config
using CSV

# CONFIG PATH
global config = ""

data_source = ""
filelist_path = ""
config = ""

# sim parameters
t_step = 0.
p_min = 0.
p_max = 0.

# mesh generation parameters
origin_mesh = ""
mesh_x = 50
mesh_y = 50
fvc_normal = 0.
fvc_patch = 0.
pressure_inlets = [] 
num_inlets = 0


function load_config(path::String)
    @assert path != "", "No valid config path set. Call load_config with a valid path before calling any refresh functions."
    global config = path
    global f = CSV.File(open(path))

    # prepared file stuff
    global data_source = f["data_source"][1]
    global filelist_path = f["filelist"][1]
    global origin_mesh = f["origin_mesh"][1]

    # sim parameters
    global t_step = f["t_step"][1]
    global p_min = f["p_min"][1]
    global p_max = f["p_max"][1]

    # mesh generation parameters
    global fvc_normal = f["fvc_normal"][1]
    global fvc_patch = f["fvc_patch"][1]
    global num_inlets = f["inlets"][1]

    if num_inlets == 7
        global pressure_inlets = [1 41 43 -1; 3 5 7 -1; 9 11 13 -1; 15 17 19 21; -1 23 25 27; 29 31 33 -1; 35 37 39 -1] # 50x50 mesh 7 inlets
    elseif num_inlets == 5
        global pressure_inlets = [1 41 43 3 5; 7 9 11 13 -1; 15 17 19 21 -1;23 25 27 29 -1;31 33 35 37 39] # 50x50 mesh 5 inlets
    elseif num_inlets == 3
        global pressure_inlets = [1 41 43 3 5 7 9 -1; 11 13 15 17 19 21 23 25; 27 29 31 33 35 37 39 -1] # 50x50 mesh 3 inlets
    end
end

end