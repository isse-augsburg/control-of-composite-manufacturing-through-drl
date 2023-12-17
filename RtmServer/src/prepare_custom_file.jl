using FileIO
include("MyMesh.jl")
include("RtmSimulation.jl")

# origin = "/cfs/share/data/RTM/LinearInjection/sim_out/output/with_shapes/2022-06-03_15-38-06_1000p/3/2022-06-03_15-38-06_3_RESULT.erfh5"
origin = "Y:\\data\\RTM\\LinearInjection\\sim_out\\output\\with_shapes\\2022-06-03_15-38-06_1000p\\3\\2022-06-03_15-38-06_3_RESULT.erfh5"


# custom values for permeability & porosity are set in the mesh creation
mesh = MyMesh.load_RtmMesh(origin)

data = RtmSimulation.prepare(mesh)
state = RtmSimulation.init(data)

# save_path = "/cfs/home/h/e/heberleo/BA/RtmServer/ressources/sim/"
save_path = "X:\\h\\e\\heberleo\\BA\\RtmServer\\ressources\\onefile_runs\\"
newname = "2022-06-03_15-38-06_3_fvc0304"

f = save_path * newname * ".jld2"
# f = save_path * "\\" * newname * ".jld2"
save(f, "data", data, "state", state)

unix_path = "/cfs/home/h/e/heberleo/BA/RtmServer/ressources/onefile_runs/" * newname * ".jld2"
l = [unix_path]

# p = "/cfs/home/h/e/heberleo/BA/RtmServer/ressources/sim" * "/onefile.jld2"
p = "X:\\h\\e\\heberleo\\BA\\RtmServer\\ressources\\onefile_runs" * "\\onefile.jld2"
save(p, "filelist", l)
