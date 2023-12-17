using FileIO

f = "/cfs/home/h/e/heberleo/BA/RtmServer/ressources/sim/2022-06-03_15-38-06_3.jld2"

l = [f]

p = "/cfs/home/h/e/heberleo/BA/RtmServer/ressources/sim" * "/prepared_2022-06-03_15-38-06_onefile3.jld2"

save(p, "filelist", l)