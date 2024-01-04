module Paths
using Sockets
export ip, addr, shared_fs, sim_storage, config_path, testset_path, validationset_path, origin_mesh_path

ip_ = string(Sockets.getipaddr(IPv4))

const addr = last(split(ip_, ".")) # last triple of IPv4 used to identify different sim servers in use
const shared_fs = "/cfs/file_exchange/"
const sim_storage = "/sim_storage/"
const testset_path = "/cfs/testsets/"
const validationset_path = "/cfs/validationsets/"
const origin_mesh_path = "/cfs/origin_meshes/"
const config_path = "/cfs/configs/"
const ip = ip_
end