module Paths
using Sockets
export ip, addr, shared_fs, sim_storage

ip_ = string(Sockets.getipaddr(IPv4))

const addr = last(split(ip_, ".")) # last triple of IPv4 used to identify different sim servers in use
const shared_fs = "/cfs/file_exchange/"
const sim_storage = "/sim_storage/"
const ip = ip_
end