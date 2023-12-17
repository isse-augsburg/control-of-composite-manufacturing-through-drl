module Paths
using Sockets
export ip, addr, shared_fs, sim_storage

ip_ = string(Sockets.getipaddr(IPv4))


const addr = ip_[11:12] # last triple of IPv4 used to identify different sim servers in use
const shared_fs = "/cfs/home/h/e/heberleo/BA/file_exchange/"
const sim_storage = "/cfs/home/h/e/heberleo/BA/RtmServer/sim_storage/"
const ip = ip_

end