FROM julia:latest

RUN mkdir /cfs
RUN mkdir /cfs/file_exchange

RUN mkdir /sim_storage

RUN julia -e 'using Pkg; Pkg.add(["Revise", "HDF5", "HTTP", "JSON3", "StatsBase", "JLD2", "ImageCore", "ImageIO", "FileIO", "Random", "CSV"]); pkg"precompile"' \
&& julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.update(); ENV["PYTHON"]=""; Pkg.build("PyCall")' \
&& julia -e 'using Pkg; Pkg.add("Conda")' \
&& julia -e 'using Pkg; Pkg.build("Conda"); Pkg.build("PyCall")' \
&& julia -e 'using Conda; Conda.add("numpy"); Conda.add("scipy")'

# copy source code
COPY ./src/ /src

ENV NENVS=1
ENV CONFIG_PATH=""
ENTRYPOINT julia -p $NENVS /src/RtmServer.jl $CONFIG_PATH


