FROM julia:latest

RUN mkdir /cfs

RUN mkdir /sim_storage

RUN julia -e 'using Pkg; Pkg.add(["Revise", "HDF5", "HTTP", "JSON3", "StatsBase", "JLD2", "ImageCore", "ImageIO", "ImageMagick", "FileIO", "Random", "CSV"]); pkg"precompile"; pkg"instantiate"' \
&& julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.update(); ENV["PYTHON"]=""; Pkg.build("PyCall")' \
&& julia -e 'using Pkg; Pkg.add("Conda")' \
&& julia -e 'using Pkg; Pkg.build("Conda"); Pkg.build("PyCall")' \
&& julia -e 'using Conda; Conda.add("numpy"); Conda.add("scipy")'

ENV NENVS=1
ENV CONFIG_NAME=""
RUN mkdir /sourcecode

# copy source code
# COPY ./src/ /src

ENTRYPOINT julia -p $NENVS /sourcecode/RtmServer.jl $CONFIG_NAME


