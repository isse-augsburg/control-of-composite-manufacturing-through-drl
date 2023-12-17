module Log

export  log

function log(msg::String)
    println(msg)
    flush(stdout)
end

function log(vec::Vector{Any})
    for i in 1:length(vec)
        println(vec[i])
    end
    flush(stdout)
end

function log(vec::Vector{String})
    for i in 1:length(vec)
        println(vec[i])
    end
    flush(stdout)
end

end