# if nprocs() == 1
# 	addprocs(18)
# end

include("StructEst.jl")


# nParticles_val =  (nprocs()-1) * 3  # number of samples to take per iteration
# nParticles_val =  0  # number of samples to take per iteration
MaxTime_val = 5 * 24 * 60.0 * 60.0  # in seconds and the number before * is the minutes
# MaxTime_val = 15 * 60.0 * 60.0  # in seconds and the number before * is the minutes
# MaxTime_val = 1 * 60.0  # in seconds and the number before * is the minutes
@time StructEst(MaxTime=MaxTime_val)