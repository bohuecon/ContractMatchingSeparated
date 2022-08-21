# if nprocs() == 1
# 	addprocs(18)
# end

include("StructEst.jl")


MaxTime_val = 5 * 24 * 60.0 * 60.0  # in seconds and the number before * is the minutes
# MaxTime_val = 15 * 60.0 * 60.0  # in seconds and the number before * is the minutes
# MaxTime_val = 5.0 * 60.0  # in seconds and the number before * is the minutes

# Method_val = :adaptive_de_rand_1_bin_radiuslimited 
# Method_val = :adaptive_de_rand_1_bin 
# Method_val = :xnes
Method_val = :generating_set_search 
# Method_val = :probabilistic_descent 

# possbile options:
# :adaptive_de_rand_1_bin, :de_rand_1_bin_radiuslimited, :adaptive_de_rand_1_bin_radiuslimited, :xnes, :generating_set_search, :dxnes, :de_rand_1_bin, :separable_nes, :resampling_inheritance_memetic_search,

println("Estimation time is $MaxTime_val .")
println("Search method is $Method_val .")

@time StructEst(SearchMethod = Method_val, MaxTime = MaxTime_val)