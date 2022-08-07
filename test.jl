include("./ContractMatching/ContractMatching.jl")
include("StructEst.jl")

# para1 = InternalCandi.para()
# est_para_initial = [8.0, 3.2, 8.0*1.2847, 3.2, 1.927, 3.602, 3.142, 4.152, -1.37, 1.5, -1.5/0.68, 0.41, 0.29, -0.001, 0.151, -1.66, -2.12, 0.001, -400.0, 14.5]
# moments1, err = InternalCandi.genMoments(est_para = est_para_initial)
# score_initial = score(est_para_initial, dataMoment, WeightMatrix)

using JLD
using DataFrames
using Distributions
using BlackBoxOptim
using LinearAlgebra

# function test(;
# 	Δβ1_ex = 0.65, 
# 	Δβ2_ex = 1.0, 
# 	Δβ3_ex = 1.0, 
# 	Δβ4_ex = 1.0, 
# 	Δβ5_ex = 1.0, 
# 	Δγ1_ex = 1.0, 
# 	Δγ3_ex = 1.0, 
# 	Δγ4_ex = 1.0, 
# 	Δγ5_ex = 1.0, 
# 	κ0_ex = -147.622, 
# 	κ1_ex = 43.5689
# 	)
	
# 	println("\n 1. Set Up: loading data moments and weight matrix")
# 	println("We have dataMoment, momentName, and WeightMatrix")
# 	println("------------------------------------ \n")

# 		include("loadMatrix.jl")

# 	println("\n 2. Minimization: Set Up \n")
# 	println("------------------------------------ \n")
	
# 		println("Suggested starting values")
# 		est_para_name = [
# 						"λi", "λe", "γi", "γe", "ai", "bi", "ae", "be", "ρ", 
# 						"Δβ1_in", "β2_in", "β3_in", "β4_in", "β5_in", "γ1_in", "γ3_in", "γ4_in", "γ5_in", "κ0_in", "κ1_in",
# 						"Δβ1_ex", "Δβ2_ex", "Δβ3_ex", "Δβ4_ex", "Δβ5_ex", "Δγ1_ex", "Δγ3_ex", "Δγ4_ex", "Δγ5_ex", "Δκ0_ex", "Δκ1_ex"
# 						]

#         est_para = [
# 				   # common parameters λi, λe, γi, γe, ai, bi, ae, be, ρ
# 				   5.54431, 14.9405, 8.01613, 1.35146, 1.16238, 6.70227, 3.12045, 3.46072, -0.526669,
# 				   # internal Δβ1, β2, β3, β4, β5, γ1, γ3, γ4, γ5, κ0, κ1
# 				   0.65, -6.09421, 0.347944, -0.810893, -0.470389, 1.20059, -3.47155, 4.24721, -5.86905, -147.622, 43.5689, 
# 				   # external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
# 				   # 0.65, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -147.622, 43.5689
# 				   # 0.65, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -147.622, 43.5689
# 				   Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
# 				   ]
				 

# 	   	display([est_para_name est_para])

#    		modelMoment, error_flag, not_convergent = ContractMatching.genMoments(est_para = est_para, diagnosis = true)
# 		thescore = (dataMoment .- modelMoment)' * WeightMatrixDiag * (dataMoment .- modelMoment)
#         thescore2 = score(est_para, dataMoment, WeightMatrixDiag, diagnosis = false)


# 	println("\n 3. Compare Model and Data Moments \n")
# 	println("------------------------------------ \n")

# 	stdErr = diag(CovDataMoment).^0.5;
# 	tStat = (modelMoment - dataMoment) ./ stdErr;
# 	dfMoment = DataFrame(Moments = momentName,  Data = dataMoment, Model = modelMoment, tStat = tStat);
# 	show(dfMoment, allcols = true, allrows = true)
# 	println("")
# 	println("\n The score computed directly is $thescore")
# 	println("The score computed by score function $thescore2.")
# 	println("error_flag = $error_flag, not_convergent = $not_convergent.")

# end

# test(κ0_ex = 1.5)

function test(est_para::Vector)
	
	println("\n 1. Set Up: loading data moments and weight matrix")
	println("We have dataMoment, momentName, and WeightMatrix")
	println("------------------------------------ \n")

		include("loadMatrix.jl")

	println("\n 2. Minimization: Set Up \n")
	println("------------------------------------ \n")
	
		println("Suggested starting values")
		est_para_name = [
						"λi", "λe", "γi", "γe", "ai", "bi", "ae", "be", "ρ", 
						"Δβ1_in", "β2_in", "β3_in", "β4_in", "β5_in", "γ1_in", "γ3_in", "γ4_in", "γ5_in", "κ0_in", "κ1_in",
						"Δβ1_ex", "Δβ2_ex", "Δβ3_ex", "Δβ4_ex", "Δβ5_ex", "Δγ1_ex", "Δγ3_ex", "Δγ4_ex", "Δγ5_ex", "Δκ0_ex", "Δκ1_ex"
						]

       #  est_para = [
				   # # common parameters λi, λe, γi, γe, ai, bi, ae, be, ρ
				   # 5.54431, 14.9405, 8.01613, 1.35146, 1.16238, 6.70227, 3.12045, 3.46072, -0.526669,
				   # # internal Δβ1, β2, β3, β4, β5, γ1, γ3, γ4, γ5, κ0, κ1
				   # 0.65, -6.09421, 0.347944, -0.810893, -0.470389, 1.20059, -3.47155, 4.24721, -5.86905, -147.622, 43.5689, 
				   # # external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
				   # # 0.65, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -147.622, 43.5689
				   # # 0.65, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -147.622, 43.5689
				   # Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
				   # ]
				 

	   	display([est_para_name est_para])

   		modelMoment, error_flag, not_convergent = ContractMatching.genMoments(est_para = est_para, diagnosis = true)
		thescore = (dataMoment .- modelMoment)' * WeightMatrixDiag * (dataMoment .- modelMoment)
        thescore2 = score(est_para, dataMoment, WeightMatrixDiag, diagnosis = false)


	println("\n 3. Compare Model and Data Moments \n")
	println("------------------------------------ \n")

	stdErr = diag(CovDataMoment).^0.5;
	tStat = (modelMoment - dataMoment) ./ stdErr;
	dfMoment = DataFrame(Moments = momentName,  Data = dataMoment, Model = modelMoment, tStat = tStat);
	show(dfMoment, allcols = true, allrows = true)
	println("")
	println("\n The score computed directly is $thescore")
	println("The score computed by score function $thescore2.")
	println("error_flag = $error_flag, not_convergent = $not_convergent.")

end



est_para = [
		   # common parameters λi, λe, γi, γe, ai, bi, ae, be, ρ
		   5.54431, 14.9405, 8.01613, 1.35146, 1.16238, 6.70227, 3.12045, 3.46072, -0.526669,
		   # internal Δβ1, β2, Δβ3, Δβ4, Δβ5, γ1, γ3, γ4, γ5, κ0, κ1
		   0.6494, -6.09421, 6.442154, 5.283317, 6.0868, 1.20059, -3.47155, 4.24721, -5.86905, -147.622, 43.5689, 
		   # external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
		   0.6494, 1.0, 6.442154, 5.283317, 6.0868, 1.0, 1.0, 1.0, 1.0, -147.622, 43.5689
		   ]

# para = est_para2model_para(est_para)

# para.vec_β
# para.vec_β_ex

test(est_para)














