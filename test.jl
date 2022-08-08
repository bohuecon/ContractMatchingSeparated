
using Parameters

include("./ContractMatching/ContractMatching.jl")
include("StructEst.jl")

function test_score(est_para::Vector)
    
    include("loadMatrix.jl")

	println("\n------------------------------------ \n")
	
		println("Suggested starting values")
		est_para_name = [
						# "λi", "λe", "γi", "γe", "ai", "bi", "ae", "be", "ρ", 
						"Δβ1_in", "β2_in", "β3_in", "β4_in", "β5_in", "γ1_in", "γ3_in", "γ4_in", "γ5_in", "κ0_in", "κ1_in",
						"Δβ1_ex", "Δβ2_ex", "Δβ3_ex", "Δβ4_ex", "Δβ5_ex", "Δγ1_ex", "Δγ3_ex", "Δγ4_ex", "Δγ5_ex", "Δκ0_ex", "Δκ1_ex"
						]
				 

	   	display([est_para_name est_para])

   		modelMoment, error_flag, not_convergent = ContractMatching.genMoments(est_para = est_para, diagnosis = true)
		thescore = (dataMoment .- modelMoment)' * WeightMatrixDiag * (dataMoment .- modelMoment)
        thescore2 = score(est_para, dataMoment, WeightMatrixDiag, diagnosis = false)

	println("\n------------------------------------ \n")

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
	# 5.54431, 14.9405, 8.01613, 1.35146, 1.16238, 6.70227, 3.12045, 3.46072, -0.526669,
	# 7.9669, 6.4285, 11.3733, 19.6905, 4.3613, 5.6546, 4.7125, 9.0024, -0.2923,
	# internal Δβ1, β2, Δβ3, Δβ4, Δβ5, γ1, γ3, γ4, γ5, κ0, κ1
	0.5665, -8.2091, 8.7583, 6.6156, 9.0544, 1.6294, 6.6008, -1.7298, -1.3128, -115.4637, 2.1944, 
	# external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
	# the benchmark (same internal and external)
	# 0.5665, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -115.4637, 2.1944, 
	# 0.8876, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -184.2481, 1.7702 
	# different internal and external
	# 0.8876, 0.935, 1.1702, 0.8635, 1.1867, 0.9432, 0.8007, 0.8696, 1.1768, -184.2481, 1.7702
	1.0, 0.935, 1.1702, 0.8635, 1.1867, 0.9432, 0.8007, 0.8696, 1.1768, -184.2481, 1.7702
	# 0.8876, 0.935, 1.1702, 0.8635, 1.1867, 0.9432, 0.8007, 0.8696, 1.1768, -184.2481, 1.7702
	]

para = ContractMatching.est_para2model_para(est_para)

# show the contract parameters
βs = hcat(para.β1, para.β2, para.vec_β ...);
βs_ex = hcat(para.β1_ex, para.β2_ex, para.vec_β_ex ...);
[βs; βs_ex]

γs = hcat(para.γ1, para.vec_γ ...);
γs_ex = hcat(para.γ1_ex, para.vec_γ_ex ...);
[γs; γs_ex]

# Understand the solution to the model
sol, not_convergent = ContractMatching.solve_main(para = para, diagnosis = true)

@unpack mat_Mu, mat_Mu_bm = sol

# there exists some type of firms that can not be matched
internal_not_matched = (minimum(sum(mat_Mu, dims = 2)) == 0) 
external_not_matched = (minimum(sum(mat_Mu_bm, dims = 2)) == 0)

# Compute the score based on sol
test_score(est_para)

