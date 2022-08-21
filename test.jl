
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
	# thescore = (dataMoment .- modelMoment)' * WeightMatrixDiag * (dataMoment .- modelMoment)
	thescore = (dataMoment .- modelMoment)' * WeightMatrix * (dataMoment .- modelMoment)

    # thescore2 = score(est_para, dataMoment, WeightMatrixDiag, diagnosis = false)
    thescore2 = score(est_para, dataMoment, WeightMatrix, diagnosis = false)

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


# est_para = [
# 		   # common parameters λi, λe, γi, γe, ai, bi, ae, be, ρ
# 		   # 7.9669, 19.4285, 11.3733, 9.6905, 3.45, 9.68, 7.30, 5.00, -0.12,
# 		   # internal β1, β2, β3, β4, β5, γ1, γ3, γ4, γ5, κ0, κ1
# 		   0.8572, -1.49635, 0.0138, -0.3717, 0.0191, 0.995683, -0.253625, 0.992825, 0.999949, -2.90806, 0.225924,
# 		   # external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
# 		   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.90806, 0.225924
# 		   ]

est_para = [
		   # common parameters λi, λe, γi, γe, ai, bi, ae, be, ρ
		   # 7.9669, 19.4285, 11.3733, 9.6905, 3.45, 9.68, 7.30, 5.00, -0.12,
		   # β1, β2, β3, β4, β5
            0.8766, -1.9993, 0.0138, -0.3717, 0.0191,
		   # γ1, γ3, γ4, γ5, κ0, κ1
		    0.9957, -0.2536, 0.9928, 0.9999, -2.9081, 0.2259,
		   # external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
		   1.0, 1.0, .0016, 1.0, 1.0051, 1.0, 1.0, 0.8, 1.0, -2.90806, 0.2259
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


minimum(mat_Mu - mat_Mu_bm)

