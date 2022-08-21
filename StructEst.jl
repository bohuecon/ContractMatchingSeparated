include("./ContractMatching/ContractMatching.jl")

using JLD
using DataFrames
using Distributions
using BlackBoxOptim
using LinearAlgebra


function score(est_para::Vector, dataMoment::Vector, WeightMatrix::Matrix; diagnosis = false)

    try 
		modelMoment, error_flag, not_convergent = ContractMatching.genMoments(est_para = est_para, diagnosis = diagnosis)
		# modelMoment, error_flag, not_convergent = ContractMatching.genMoments(est_para = est_para_initial, diagnosis = false)

		if error_flag
			# if either an explicit error by error_flag or an implicit error by NaN is simuMoment
			thescore = Inf 
		else
			thescore = (dataMoment .- modelMoment)' * WeightMatrix * (dataMoment .- modelMoment)
		end

		# if thescore < 100000.0

			open("estimation_log.csv", "a") do f
			     # write(f, "ee, rho_z, mean_z, var_z, mean_logdelta, mean_logwage, mean_logsize, beta_wagesize, beta_deltasize, beta_deltawage, valid_fraction, lambda1_val, lambda1, z_rho, zw_mean, z_std, s_mean, s_std, c, sigma, score \n")

				write(f, "$(round(thescore, digits = 4)),")
				write(f, "$not_convergent,")
		        
		        for i in 1:length(est_para)
		         	write(f, "$(round(est_para[i], digits = 4)),")
		        end
			    
			    for i in 1:length(modelMoment)
		         	write(f, "$(round(modelMoment[i], digits = 4)),")
		        end

		        write(f, "$(round(thescore, digits = 4)) \n")

			 end

		# end

		if diagnosis
	        println("       >>> score is $thescore.")
	    end

	    return thescore

    catch err
		open("log_err", "a") do f
			write(f, "\n _____________________________________________ \n")
			write(f, "parameters: $est_para; \n \n")
			write(f, "error: $err; \n \n ")
	      end
	    return Inf
	end

end


function StructEst(;
	SearchMethod = :adaptive_de_rand_1_bin_radiuslimited,
	MaxTime::Float64=120.0) # optimizer: the maximum running time in seconds 
	
	println("\n 1. Set Up: loading data moments and weight matrix")
	println("We have dataMoment, momentName, and WeightMatrix")
	println("------------------------------------ \n")

		include("loadMatrix.jl")

	println("\n 2. Minimization: Set Up \n")
	println("------------------------------------ \n")
	
	# λi, λe, γi, γe, ai, bi, ae, be, ρ, 

		open("estimation_log.csv", "w") do f
		     write(f, "score, not_convergent, Δβ1_in, β2_in, β3_in, β4_in, β5_in, γ1_in, γ3_in, γ4_in, γ5_in, κ0_in, κ1_in, Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, Δκ0_ex, Δκ1_ex, mean_search, var_search, mean_internal, var_internal, mean_delta_in, mean_delta_ex, var_delta_in, var_delta_ex, mean_explicit_in, mean_explicit_ex, mean_atwill_in, mean_atwill_ex, cov_delta_explicit_in, cov_delta_explicit_ex, cov_delta_atwill_in, cov_delta_atwill_ex, cov_explicit_atwill_in, cov_explicit_atwill_ex, cov_delta_search_in, cov_delta_search_ex, cov_explicit_search_in, cov_explicit_search_ex, cov_atwill_search_in, cov_atwill_search_ex, mean_long_in, mean_long_ex, cov_long_delta_in, cov_long_delta_ex, cov_long_explicit_in, cov_long_explicit_ex, cov_long_atwill_in, cov_long_atwill_ex, cov_long_internal, cov_long_search_in, cov_long_search_ex, score \n") 
		 end

		 open("log_err", "w") do f
			write(f, "ERROR LOG \n")
	      end
	
		println("Suggested starting values")
		est_para_name = [
						# "λi", "λe", "γi", "γe", "ai", "bi", "ae", "be", "ρ", 
						"Δβ1_in", "β2_in", "β3_in", "β4_in", "β5_in", "γ1_in", "γ3_in", "γ4_in", "γ5_in", "κ0_in", "κ1_in",
						"Δβ1_ex", "Δβ2_ex", "Δβ3_ex", "Δβ4_ex", "Δβ5_ex", "Δγ1_ex", "Δγ3_ex", "Δγ4_ex", "Δγ5_ex", "κ0_ex", "κ1_ex"
						# "β1_ex", "β2_ex", "β3_ex", "β4_ex", "β5_ex", "γ1_ex", "γ3_ex", "γ4_ex", "γ5_ex", "κ0_ex", "κ1_ex",
						]

        est_para_initial = [
				   # common parameters λi, λe, γi, γe, ai, bi, ae, be, ρ
				   # 7.9669, 19.4285, 11.3733, 9.6905, 3.45, 9.68, 7.30, 5.00, -0.12,
				   # internal β1, β2, β3, β4, β5, γ1, γ3, γ4, γ5, κ0, κ1
				   0.8572, -1.49635, 0.0138, -0.3717, 0.0191, 0.995683, -0.253625, 0.992825, 0.999949, -2.90806, 0.225924,
				   # external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
				   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.90806, 0.225924
				   ]

		para_initial = ContractMatching.est_para2model_para(est_para_initial)

		display([est_para_name est_para_initial])
		
        score_initial = score(est_para_initial, dataMoment, WeightMatrix, diagnosis = true)
        println("\n The initial score is $score_initial. \n")
   
	println("\n 3. Minimization: going on ... ")
		println("------------------------------------ \n")

		tBounds = [ 
			#bounds,  est_para,  model_para
			# (7.966, 7.967),   # λi,  λi
			# (6.428, 6.429),   # λe,  λe
			# (11.373, 11.374),   # γi,  γi
			# (16.690, 16.691),   # γe,  γe
			# (4.361, 4.362),   # ai,  ai
			# (2.0, 15.0),   # bi,  bi
			# (1.0, 10.0),   # ae,  ae
			# (1.0, 15.0),   # be,  be
			# (-1.0, -0.05), # ρ,  ρ
			# (0.5, 0.9),    # Δβ1,  β1 = -β2 * Δβ1
			# (-10.0, -6.0),  # β2,  β2
			# (5.0, 10.0),   # Δβ3, β3 = β2 + Δβ3
			# (0.1, 10.0),   # Δβ4, β4 = β2 + Δβ4
			# (0.1, 25.0),   # Δβ5, β5 = β2 - β3 - β4 + Δβ5
			(0.0, 1.0),    # β1
			(-2.0, 0.0),   # β2
			(-1.0, 1.0),   # β3
			(-1.0, 1.0),   # β4
			(-1.0, 1.0),   # β5
			(0.0, 2.5),    # γ1
			(-1.0, 2.5),   # γ3
			(-1.0, 2.5),   # γ4
			(-1.0, 2.5),   # γ5
			(-5.0, 0.0),   # κ0
			(0.001, 1.00), # κ1
			(0.8, 1.2),       # Δβ1_ex, β1_ex = β1 * Δβ1_ex
			(0.8, 1.2),       # Δβ2_ex, β2_ex = β2 * Δβ2
			(0.8, 1.2),       # Δβ3_ex, β3_ex = β3 * Δβ3_ex
			(0.8, 1.2),       # Δβ4_ex, β4_ex = β4 * Δβ4_ex
			(0.8, 1.2),       # Δβ5_ex, β5_ex = β5 * Δβ5_ex
			(0.8, 1.2),       # Δγ1_ex, γ1_ex = γ1 * Δγ1_ex
			(0.8, 1.2),       # Δγ3_ex, γ3_ex = γ3 * Δγ3_ex
			(0.8, 1.2),       # Δγ4_ex, γ4_ex = γ4 * Δγ4_ex
			(0.8, 1.2),       # Δγ5_ex, γ5_ex = γ5 * Δγ5_ex
			(-5.0, 0.0),      # κ0_ex, κ0_ex
			(0.001, 1.00),    # κ1_ex, κ1_ex
		] 

		score_obj(θ) = score(θ, dataMoment, WeightMatrix, diagnosis = false)
		res = bboptimize(score_obj, est_para_initial; Method = SearchMethod, SearchRange = tBounds, TraceMode = :verbose, MinDeltaFitnessTolerance=1e-4, SaveTrace = true, SaveParameters = true, MaxTime = MaxTime)
		ThetaStar = res.archive_output.best_candidate
		println("------------------------------------ \n")

	println("\n 4. Inference ")
	println("------------------------------------ \n")

	println("\n 4.1 Compare Model and Data Moments")
	println("------------------------------------ \n")

	# ThetaStar = est_para_initial;

	modelMomentThetaStar, error_flag = ContractMatching.genMoments(est_para = ThetaStar);
	stdErr = diag(CovDataMoment).^0.5;
	tStat = (modelMomentThetaStar - dataMoment) ./ stdErr;
	dfMoment = DataFrame(Moments = momentName,  Data = dataMoment, Model = modelMomentThetaStar, tStat = tStat);
	show(dfMoment, allcols = true)
	println()


	# println("\n 4.2 Gradient and Local Identification \n")
	# println("------------------------------------ \n")

	# mGrad = getgradient(ThetaStar, dataMoment)
	# println("d m_star / d ParamsEst =")
	# dfGrad = DataFrame(mGrad, :auto)
	# show(dfGrad)
	# println("\n \n")
	# println("Check for local identification: Does gradient have full rank? \n")
	# # it means it should have a rank of the parameters to be estimated.
	# println("Rank of d m_star / d Params = ", string(rank(mGrad)) )
	# println("Condition number of d m_star / d Params = ", string(cond(mGrad)) )
	# println()

	# println("\n 4.3 Estimates and Standard errors \n")
	# println("------------------------------------ \n")

	# # Nsim = 10 #CHANGE THIS TO APPROPRIATE VALUE
	# # Sum of variance of empirical and simulated moments:
	# # CovM = CovDataMoment + (1/Nsim)*CovDataMoment
	# # CovM = CovDataMoment

	# # Chunk = mGrad' * WeightMatrix * mGrad
	# # XXX = Chunk\mGrad' # A\B = inv(A)*B

	# # CovarEst = XXX * WeightMatrix * CovM * WeightMatrix' * XXX'

 #    CovarEst = inv(mGrad' * WeightMatrix * mGrad)/709

	# StdErrsEst = diag(CovarEst).^0.5

	# println("Estimated Parameters: \n")
	# dfTheta= DataFrame(parameter = est_para_name, estimates = ThetaStar, StdErr = StdErrsEst, t_val = ThetaStar./StdErrsEst)
	# show(dfTheta)
	# println()

	# # println("\n 4.4 Over-identification test \n")
	# # println("------------------------------------ \n")

 #    nDim = length(ThetaStar)
	# nMom = length(dataMoment)
	# Chunk2 = eye(nMom) - mGrad * XXX * WeightMatrix
	# g = dataMoment - simuMomentThetaStar
	# ChiStat = g' * pinv(Chunk2 * CovM * Chunk2') * g
	# Chip = 1 - cdf(Chisq(nMom-nDim), ChiStat) # prob true chi is less than Chi
	# println("Chi-squared statistic is $ChiStat, with p-value = $Chip")
	# println()

end


# function getgradient(ThetaStar::Vector,dataMoment::Vector)

# 	nDim = length(ThetaStar)
# 	nMom = length(dataMoment)
# 	mGrad = fill(NaN, (nMom, nDim))

# 	for i=1:nDim

# 		# h = Delta for derivative. h= a small number
# 	    h = abs(ThetaStar[i])*0.01; # See Judd page 281, "two-sided differences"

# 		# if i in [19, 20]
# 		#     h = abs(ThetaStar[i])*0.5; 
# 		# elseif i in [14, 18]
# 		# 	h = abs(ThetaStar[i])*10.0;
# 		# else
# 		#     h = abs(ThetaStar[i])*0.01; 
# 		# end

# 	    ThetaStar_U = copy(ThetaStar)
# 	    ThetaStar_D = copy(ThetaStar)

# 	    # Shock parameter i up and down by h:
# 	    ThetaStar_U[i] = ThetaStar_U[i] + h;
# 	    ThetaStar_D[i] = ThetaStar_D[i] - h;

# 	    # UP
# 	    modelMoment_U, error_flag_U, not_convergent = InternalCandi.genMoments(est_para = ThetaStar_U)

# 	    # DOWN
# 	    modelMoment_D, error_flag_D, not_convergent = InternalCandi.genMoments(est_para = ThetaStar_D)

# 	    # Formula for two-sided numerical derivative:
# 	    mGrad[:, i] = (modelMoment_U - modelMoment_D) ./ (2*h)
# 	end

# 	return mGrad
# end
