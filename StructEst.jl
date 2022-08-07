include("./ContractMatching/ContractMatching.jl")

# para1 = InternalCandi.para()
# est_para_initial = [8.0, 3.2, 8.0*1.2847, 3.2, 1.927, 3.602, 3.142, 4.152, -1.37, 1.5, -1.5/0.68, 0.41, 0.29, -0.001, 0.151, -1.66, -2.12, 0.001, -400.0, 14.5]
# moments1, err = InternalCandi.genMoments(est_para = est_para_initial)
# score_initial = score(est_para_initial, dataMoment, WeightMatrix)

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


function StructEst(;MaxTime::Float64=120.0) 
	# MaxTime optimizer: the maximum running time in seconds 
	
	println("\n 1. Set Up: loading data moments and weight matrix")
	println("We have dataMoment, momentName, and WeightMatrix")
	println("------------------------------------ \n")

		include("loadMatrix.jl")

	println("\n 2. Minimization: Set Up \n")
	println("------------------------------------ \n")
	
		open("estimation_log.csv", "w") do f
		     write(f, "score, not_convergent, λi, λe, γi, γe, ai, bi, ae, be, ρ, Δβ1_in, β2_in, β3_in, β4_in, β5_in, γ1_in, γ3_in, γ4_in, γ5_in, κ0_in, κ1_in, Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, Δκ0_ex, Δκ1_ex, mean_search, var_search, mean_internal, var_internal, mean_delta_in, mean_delta_ex, var_delta_in, var_delta_ex, mean_explicit_in, mean_explicit_ex, mean_atwill_in, mean_atwill_ex, cov_delta_explicit_in, cov_delta_explicit_ex, cov_delta_atwill_in, cov_delta_atwill_ex, cov_explicit_atwill_in, cov_explicit_atwill_ex, cov_delta_search_in, cov_delta_search_ex, cov_explicit_search_in, cov_explicit_search_ex, cov_atwill_search_in, cov_atwill_search_ex, mean_long_in, mean_long_ex, cov_long_delta_in, cov_long_delta_ex, cov_long_explicit_in, cov_long_explicit_ex, cov_long_atwill_in, cov_long_atwill_ex, cov_long_internal, cov_long_search_in, cov_long_search_ex, score \n") 
		 end

		 open("log_err", "w") do f
			write(f, "ERROR LOG \n")
	      end
	
		println("Suggested starting values")
		est_para_name = [
						"λi", "λe", "γi", "γe", "ai", "bi", "ae", "be", "ρ", 
						"Δβ1_in", "β2_in", "β3_in", "β4_in", "β5_in", "γ1_in", "γ3_in", "γ4_in", "γ5_in", "κ0_in", "κ1_in",
						"Δβ1_ex", "Δβ2_ex", "Δβ3_ex", "Δβ4_ex", "Δβ5_ex", "Δγ1_ex", "Δγ3_ex", "Δγ4_ex", "Δγ5_ex", "Δκ0_ex", "Δκ1_ex"
						# "β1_ex", "β2_ex", "β3_ex", "β4_ex", "β5_ex", "γ1_ex", "γ3_ex", "γ4_ex", "γ5_ex", "κ0_ex", "κ1_ex",
						]

        # est_para_initial = [8.0, 3.2, 8.0*1.2847, 3.2, 1.927, 3.602, 3.142, 4.152, -1.37, 1.5, -1.5/0.68, 0.41, 0.29, -0.001, 0.151, -1.66, -2.12, 0.001, -400.0, 14.5]
        # est_para_initial = [12.923, 10.4798, 18.4656, 10.8253, 5.52592, 7.01597, 6.56913, 0.226888, -1.40314, 1.98749, -6.4091, 2.96237, -1.93548, 0.444441, 2.79386, 3.29856, -1.13019, 1.37469, -56.6772, 3.91465]
        # est_para_initial = [3.65803, 2.72272, 4.94808, 17.0181, 4.11957, 5.3761, 4.04063, 2.58149, -0.547694, 4.59406, -6.87669, 0.290423, -0.793224, -0.240003, 0.479697, -2.84134, 4.54637, 1.99139, -153.041, 3.6081]
        # est_para_initial = [5.54431, 14.9405, 8.01613, 1.35146, 1.16238, 6.70227, 3.12045, 3.46072, -0.526669, 3.95762, -6.09421, 0.347944, -0.810893, -0.470389, 1.20059, -3.47155, 4.24721, -5.86905, -147.622, 43.5689]

        est_para_initial = [
				   # common parameters λi, λe, γi, γe, ai, bi, ae, be, ρ
				   5.54431, 14.9405, 8.01613, 1.35146, 1.16238, 6.70227, 3.12045, 3.46072, -0.526669,
				   # internal Δβ1, β2, Δβ3, Δβ4, Δβ5, γ1, γ3, γ4, γ5, κ0, κ1
				   0.65, -6.09421, 6.44, 5.29, 6.08, 1.20059, -3.47155, 4.24721, -5.86905, -147.622, 43.5689, 
				   # external Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex
				   0.65, 1.0, 6.44, 5.29, 6.08, 1.0, 1.0, 1.0, 1.0, -147.622, 43.5689
				   ]


        # score_initial = score(est_para_initial, dataMoment, WeightMatrix, diagnosis = true)
        score_initial = score(est_para_initial, dataMoment, WeightMatrixDiag, diagnosis = false)

        println("The initial score is $score_initial.")
		display([est_para_name est_para_initial])
		println()

		# set bounds on parameters
		tBounds = [ 
					#bounds,  est_para,  model_para
					(3.0, 20.0),   # λi,  λi
					(1.0, 20.0),   # λe,  λe
					(3.0, 20.0),   # γi,  γi
					(2.0, 20.0),   # γe,  γe
					(2.0, 15.0),   # ai,  ai
					(2.0, 15.0),   # bi,  bi
					(1.0, 10.0),   # ae,  ae
					(1.0, 15.0),   # be,  be
					(-1.0, -0.05), # ρ,  ρ
					(0.5, 0.9),    # Δβ1,  β1 = -β2 * Δβ1
					(-10.0, -1.0),  # β2,  β2
					(0.1, 30.0),   # Δβ3, β3 = β2 + Δβ3
					(0.1, 30.0),   # Δβ4, β4 = β2 + Δβ4
					(0.1, 30.0),   # Δβ5, β5 = β2 + β3 + β4 + Δβ5
					(0.1, 25.00),  # γ1, γ1
					(-25.00, 25.00),  # γ3, γ3
					(-25.00, 25.00),  # γ4, γ4
					(-25.00, 25.00),  # γ5, γ5
					(-250.0, 100.0),  # κ0, κ0
					(0.01, 80.00),    # κ1, κ1
					(0.5, 0.9),       # Δβ1_ex, β1_ex = - β2_ex * Δβ1_ex
					(0.5, 1.5),       # Δβ2_ex, β2_ex = β2 * Δβ2
					(0.1, 30.0),      # Δβ3_ex, β3_ex = β2_ex + Δβ3_ex
					(0.1, 30.0),      # Δβ4_ex, β4_ex = β2_ex + Δβ4_ex
					(0.1, 30.0),      # Δβ5_ex, β5_ex = β2_ex + β3_ex + β4_ex + Δβ5_ex
					(0.5, 1.5),       # Δγ1_ex, γ1_ex = γ1 * Δγ1_ex
					(0.5, 1.5),       # Δγ3_ex, γ3_ex = γ3 * Δγ3_ex
					(0.5, 1.5),       # Δγ4_ex, γ4_ex = γ4 * Δγ4_ex
					(0.5, 1.5),       # Δγ5_ex, γ5_ex = γ5 * Δγ5_ex
					(-250.0, 100.0),  # κ0_ex, κ0_ex
					(0.01, 80.00),    # κ1_ex, κ1_ex
								] 

        # set lambda: number of samples to take per iteration, here means each process take 3 samples, could also be 4 or 5
        # if lambda = 0 is set, then it will be set based on the number of dimensions
        # default value of lambda (when it is set as 0), lambda = 4 + ceil(Int, log(3*d))
        # nParticles =  (nprocs()-1) * 3  # number of samples to take per iteration
	    # MaxTime = 1 * 60.0  # in seconds and the number before * is the minutes
		# other options
		# TraceInterval= .1, 
		# PopulationSize = 10.,
		# SaveFitnessTraceToCsv = true, 
		# SaveParameters = true,

		opt = bbsetup(θ -> score(θ, dataMoment, WeightMatrix, diagnosis = false);
					  # Method = :separable_nes, 
					  Method = :adaptive_de_rand_1_bin_radiuslimited,
					  SearchRange = tBounds, 
					  ini_x = est_para_initial,
					  TraceMode = :verbose, 
					  MinDeltaFitnessTolerance=1e-4,
					  SaveTrace = true, 
					  SaveParameters = true, 
					  MaxTime = MaxTime,   # MaxTime = 40*1.0,
					  # lambda = nParticles, # number of samples to take per iteration
					  # max_sigma = 100.,   # Maximal sigma, default max_sigma => 1.0E+10
					  # Workers = workers(),
					  ) 
		
   
	println("3. Minimization: going on ... ")
		println("------------------------------------ \n")
		res = bboptimize(opt)
		# get moments and covariance matrix at optimal theta
		ThetaStar = res.archive_output.best_candidate
		println("------------------------------------ \n")

	println("4. Inference \n")
	println("------------------------------------ \n")

	println("\n 4.1 Compare Model and Data Moments \n")
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
