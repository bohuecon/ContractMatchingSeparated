
    println("\n Moments and Weights from Data")
    println("-------------------------------- \n")

	# load matrix of all moments
	dataMoment = load("./dataMoment.jld", "dataMoment");
	momentName = load("./dataMoment.jld", "momentName");
	CovDataMoment = load("./dataMoment.jld", "CovMoment");
	WeightMatrix = load("./dataMoment.jld", "WeightMatrix");
	WeightMatrixDiag = load("./dataMoment.jld", "WeightMatrixDiag");
	# WeightMatrix = WeightMatrix / minimum(diag(WeightMatrix))
	# WeightMatrixDiag = WeightMatrixDiag / minimum(diag(WeightMatrixDiag))


	# select moments
	# col = Dict();
	# for (number, name) in enumerate(momentName_all)
	#    col[name] = number
	# end

	# # select matrix of used moments
	# momentNames = ["mean_search", "var_search", "mean_internal", "var_internal", "mean_delta", "var_delta", "mean_explicit", "mean_atwill", "cov_delta_explicit", "cov_delta_atwill", "cov_explicit_atwill", "cov_delta_search", "cov_explicit_search", "cov_atwill_search", "mean_long", "var_long", "cov_long_delta", "cov_long_explicit", "cov_long_atwill", "cov_long_internal", "cov_long_search"];


	# dataMomentCol = [col[name] for name in momentName]; 
	# dataMoment = dataMoment_all[dataMomentCol];

	# # covariance of selected moments and the initial weighting matrix
	# CovDataMoment = CovDataMoment_all[dataMomentCol, dataMomentCol];
	# D, V = eig(CovDataMoment);
	# if minimum(D)<=0 
	#     error("Covariance matrix is not positive definite");
	# end
	# WeightMatrix = inv(CovDataMoment);

	# WeightMatrix = inv(CovDataMoment_all);

	# normalize WeightMatrix and adjust
	# WeightNorm = WeightMatrix/minimum(diag(WeightMatrix))
	# WeightNormDiagnal = diag(WeightNorm)
	# WeightNormDiagnal[9] = WeightNormDiagnal[9] * 10.0
	# WeightNormDiagnal[11] = WeightNormDiagnal[11] / 1000.

	# simplified_weight = WeightNormDiagnal/sum(WeightNormDiagnal)
	# WeightMatrix = diagm(WeightNormDiagnal)

	df_data = DataFrame(moment_names = momentName, moment_value = dataMoment, weight = diag(WeightMatrix))
	show(df_data, allcols=true)
    println()

 #    df_data = DataFrame(moment_names = momentName, moment_value = dataMoment, weight = diag(WeightMatrixDiag)./1000, importance = abs.(dataMoment) .* diag(WeightMatrixDiag)./1000)
	# show(df_data, allcols=true)
 #    println()

