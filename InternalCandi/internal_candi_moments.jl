
function compute_moments(sol, para)

    @unpack vec_prob_i, vec_prob_e, mat_prob_ie, num_e, num_i, λi, γi, vec_i = para
    @unpack mat_Mu, arr_Mu, mat_cStar, arr_cStar, mat_dummiesStar, arr_dummiesStar = sol

    mat_D1 = [dummies[1] for dummies in mat_dummiesStar];
    mat_D2 = [dummies[2] for dummies in mat_dummiesStar];
    arr_D1 = [dummies[1] for dummies in arr_dummiesStar];
    arr_D2 = [dummies[2] for dummies in arr_dummiesStar];

    # Matching probabilities
    # internal matching probability for vec_i, one for each type i
    vec_pi = mat_Mu * vec_prob_e

    # external matching probability for vec_i, one for each type i
	one_layer_prob_ei = reshape(mat_prob_ie', 1, num_e, num_i) 
	arr_prob_ie =  repeat(one_layer_prob_ei, num_i, 1, 1)
	vec_pi_tilde = sum(sum(arr_Mu .* arr_prob_ie, dims = 3), dims = 2)[:, :, 1]

	# vec_omega_i_star equilibrium matching probability
	# total_matcings: the expected number of matches in unit time
	# vec_fi: the aggregate matching rate (internal plus external)
	# vec_matches: the expected number of matches made by each type i
	# vec_omega_i_star: the equilibrium matching weight of each type i

    vec_fi_in = γi * vec_pi  
    vec_fi_ex = λi * vec_pi_tilde
    vec_fi = vec_fi_in + vec_fi_ex      # ω_e x m(i,e) in the appendix with poisson rate weights

	total_matches =  λi * dot(vec_prob_i, vec_pi_tilde) + γi * dot(vec_prob_i, vec_pi)
	vec_matches = vec_prob_i .* vec_fi                # ω_i x ω_e x m(i,e) in the appendix
	vec_omega_i_star = vec_matches / total_matches    # \frac{ω_i x ω_e x m(i,e)}{∑ω_i x ω_e x m(i,e)}

	# SEARCH BEHAVIOR

    # overal search time (internal + external)
	vec_taui = 1 ./ vec_fi # expected search time for each type i

	E_tau = sum(vec_omega_i_star .* vec_taui) # 
    # dot(vec_matches, vec_taui)/sum(vec_matches)

	V_tau1 = sum(vec_omega_i_star .* (vec_taui.^2))
    V_tau2 = sum(vec_omega_i_star .* (vec_taui .- E_tau).^2)
    V_tau = V_tau1 + V_tau2

    # interal matching probability

    vec_inprob = γi * vec_pi ./ vec_fi
    E_inprob = dot(vec_omega_i_star, vec_inprob)
    V_inprob1 = sum(vec_omega_i_star .* (vec_inprob .* (1 .-vec_inprob)))
    V_inprob2 = sum(vec_omega_i_star .* (vec_inprob .- E_inprob).^2)
    V_inprob = V_inprob1 + V_inprob2

    # correlation between internal match and firm type
    # COV_inprob_i = dot(vec_omega_i_star, (vec_inprob .- E_inprob) .* (vec_i .- dot(vec_omega_i_star, vec_i)))
    # protracted probability
    # T = 0.4
    # E_protracted = dot(vec_omega_i_star, exp.(-vec_fi*T))

    # CONTRACT TERMS

    # c, expectation and variance
    vec_eWeightedcStarIn = ((mat_Mu .* mat_cStar) * vec_prob_e)
    vec_eWeightedcStarEx = sum(sum(((arr_Mu .* arr_cStar) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    E_cStar_numerator = γi * dot(vec_prob_i, vec_eWeightedcStarIn) + λi * dot(vec_prob_i, vec_eWeightedcStarEx)
    E_cStar = E_cStar_numerator / total_matches
    # E_cStarIn = dot(vec_prob_i, vec_eWeightedcStarIn) / dot(vec_prob_i, vec_pi)
    # E_cStarEx = dot(vec_prob_i, vec_eWeightedcStarEx) / dot(vec_prob_i, vec_pi_tilde)

    vec_eWeightedcStarDevIn = ((mat_Mu .* (mat_cStar .- E_cStar).^2.0) * vec_prob_e)
    vec_eWeightedcStarDevEx = sum(sum(((arr_Mu .* (arr_cStar .- E_cStar).^2.0) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    V_cStar_numerator = γi * dot(vec_prob_i, vec_eWeightedcStarDevIn) + λi * dot(vec_prob_i, vec_eWeightedcStarDevEx)
    V_cStar = V_cStar_numerator / total_matches
    # V_cStarIn = dot(vec_prob_i, vec_eWeightedcStarDevIn) / dot(vec_prob_i, vec_pi) 
    # V_cStarEx = dot(vec_prob_i, vec_eWeightedcStarDevEx) / dot(vec_prob_i, vec_pi) 

    # d1, expectation and variance
    vec_eWeightedD1In = ((mat_Mu .* mat_D1) * vec_prob_e)
    vec_eWeightedD1Ex = sum(sum(((arr_Mu .* arr_D1) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    E_D1_numerator = γi * dot(vec_prob_i, vec_eWeightedD1In) + λi * dot(vec_prob_i, vec_eWeightedD1Ex)
    E_D1 = E_D1_numerator / total_matches
    # E_D1In = dot(vec_prob_i, vec_eWeightedD1In) / dot(vec_prob_i, vec_pi)
    # E_D1Ex = dot(vec_prob_i, vec_eWeightedD1Ex) / dot(vec_prob_i, vec_pi_tilde)

    # vec_eWeightedD1DevIn = ((mat_Mu .* (mat_D1 .- E_D1).^2.0) * vec_prob_e)
    # vec_eWeightedD1DevEx = sum(sum(((arr_Mu .* (arr_D1 .- E_D1).^2.0) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    # V_D1_numerator = γi * dot(vec_prob_i, vec_eWeightedD1DevIn) + λi * dot(vec_prob_i, vec_eWeightedD1DevEx)
    # V_D1 = V_D1_numerator / total_matches
    # V_D1In = dot(vec_prob_i, vec_eWeightedD1DevIn) / dot(vec_prob_i, vec_pi)
    # V_D1Ex = dot(vec_prob_i, vec_eWeightedD1DevEx) / dot(vec_prob_i, vec_pi_tilde)

    # d2 expectation and variance
    vec_eWeightedD2In = ((mat_Mu .* mat_D2) * vec_prob_e)
    vec_eWeightedD2Ex = sum(sum(((arr_Mu .* arr_D2) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    E_D2_numerator = γi * dot(vec_prob_i, vec_eWeightedD2In) + λi * dot(vec_prob_i, vec_eWeightedD2Ex)
    E_D2 = E_D2_numerator / total_matches
    # E_D2In = dot(vec_prob_i, vec_eWeightedD2In) / dot(vec_prob_i, vec_pi)
    # E_D2Ex = dot(vec_prob_i, vec_eWeightedD2Ex) / dot(vec_prob_i, vec_pi_tilde)

    # vec_eWeightedD2DevIn = ((mat_Mu .* (mat_D2 .- E_D2).^2.0) * vec_prob_e)
    # vec_eWeightedD2DevEx = sum(sum(((arr_Mu .* (arr_D2 .- E_D2).^2.0) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    # V_D2_numerator = γi * dot(vec_prob_i, vec_eWeightedD2DevIn) + λi * dot(vec_prob_i, vec_eWeightedD2DevEx)
    # V_D2 = V_D2_numerator / total_matches
    # V_D2In = dot(vec_prob_i, vec_eWeightedD2DevIn) / dot(vec_prob_i, vec_pi)
    # V_D2Ex = dot(vec_prob_i, vec_eWeightedD2DevEx) / dot(vec_prob_i, vec_pi_tilde)

    # covariance among contract terms

    vec_eWeightedcStarD1DevIn = ((mat_Mu .* (mat_cStar .- E_cStar) .* (mat_D1 .- E_D1)) * vec_prob_e)
    vec_eWeightedcStarD1DevEx = sum(sum(((arr_Mu .* (arr_cStar .- E_cStar) .* (arr_D1 .- E_D1)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_cStarD1_numerator = γi * dot(vec_prob_i, vec_eWeightedcStarD1DevIn) + λi * dot(vec_prob_i, vec_eWeightedcStarD1DevEx)
    Cov_cStar_D1 = Cov_cStarD1_numerator / total_matches
    # Cov_cStar_D1In = dot(vec_prob_i, vec_eWeightedcStarD1DevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_cStar_D1Ex = dot(vec_prob_i, vec_eWeightedcStarD1DevEx) / dot(vec_prob_i, vec_pi_tilde)

    vec_eWeightedcStarD2DevIn = ((mat_Mu .* (mat_cStar .- E_cStar) .* (mat_D2 .- E_D2)) * vec_prob_e)
    vec_eWeightedcStarD2DevEx = sum(sum(((arr_Mu .* (arr_cStar .- E_cStar) .* (arr_D2 .- E_D2)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_cStarD2_numerator = γi * dot(vec_prob_i, vec_eWeightedcStarD2DevIn) + λi * dot(vec_prob_i, vec_eWeightedcStarD2DevEx)
    Cov_cStar_D2 = Cov_cStarD2_numerator / total_matches
    # Cov_cStar_D2In = dot(vec_prob_i, vec_eWeightedcStarD2DevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_cStar_D2Ex = dot(vec_prob_i, vec_eWeightedcStarD2DevEx) / dot(vec_prob_i, vec_pi_tilde) 

    vec_eWeightedD1D2DevIn = ((mat_Mu .* (mat_D1 .- E_D1) .* (mat_D2 .- E_D2)) * vec_prob_e)
    vec_eWeightedD1D2DevEx = sum(sum(((arr_Mu .* (arr_D1 .- E_D1) .* (arr_D2 .- E_D2)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_D1D2_numerator = γi * dot(vec_prob_i, vec_eWeightedD1D2DevIn) + λi * dot(vec_prob_i, vec_eWeightedD1D2DevEx)
    Cov_D1_D2 = Cov_D1D2_numerator / total_matches
    # Cov_D1_D2In = dot(vec_prob_i, vec_eWeightedD1D2DevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_D1_D2Ex = dot(vec_prob_i, vec_eWeightedD1D2DevEx) / dot(vec_prob_i, vec_pi_tilde)

    # covariance between search time and contract terms

    vec_eWeightedtaucStarDevIn = (vec_taui .- E_tau) .* ( mat_Mu .* (mat_cStar .- E_cStar) * vec_prob_e)
    vec_eWeightedtaucStarDevEx = (vec_taui .- E_tau) .* sum(sum(((arr_Mu .* (arr_cStar .- E_cStar)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_taucStar_numerator = γi * dot(vec_prob_i, vec_eWeightedtaucStarDevIn) + λi * dot(vec_prob_i, vec_eWeightedtaucStarDevEx)
    Cov_tau_cStar = Cov_taucStar_numerator / total_matches
    # Cov_tau_cStarIn = dot(vec_prob_i, vec_eWeightedtaucStarDevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_tau_cStarEx = dot(vec_prob_i, vec_eWeightedtaucStarDevEx) / dot(vec_prob_i, vec_pi_tilde)

    vec_eWeightedtauD1DevIn = (vec_taui .- E_tau) .* ( mat_Mu .* (mat_D1 .- E_D1) * vec_prob_e)
    vec_eWeightedtauD1DevEx = (vec_taui .- E_tau) .* sum(sum(((arr_Mu .* (arr_D1 .- E_D1)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_tauD1_numerator = γi * dot(vec_prob_i, vec_eWeightedtauD1DevIn) + λi * dot(vec_prob_i, vec_eWeightedtauD1DevEx)
    Cov_tau_D1 = Cov_tauD1_numerator / total_matches
    # Cov_tau_D1In = dot(vec_prob_i, vec_eWeightedtauD1DevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_tau_D1Ex = dot(vec_prob_i, vec_eWeightedtauD1DevEx) / dot(vec_prob_i, vec_pi_tilde)

    vec_eWeightedtauD2DevIn = (vec_taui .- E_tau) .* ( mat_Mu .* (mat_D2 .- E_D2) * vec_prob_e)
    vec_eWeightedtauD2DevEx = (vec_taui .- E_tau) .* sum(sum(((arr_Mu .* (arr_D2 .- E_D2)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_tauD2_numerator = γi * dot(vec_prob_i, vec_eWeightedtauD2DevIn) + λi * dot(vec_prob_i, vec_eWeightedtauD2DevEx)
    Cov_tau_D2 = Cov_tauD2_numerator / total_matches
    # Cov_tau_D2In = dot(vec_prob_i, vec_eWeightedtauD2DevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_tau_D2Ex = dot(vec_prob_i, vec_eWeightedtauD2DevEx) / dot(vec_prob_i, vec_pi_tilde)

    # OUTCOMES

    @unpack ρ, β1, β2, vec_β, vec_i, vec_e, κ0, κ1 = para;

    # expected matching values of all success matches
    mat_mv = fill(0.0, num_i, num_e);
    arr_mv = fill(0.0, num_i, num_e, num_i);

    mv(i, e, c, dummies) = g(i, e, ρ) * h(c, dummies, β1, β2, vec_β)

    for ii in 1:num_i, ei in 1:num_e
        i = vec_i[ii]
        e = vec_e[ei]
        c = mat_cStar[ii, ei]
        dummies = mat_dummiesStar[ii, ei]
        mat_mv[ii, ei] = mv(i,e,c,dummies)
    end


    for ii in 1:num_i, ei in 1:num_e, iotai in 1:num_i
        i = vec_i[ii]
        e = vec_e[ei]
        c = arr_cStar[ii, ei, iotai]
        dummies = arr_dummiesStar[ii, ei, iotai]
        arr_mv[ii, ei, iotai] = mv(i,e,c,dummies)
    end

    vec_eWeightedmvIn = ((mat_Mu .* mat_mv) * vec_prob_e)
    vec_eWeightedmvEx = sum(sum(((arr_Mu .* arr_mv) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    E_mv_numerator = γi * dot(vec_prob_i, vec_eWeightedmvIn) + λi * dot(vec_prob_i, vec_eWeightedmvEx)
    E_mv = E_mv_numerator / total_matches
    # E_mvIn = dot(vec_prob_i, vec_eWeightedmvIn) / dot(vec_prob_i, vec_pi)
    # E_mvEx = dot(vec_prob_i, vec_eWeightedmvEx) / dot(vec_prob_i, vec_pi_tilde)

    # success rate expectation and variance

    Φ = Normal(0.0, 1.0)
    mat_successProb = cdf.(Φ, κ0 .+ κ1 * mat_mv)
    arr_successProb = cdf.(Φ, κ0 .+ κ1 * arr_mv)

    vec_eWeightedsuccessProbIn = ((mat_Mu .* mat_successProb) * vec_prob_e)
    vec_eWeightedsuccessProbEx = sum(sum(((arr_Mu .* arr_successProb) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    E_successProb_numerator = γi * dot(vec_prob_i, vec_eWeightedsuccessProbIn) + λi * dot(vec_prob_i, vec_eWeightedsuccessProbEx)
    E_successProb = E_successProb_numerator / total_matches
    # E_successProbIn = dot(vec_prob_i, vec_eWeightedsuccessProbIn) / dot(vec_prob_i, vec_pi)
    # E_successProbEx = dot(vec_prob_i, vec_eWeightedsuccessProbEx) / dot(vec_prob_i, vec_pi_tilde)

    # success rate variance

    vec_eWeightedsuccessProbDevIn = (mat_Mu .* (mat_successProb .* (1.0 .- mat_successProb) + (mat_successProb .- E_successProb).^2.0)) * vec_prob_e
    vec_eWeightedsuccessProbDevEx = sum(sum((arr_Mu .* (arr_successProb .* (1.0 .- arr_successProb) + (arr_successProb .- E_successProb).^2.0)) .* arr_prob_ie, dims = 3), dims = 2)[:, :, 1]
    V_successProb_numerator = γi * dot(vec_prob_i, vec_eWeightedsuccessProbDevIn) + λi * dot(vec_prob_i, vec_eWeightedsuccessProbDevEx)
    V_successProb = V_successProb_numerator / total_matches
    # V_successProbIn = dot(vec_prob_i, vec_eWeightedsuccessProbDevIn) / dot(vec_prob_i, vec_pi) 
    # V_successProbEx = dot(vec_prob_i, vec_eWeightedsuccessProbDevEx) / dot(vec_prob_i, vec_pi) 

    # success rate and contract terms covariance

    vec_eWeightedsuccessProbcStarDevIn = ((mat_Mu .* (mat_successProb .- E_successProb) .* (mat_cStar .- E_cStar)) * vec_prob_e)
    vec_eWeightedsuccessProbcStarDevEx = sum(sum(((arr_Mu .* (arr_successProb .- E_successProb) .* (arr_cStar .- E_cStar)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_successProbcStar_numerator = γi * dot(vec_prob_i, vec_eWeightedsuccessProbcStarDevIn) + λi * dot(vec_prob_i, vec_eWeightedsuccessProbcStarDevEx)
    Cov_successProb_cStar = Cov_successProbcStar_numerator / total_matches
    # Cov_successProb_cStarIn = dot(vec_prob_i, vec_eWeightedsuccessProbcStarDevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_successProb_cStarEx = dot(vec_prob_i, vec_eWeightedsuccessProbcStarDevEx) / dot(vec_prob_i, vec_pi_tilde)

    vec_eWeightedsuccessProbD1DevIn = ((mat_Mu .* (mat_successProb .- E_successProb) .* (mat_D1 .- E_D1)) * vec_prob_e)
    vec_eWeightedsuccessProbD1DevEx = sum(sum(((arr_Mu .* (arr_successProb .- E_successProb) .* (arr_D1 .- E_D1)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_successProbD1_numerator = γi * dot(vec_prob_i, vec_eWeightedsuccessProbD1DevIn) + λi * dot(vec_prob_i, vec_eWeightedsuccessProbD1DevEx)
    Cov_successProb_D1 = Cov_successProbD1_numerator / total_matches
    # Cov_successProb_D1In = dot(vec_prob_i, vec_eWeightedsuccessProbD1DevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_successProb_D1Ex = dot(vec_prob_i, vec_eWeightedsuccessProbD1DevEx) / dot(vec_prob_i, vec_pi_tilde)

    vec_eWeightedsuccessProbD2DevIn = ((mat_Mu .* (mat_successProb .- E_successProb) .* (mat_D2 .- E_D2)) * vec_prob_e)
    vec_eWeightedsuccessProbD2DevEx = sum(sum(((arr_Mu .* (arr_successProb .- E_successProb) .* (arr_D2 .- E_D2)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_successProbD2_numerator = γi * dot(vec_prob_i, vec_eWeightedsuccessProbD2DevIn) + λi * dot(vec_prob_i, vec_eWeightedsuccessProbD2DevEx)
    Cov_successProb_D2 = Cov_successProbD2_numerator / total_matches
    # Cov_successProb_D2In = dot(vec_prob_i, vec_eWeightedsuccessProbD2DevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_successProb_D2Ex = dot(vec_prob_i, vec_eWeightedsuccessProbD2DevEx) / dot(vec_prob_i, vec_pi_tilde)

    # success rate and search time

    vec_eWeightedtausuccessProbDevIn = (vec_taui .- E_tau) .* ( mat_Mu .* (mat_successProb .- E_successProb) * vec_prob_e)
    vec_eWeightedtausuccessProbDevEx = (vec_taui .- E_tau) .* sum(sum(((arr_Mu .* (arr_successProb .- E_successProb)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_tausuccessProb_numerator = γi * dot(vec_prob_i, vec_eWeightedtausuccessProbDevIn) + λi * dot(vec_prob_i, vec_eWeightedtausuccessProbDevEx)
    Cov_tau_successProb = Cov_tausuccessProb_numerator / total_matches
    # Cov_tau_successProbIn = dot(vec_prob_i, vec_eWeightedtausuccessProbDevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_tau_successProbEx = dot(vec_prob_i, vec_eWeightedtausuccessProbDevEx) / dot(vec_prob_i, vec_pi_tilde)


    vec_eWeightedinprob_successProbDevIn = (vec_inprob .- E_inprob) .* ( mat_Mu .* (mat_successProb .- E_successProb) * vec_prob_e)
    vec_eWeightedinprob_successProbDevEx = (vec_inprob .- E_inprob) .* sum(sum(((arr_Mu .* (arr_successProb .- E_successProb)) .* arr_prob_ie), dims = 3), dims = 2)[:, :, 1]
    Cov_inprob_successProb_numerator = γi * dot(vec_prob_i, vec_eWeightedinprob_successProbDevIn) + λi * dot(vec_prob_i, vec_eWeightedinprob_successProbDevEx)
    Cov_inprob_successProb = Cov_inprob_successProb_numerator / total_matches
    # Cov_inprob_successProbIn = dot(vec_prob_i, vec_eWeightedinprob_successProbDevIn) / dot(vec_prob_i, vec_pi) 
    # Cov_inprob_successProbEx = dot(vec_prob_i, vec_eWeightedinprob_successProbDevEx) / dot(vec_prob_i, vec_pi_tilde)


    # return (E_tau = E_tau, V_tau = V_tau, 
    # E_inprob = E_inprob, V_inprob = V_inprob, 
    # E_cStar = E_cStar, E_cStarIn = E_cStarIn, E_cStarEx = E_cStarEx, 
    # V_cStar = V_cStar, V_cStarIn = V_cStarIn, V_cStarEx = V_cStarEx, 
    # E_D1 = E_D1, E_D1In = E_D1In, E_D1Ex = E_D1Ex, 
    # E_D2 = E_D2, E_D2In = E_D2In, E_D2Ex = E_D2Ex, 
    # Cov_cStar_D1 = Cov_cStar_D1, 
    # Cov_cStar_D1In = Cov_cStar_D1In, 
    # Cov_cStar_D1Ex = Cov_cStar_D1Ex, 
    # Cov_cStar_D2 = Cov_cStar_D2, 
    # Cov_cStar_D2In = Cov_cStar_D2In, 
    # Cov_cStar_D2Ex = Cov_cStar_D2Ex, 
    # Cov_D1_D2 = Cov_D1_D2, 
    # Cov_D1_D2In = Cov_D1_D2In, 
    # Cov_D1_D2Ex = Cov_D1_D2Ex, 
    # Cov_tau_cStar = Cov_tau_cStar,
    # Cov_tau_cStarIn = Cov_tau_cStarIn,
    # Cov_tau_cStarEx = Cov_tau_cStarEx,
    # Cov_tau_D1 = Cov_tau_D1,
    # Cov_tau_D1In = Cov_tau_D1In,
    # Cov_tau_D1Ex = Cov_tau_D1Ex,
    # Cov_tau_D2 = Cov_tau_D2,
    # Cov_tau_D2In = Cov_tau_D2In,
    # Cov_tau_D2Ex = Cov_tau_D2Ex,
    # E_successProb = E_successProb, 
    # E_successProbIn = E_successProbIn, 
    # E_successProbEx = E_successProbEx, 
    # V_successProb = V_successProb,
    # V_successProbIn = V_successProbIn,
    # V_successProbEx = V_successProbEx,
    # Cov_successProb_cStar = Cov_successProb_cStar,
    # Cov_successProb_cStarIn = Cov_successProb_cStarIn,
    # Cov_successProb_cStarEx = Cov_successProb_cStarEx,
    # Cov_successProb_D1 = Cov_successProb_D1,
    # Cov_successProb_D1In = Cov_successProb_D1In,
    # Cov_successProb_D1Ex = Cov_successProb_D1Ex,
    # Cov_successProb_D2 = Cov_successProb_D2,
    # Cov_successProb_D2In = Cov_successProb_D2In,
    # Cov_successProb_D2Ex = Cov_successProb_D2Ex,
    # Cov_tau_successProb = Cov_tau_successProb,
    # Cov_tau_successProbIn = Cov_tau_successProbIn,
    # Cov_tau_successProbEx = Cov_tau_successProbEx
    # )

    # return (E_tau = E_tau, V_tau = V_tau, 
    # E_inprob = E_inprob, V_inprob = V_inprob, 
    # E_cStar = E_cStar, 
    # V_cStar = V_cStar, 
    # E_D1 = E_D1, 
    # E_D2 = E_D2, 
    # Cov_cStar_D1 = Cov_cStar_D1, 
    # Cov_cStar_D2 = Cov_cStar_D2, 
    # Cov_D1_D2 = Cov_D1_D2, 
    # Cov_tau_cStar = Cov_tau_cStar,
    # Cov_tau_D1 = Cov_tau_D1,
    # Cov_tau_D2 = Cov_tau_D2,
    # E_successProb = E_successProb, 
    # V_successProb = V_successProb,
    # Cov_successProb_cStar = Cov_successProb_cStar,
    # Cov_successProb_D1 = Cov_successProb_D1,
    # Cov_successProb_D2 = Cov_successProb_D2,
    # Cov_inprob_successProb = Cov_inprob_successProb,
    # Cov_tau_successProb = Cov_tau_successProb
    # )

    return [E_tau, V_tau, E_inprob, V_inprob, E_cStar, V_cStar, E_D1, E_D2, Cov_cStar_D1, Cov_cStar_D2, Cov_D1_D2, Cov_tau_cStar, Cov_tau_D1, Cov_tau_D2, E_successProb, Cov_successProb_cStar, Cov_successProb_D1, Cov_successProb_D2, Cov_inprob_successProb, Cov_tau_successProb]
end

