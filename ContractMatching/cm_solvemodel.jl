

#------------------ solve the model ------------------#

function solve_main(; para = para_de, diagnosis = false, save_results = false)

    # function solve_main(; para = para_de, save_results = false)

    @unpack num_i, num_e = para

    sol_uc_in = optimal_c_uc(para = para, external = false)
    sol_uc_ex = optimal_c_uc(para = para, external = true)
    bounds_in = cBoundsFunc(para = para, sol_uc = sol_uc_in, external = false)
    bounds_ex = cBoundsFunc(para = para, sol_uc = sol_uc_ex, external = true)

    # initial values
    ini_vec_Vi = zeros(num_i)
    ini_vec_Ve = zeros(num_e)
    ini_mat_Ve = zeros(num_e, num_i)
    # ini_mat_Ve = repeat(ini_vec_Ve, 1, num_i)

    sol, error_flag = solve_model(ini_vec_Vi, ini_vec_Ve, ini_mat_Ve, bounds_in = bounds_in, bounds_ex = bounds_ex, sol_uc_in = sol_uc_in, sol_uc_ex = sol_uc_ex, para = para, max_iter = 5000, diagnosis = diagnosis)

    if save_results
        save("cm.jld", "vec_Vi", sol.vec_Vi, "vec_Ve", sol.vec_Ve, "mat_Ve", sol.mat_Ve, "mat_Mu", sol.mat_Mu, "mat_cStar", sol.mat_cStar, "mat_dummiesStar", sol.mat_dummiesStar, "arr_Mu", sol.arr_Mu, "arr_cStar", sol.arr_cStar, "arr_dummiesStar", sol.arr_dummiesStar, "mat_Mu_bm", sol.mat_Mu_bm, "mat_cStar_bm", sol.mat_cStar_bm, "mat_dummiesStar_bm", sol.mat_dummiesStar_bm)
    end

    return sol, error_flag
end

function solve_model(vec_Vi, vec_Ve, mat_Ve; bounds_in = bounds_de, bounds_ex = bounds_de, sol_uc_in = sol_uc_de, sol_uc_ex = sol_uc_de, para = para_de, err_tol = 1e-4, diff_err_tol = 1e-10, max_iter = 5000, diagnosis = false)

    @unpack r, λi, λe, γi, γe, num_i, num_e, vec_prob_i, vec_prob_e, mat_prob_ie, num_dcases = para

    # initialize pair-value arrays
    mat_Πi = Array{Float64}(undef, num_i, num_e)
    mat_Πe = Array{Float64}(undef, num_i, num_e)    
    mat_Πi_bm = Array{Float64}(undef, num_i, num_e)
    mat_Πe_bm = Array{Float64}(undef, num_i, num_e)
    arr_Πi = Array{Float64}(undef, num_i, num_e, num_i)
    arr_Πe = Array{Float64}(undef, num_i, num_e, num_i) 
    
    # initialize type-value arrays
    vec_TVe = copy(vec_Ve)
    mat_TVe = copy(mat_Ve)
    vec_TVi = copy(vec_Vi)

    # initialize iterations
    print_skip = 50
    iterate_count = 0
    err = err_tol + 1.0
    diff_err = diff_err_tol + 1.0

    # while (iterate_count < max_iter) && (err > err_tol)

    while (iterate_count < max_iter) && (err > err_tol) && (diff_err > diff_err_tol)

        ######################
        # INTERNAL CONTRACTS #
        ######################

        # the contract between firm and internal candidates given both have outside search values and given β and γ. The firm's search value comes from vec_Vi, candidate's search value comes from vec_Ve. The output is updated mat_Πi and mat_Πe, both refer to the value of the pair (i, e) where e has no current firm options and i is the current firm 

        # mat_Πi default is to have Vi. Vi is a vector, each is current search value of i. Update if (i, e) can form a match (e' value exclude internal search)
        mat_Πi .= vec_Vi

        # mat_Πe default is to have Ve. Ve is a vector, each is current search value of e. Update if (i, e) can form a match (e' value exclude internal search)
        mat_Πe .= vec_Ve'

        # mat_Mu default to all pairs fail to form a match
        mat_Mu = falses(num_i, num_e)

        # update mat_Πi, mat_Πe by computing c(i, e)
        internalContractValueUpdate!(mat_Πi, mat_Πe, mat_Mu, vec_Vi, vec_Ve, sol_uc = sol_uc_in, bounds = bounds_in, para = para, external = false)

        ######################
        # EXTERNAL CONTRACTS #
        ######################

        # the contract between firm and external candidates (e, \iota). The firm's search value comes from vec_Vi, candidate's search value comes from mat_Ve. The outputs are updated arr_Πi, arr_Πe

        # First, use the external meeting β and γ to calculate if outside value is internal, we can this bm (benchmark of external contracts)

        mat_Πi_bm .= vec_Vi
        mat_Πe_bm .= vec_Ve'
        mat_Mu_bm = falses(num_i, num_e)

        internalContractValueUpdate!(mat_Πi_bm, mat_Πe_bm, mat_Mu_bm, vec_Vi, vec_Ve, sol_uc = sol_uc_ex, bounds = bounds_ex, para = para, external = true) # here external = true gurantees contract() will use the correct β and γ

        # Second, with bm, we calculate external contract.
        # arr_Πi default is to have mat_Πi or vec_Vi. Update if (e, iota) can form a match, so e's internal search has a value (according to benchmark). Then e's reservation value is higher, optimal contract may change
        arr_Πi = repeat(mat_Πi_bm, 1, 1, num_i)

        # arr_Πe default is to have mat_Πe or mat_Ve. Update if (e, iota) can form a match, so e's internal search has a value. Then e's reservation value is higher, optimall contract may change

        arr_Πe = repeat(mat_Πe_bm, 1, 1, num_i)
        
        # update arr_Πi and arr_Πe by computing c(i,e,\iota)
        # search value for (e, \iota) comes from mat_Ve
        externalContractValueUpdate!(arr_Πi, arr_Πe, mat_Mu_bm, vec_Vi, mat_Ve, sol_uc = sol_uc_ex, bounds = bounds_ex, para = para)


        #########################
        # ONE-STEP AHEAD UPDATE #
        #########################

        # e‘s search value of only external search 
        vec_TVe = (λe) / (r + λe) * (transpose(mat_Πe_bm) * vec_prob_i)

        # i's search value  
        # construct a box of probabilities 
        one_layer_prob_ei = reshape(mat_prob_ie', 1, num_e, num_i) 
        arr_prob_ie =  repeat(one_layer_prob_ei, num_i, 1, 1)
        vec_TVi_ex = sum(sum(arr_Πi .* arr_prob_ie, dims = 3), dims = 2)
        vec_TVi = (λi) / (r + λi + γi) * vec_TVi_ex  + (γi) / (r + λi + γi) * (mat_Πi * vec_prob_e)
        vec_TVi = vec_TVi[:, :, 1]

        # e_tilde's search value (including internal search)
        mat_TVe_ex = sum(arr_Πe .* repeat(vec_prob_i, 1, num_e, num_i), dims = 1)[1, :, :]
        # size(mat_TVe_ex) # num_e x num_i
        # internal meeting is equivalent to e meet iota and e has no internal search opportunities
        mat_TVe_in = transpose(mat_Πe)
        mat_TVe = (λe) / (r + λe + γe) * mat_TVe_ex  + (γe) / (r + λe + γe) * mat_TVe_in

        #######################
        # ITERATION PROCEDURE #
        #######################

        iterate_count += 1
        V = vcat(vec_Ve, reshape(mat_Ve, :, 1), vec_Vi)
        TV = vcat(vec_TVe, reshape(mat_TVe, :, 1), vec_TVi)

        old_err = err # save the previous err to old_err
        err = Base.maximum(abs, TV - V) # calculate the new err as err
        diff_err = abs(err - old_err) # calculate the difference

        if (iterate_count % print_skip == 0) && diagnosis
            println("       >>> compute iterate $iterate_count with error $err ...")
        end

        vec_TVi, vec_Vi = vec_Vi, vec_TVi
        vec_TVe, vec_Ve = vec_Ve, vec_TVe
        mat_TVe, mat_Ve = mat_Ve, mat_TVe

    end

    # The loop yields vec_Vi, vec_Ve, mat_Ve

    ####################
    # OPTIMAL CONTRACT #
    ####################

    # internal contract policy update

    # set default values, suppose no matching as initial values
    mat_Mu = falses(num_i, num_e)
    mat_cStar = fill(-0.5, num_i, num_e)
    mat_dummiesStar = fill([0,0,0], num_i, num_e)

    internalContractPolicyUpdate!(mat_Mu, mat_cStar, mat_dummiesStar, vec_Vi, vec_Ve, para = para, sol_uc = sol_uc_in, bounds = bounds_in, external = false)

    # external contract policy update

    mat_Mu_bm = falses(num_i, num_e)
    mat_cStar_bm = fill(-0.5, num_i, num_e)
    mat_dummiesStar_bm = fill([0,0,0], num_i, num_e)

    internalContractPolicyUpdate!(mat_Mu_bm, mat_cStar_bm, mat_dummiesStar_bm, vec_Vi, vec_Ve, para = para, sol_uc = sol_uc_ex, bounds = bounds_ex, external = true)

    arr_Mu = repeat(mat_Mu_bm, 1, 1, num_i)
    arr_cStar = repeat(mat_cStar_bm, 1, 1, num_i)
    arr_dummiesStar = repeat(mat_dummiesStar_bm, 1, 1, num_i)

    externalContractPolicyUpdate!(arr_Mu, arr_cStar, arr_dummiesStar, mat_Mu_bm, vec_Vi, mat_Ve, para = para, sol_uc = sol_uc_ex, bounds = bounds_ex)

    ######################
    # RECORD CONVERGENCE #
    ######################

    not_convergent = false

    if diff_err < diff_err_tol
        not_convergent = true
    end 


    return (vec_Vi = vec_Vi, vec_Ve = vec_Ve, mat_Ve = mat_Ve, mat_Mu = mat_Mu, mat_cStar = mat_cStar, mat_dummiesStar = mat_dummiesStar, arr_Mu = arr_Mu, arr_cStar = arr_cStar, arr_dummiesStar = arr_dummiesStar, mat_Mu_bm = mat_Mu_bm,mat_cStar_bm = mat_cStar_bm, mat_dummiesStar_bm = mat_dummiesStar_bm), not_convergent
end

