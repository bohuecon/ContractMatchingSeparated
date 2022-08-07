
"""
    (ci_star_uc = ci_star_uc, di_star_uc = di_star_uc, ce_star_uc = ce_star_uc, de_star_uc = de_star_uc, vec_ci = vec_ci, vec_ce = vec_ce, vec_πi = vec_πi, vec_πe = vec_πe) = optimal_c_uc(;para = para_de)

    Compute the unconstrained optimal contract for the investor and executive. One should be able to calculate optimal_c_uc for each parameter para. 
        
    ### in
        para: model parameters

    ### out is a named tuple
        global optimizations (without constraint)
        optimal c and dummies for i: ci_star_uc, di_star_uc
        optimal c and dummies for e: ce_star_uc, de_star_uc

        Optimization of c only (for a given value of dummies)
        given each case of dummies, the optimal c for i: vec_ci, vec_πi
        given each case of dummies, the optimal c for e: vec_ce, vec_πe

"""
function optimal_c_uc(;para = para_de, external = false)

    # read model parameters 

    @unpack vec_dummies, num_dcases, h, ρ = para
    β1, β2, vec_β, γ1, vec_γ = read_contract_para(para, external) 

    # ignore g(i,e) since optimal c's are independent of g
    πi_s(c, dummies) = πi(c, dummies, β1, β2, vec_β, γ1, vec_γ)
    πe_s(c, dummies) = πe(c, dummies, β1, β2, vec_β, γ1, vec_γ)

    # optimal c of i/e for each value of dummy combinations
    vec_ci = fill(0.0, num_dcases) 
    vec_ce = fill(0.0, num_dcases) 
    # optimal pi_i/e for each value of dummy combinations
    vec_πi = fill(0.0, num_dcases) 
    vec_πe = fill(0.0, num_dcases) 

    for dummies_ind in eachindex(vec_dummies)
        
        dummies = vec_dummies[dummies_ind]

        # optimization for i given a combination of dummy variables
        result_i = optimize(c -> - πi_s(c, dummies), 0.0, 1.0)
        vec_ci[dummies_ind] = result_i.minimizer
        vec_πi[dummies_ind] = - result_i.minimum

        # optimization for e
        result_e = optimize(c -> - πe_s(c, dummies), 0.0, 1.0)
        vec_ce[dummies_ind] = result_e.minimizer
        vec_πe[dummies_ind] = - result_e.minimum
    end

    # the optimal c and dummies for i
    πi_star, ci_star_ind = findmax(vec_πi)
    ci_star_uc = vec_ci[ci_star_ind]
    di_star_uc = vec_dummies[ci_star_ind]

    # the optimal c and dummies for e
    πe_star, ce_star_ind = findmax(vec_πe)
    ce_star_uc = vec_ce[ce_star_ind]
    de_star_uc = vec_dummies[ce_star_ind]

    return (ci_star_uc = ci_star_uc, di_star_uc = di_star_uc, ce_star_uc = ce_star_uc, de_star_uc = de_star_uc, πi_star = πi_star, πe_star = πe_star, vec_ci = vec_ci, vec_ce = vec_ce, vec_πi = vec_πi, vec_πe = vec_πe)

end

"""
    (c_star = c_star, dummies_star = dummies_star, πi_star = πi_star, πe_star = πe_star, flag = flag) = contract(ii, ei, vi_val, ve_val, sol_uc; para = para_de)

    Compute the unconstrained optimal contract for the investor and executive. One should be able to calculate optimal_c_uc for each parameter para. 
        
    ### in
        ii, ei: index of i and e
        vi_val, ve_val: search values of i and e 
        sol_uc: solution of unconstrained contracting problem 
        para: model parameters

    ### out is a named tuple
        c_star: optimal c (-1.0 if no matching)
        dummies_star: optimal dummies (vec(-1.0) if no matching)
        πi_star: profit of i at the optimal constract (0.0 if no matching)
        πe_star: profit of e at the optimal constract
        flag: 0 no matching, 1 matching

"""
function contract(ii, ei, vi_val, ve_val; sol_uc = sol_uc_de, bounds = bounds_de, para = para_de, external = false)

    @unpack vec_i, vec_e, πi, πe, ρ, vec_dummies, num_dcases, num_dummies, mat_g = para

    @unpack ci_star_uc, di_star_uc, ce_star_uc, de_star_uc, vec_ci, vec_ce = sol_uc

    β1, β2, vec_β, γ1, vec_γ = read_contract_para(para, external) 

    # set holders and set default value if no matching
    πi_star = 0.0; πe_star = 0.0; c_star = -0.5
    dummies_star = fill(-1, num_dummies)
    flag = false # false represents not match, true represents match

    # pre calculation
    i = vec_i[ii]; e = vec_e[ei]; g_val = mat_g[ii, ei]

    # πi according to ci_star_uc, di_star_uc
    πi_s(c, dummies) = πi(c, dummies, β1, β2, vec_β, γ1, vec_γ) * g_val
    πe_s(c, dummies) = πe(c, dummies, β1, β2, vec_β, γ1, vec_γ) * g_val

    πii_star_uc = πi_s(ci_star_uc, di_star_uc)
    πei_star_uc = πe_s(ci_star_uc, di_star_uc)
    πee_star_uc = πe_s(ce_star_uc, de_star_uc)

    if πii_star_uc > vi_val && πee_star_uc > ve_val
        # only so, a match is possible; if not, no match is formed, then there is no need to update value functions   
       
        if πei_star_uc >= ve_val 
            # the optimal value can be achieved
            flag = true
            πi_star = πii_star_uc
            πe_star = πei_star_uc
            c_star = ci_star_uc
            dummies_star = di_star_uc
       
        else 
            # πei_star_uc < ve_val, the maximized profits for i cannot be achieved, but still matching is possible

            # make containers of optimal πi, πe, c, for each combination of dummy variables
            vec_πi_star = fill(0.0, num_dcases)
            vec_πe_star = fill(0.0, num_dcases)
            vec_c_star = fill(0.0, num_dcases)
            vec_flag = fill(0, num_dcases)
            vi_val_adjusted = vi_val/g_val
            ve_val_adjusted = ve_val/g_val

            for dummies_ind in eachindex(vec_dummies)

                # for each dummies, obtain the constriained optimal c1_star, πi_star, πe_star, exist or not

                ci_lower = bounds.vec_ciLowerBounds[dummies_ind](vi_val_adjusted)
                ci_upper = bounds.vec_ciUpperBounds[dummies_ind](vi_val_adjusted)
                ce_lower = bounds.vec_ceLowerBounds[dummies_ind](ve_val_adjusted)
                ce_upper = bounds.vec_ceUpperBounds[dummies_ind](ve_val_adjusted)

                # for a given dummies, solve for optimal c
                # πi_s, πe_s are inserted, no need to have 'external'

                vec_c_star[dummies_ind], vec_πi_star[dummies_ind], vec_πe_star[dummies_ind], vec_flag[dummies_ind] = contract_c(dummies_ind, vi_val, ve_val, πi_s, πe_s, ci_lower, ci_upper, ce_lower, ce_upper, sol_uc, para)
            end

            if sum(vec_flag) > 0
                # println("get in here")
                flag = true # at least one value of dummies can lead to successful matching 
                πi_star, index_star = findmax(vec_πi_star)
                πe_star = vec_πe_star[index_star]
                c_star = vec_c_star[index_star]
                dummies_star = vec_dummies[index_star]
            end                  

        end

    end

    # return c_star, dummies_star, πi_star, πe_star, flag

    return (c_star = c_star, dummies_star = dummies_star, πi_star = πi_star, πe_star = πe_star, flag = flag)

end



function externalContractFunc(ii, ei, vi_val, ve_val; sol_uc = sol_uc_de, bounds = bounds_de, para = para_de)

    # make sure the assigned sol_uc, bounds are for external match

    @unpack vec_i, vec_e, πi, πe, ρ, vec_dummies, num_dcases, num_dummies, mat_g = para

    β1, β2, vec_β, γ1, vec_γ = para.β1_ex, para.β2_ex, para.vec_β_ex, para.γ1_ex, para.vec_γ_ex

    @unpack ci_star_uc, di_star_uc, ce_star_uc, de_star_uc, vec_ci, vec_ce = sol_uc

    # set holders and set default value if no matching
    πi_star = 0.0; πe_star = 0.0; c_star = -0.5
    dummies_star = fill(-1, num_dummies)
    flag = false # false means no match, true means match

    # pre calculation
    i = vec_i[ii]; e = vec_e[ei]; g_val = mat_g[ii, ei]
    
    πi_s(c, dummies) = πi(c, dummies, β1, β2, vec_β, γ1, vec_γ) * g_val
    πe_s(c, dummies) = πe(c, dummies, β1, β2, vec_β, γ1, vec_γ) * g_val

    # πii_star_uc = πi_s(ci_star_uc, di_star_uc)
    # πei_star_uc = πe_s(ci_star_uc, di_star_uc)
    πee_star_uc = πe_s(ce_star_uc, de_star_uc)

    if πee_star_uc > ve_val

        # for external contracts that need to be updated, we must have
        # πii_star_uc > vi_val 
        # πei_star_uc < ve_val 
        # thus, only if πee_star_uc > ve_val, matching is possible

        # make the container of optimal πi, πe, c1, for each value of c2

        vec_πi_star = fill(0.0, num_dcases)
        vec_πe_star = fill(0.0, num_dcases)
        vec_c_star = fill(0.0, num_dcases)
        vec_flag = fill(0, num_dcases)

        vi_val_adjusted = vi_val/g_val
        ve_val_adjusted = ve_val/g_val

        for dummies_ind in eachindex(vec_dummies)

            ci_lower = bounds.vec_ciLowerBounds[dummies_ind](vi_val_adjusted)
            ci_upper = bounds.vec_ciUpperBounds[dummies_ind](vi_val_adjusted)
            ce_lower = bounds.vec_ceLowerBounds[dummies_ind](ve_val_adjusted)
            ce_upper = bounds.vec_ceUpperBounds[dummies_ind](ve_val_adjusted)

            vec_c_star[dummies_ind], vec_πi_star[dummies_ind], vec_πe_star[dummies_ind], vec_flag[dummies_ind] = contract_c(dummies_ind, vi_val, ve_val, πi_s, πe_s, ci_lower, ci_upper, ce_lower, ce_upper, sol_uc, para)
        end


        if sum(vec_flag) > 0
            # println("get in here")
            flag = true # at least one value of dummies can lead to successful matching 
            πi_star, index_star = findmax(vec_πi_star)
            πe_star = vec_πe_star[index_star]
            c_star = vec_c_star[index_star]
            dummies_star = vec_dummies[index_star]
        end                  
    end

    # return c_star, dummies_star, πi_star, πe_star, flag

    return (c_star = c_star, dummies_star = dummies_star, πi_star = πi_star, πe_star = πe_star, flag = flag)

end


"""

    c_star, πi_star, πe_star, flag = contract_c(dummies_ind, vi_val, ve_val, πi_s, πe_s, sol_uc, para)


    Compute the constrained optimal contract of c for given a case of dummies
    
    ### in
    dummies_ind: the index of dummies in vec_dummies (in total num_dcases cases) 
    vi_val, ve_val: search values 
    πi_s, πe_s: short hand of profit functions
    sol_uc: solution of unconstrained contracting problem 
    para: model parameters

    ### out
    c_star: optimal c for given dummies
    πi_star, πe_star: the corresponding profits of i and e 
    flag: 0 no match, 1 match

"""
function contract_c(dummies_ind::Int, vi_val::Float64, ve_val::Float64, πi_s::Function, πe_s::Function, ci_lower::Float64, ci_upper::Float64, ce_lower::Float64, ce_upper::Float64, sol_uc::NamedTuple, para::NamedTuple)

    @unpack vec_dummies = para
    @unpack vec_ci, vec_ce = sol_uc

    dummies = vec_dummies[dummies_ind]
    πi_ss(c) = πi_s(c, dummies)
    πe_ss(c) = πe_s(c, dummies)

    c_lower = max(ci_lower, ce_lower)
    c_upper = min(ci_upper, ce_upper)

    if c_lower < c_upper
        flag = 1
        result = optimize(c -> - πi_ss(c), c_lower, c_upper)
        c_star = result.minimizer
        πi_star = -result.minimum
        πe_star = πe_ss(c_star)
    else
        flag = 0
        c_star = 0.0
        πi_star = vi_val
        πe_star = ve_val
    end

    return c_star, πi_star, πe_star, flag
end







