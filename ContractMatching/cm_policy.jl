
function internalContractPolicyUpdate!(mat_Mu, mat_cStar, mat_dummiesStar, vec_Vi, vec_Ve; sol_uc = sol_uc_de, bounds = bounds_de, para = para_de, external = false)

    @unpack num_i, num_e = para

    for i_ind in 1:num_i, e_ind in 1:num_e

        vi_val = vec_Vi[i_ind] # obtain search value of i
        ve_val = vec_Ve[e_ind] # obtain search value of e 

        internalContract = contract(i_ind, e_ind, vi_val, ve_val, sol_uc = sol_uc, bounds = bounds, para = para, external = external)

        if internalContract.flag
            mat_Mu[i_ind, e_ind] = true
            mat_cStar[i_ind, e_ind] = internalContract.c_star
            mat_dummiesStar[i_ind, e_ind] = internalContract.dummies_star
        end
    end 

    return nothing
end



function externalContractPolicyUpdate!(arr_Mu, arr_cStar, arr_dummiesStar, mat_Mu, vec_Vi, mat_Ve; para = para_de, sol_uc = sol_uc_de, bounds = bounds_de)

    @unpack num_i, num_e = para

    for i_ind in 1:num_i, e_ind in 1:num_e, iota_ind in 1:num_i 

        # values of i and e in an internal matching
        internalMatchingValue_e = mat_Ve[e_ind, i_ind]

        # the possibly higher search value
        vi_val = vec_Vi[i_ind] # obtain search value of i
        ve_val = mat_Ve[e_ind, iota_ind] # obtain search value of e

        ieInternalMatch = mat_Mu[i_ind, e_ind]
        eiotaInternalMatch = mat_Mu[iota_ind, e_ind]

        # external match is possible only if i and e can form internal match, otherwise, you follow the initial values

        # external match is different from internal match only if e and iota can form internal match and the internal match value of (i,e) is lower

        if ieInternalMatch && eiotaInternalMatch && internalMatchingValue_e < ve_val                    

            externalContract = externalContractFunc(i_ind, e_ind, vi_val, ve_val, sol_uc = sol_uc, para = para, bounds = bounds)

            if externalContract.flag # if matching with higher mat_Ve, then update with the new contract
                arr_Mu[i_ind, e_ind, iota_ind] = true
                arr_cStar[i_ind, e_ind, iota_ind] = externalContract.c_star
                arr_dummiesStar[i_ind, e_ind, iota_ind] = externalContract.dummies_star

            # check if the following is needed
            else # matching is not possible with this higher ve_val, the pair should get their search value which is
                arr_Mu[i_ind, e_ind, iota_ind] = false
                arr_cStar[i_ind, e_ind, iota_ind] = -0.5
                arr_dummiesStar[i_ind, e_ind, iota_ind] = [0,0,0]
            end
        end
    end

    return nothing
end
