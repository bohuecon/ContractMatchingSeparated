

function internalContractValueUpdate!(mat_Πi, mat_Πe, mat_Mu, vec_Vi, vec_Ve; sol_uc = sol_uc_de, bounds = bounds_de, para = para_de, external = false)

    @unpack num_i, num_e = para

    for i_ind in 1:num_i, e_ind in 1:num_e

        vi_val = vec_Vi[i_ind] # obtain search value of i
        ve_val = vec_Ve[e_ind] # obtain search value of e 

        internalContract = contract(i_ind, e_ind, vi_val, ve_val, sol_uc = sol_uc, bounds = bounds, para = para, external = external)

        if internalContract.flag
            mat_Πi[i_ind, e_ind] = internalContract.πi_star
            mat_Πe[i_ind, e_ind] = internalContract.πe_star
            mat_Mu[i_ind, e_ind] = internalContract.flag
        end
    end 

    return nothing
end

function externalContractValueUpdate!(arr_Πi, arr_Πe, mat_Mu, vec_Vi, mat_Ve; para = para_de, sol_uc = sol_uc_de, bounds = bounds_de)

    @unpack num_i, num_e = para

    for i_ind in 1:num_i, e_ind in 1:num_e, iota_ind in 1:num_i 

        # values of i and e in an internal matching
        internalMatchingValue_e = arr_Πe[i_ind, e_ind, iota_ind]

        # internal match or not
        ieInternalMatch = mat_Mu[i_ind, e_ind]
        eiotaInternalMatch = mat_Mu[iota_ind, e_ind]

        # search values
        vi_val = vec_Vi[i_ind] # obtain search value of i
        ve_val = mat_Ve[e_ind, iota_ind] # obtain search value of e

        # mat_Mu[iota_ind, e_ind] == true then e and iota can form a match, we may need to solve the constract again
        # internalMatchingValue_e >= ve_val means the higher search value of e is not a problem, no need to update

        # if mat_Mu[i_ind, e_ind] && mat_Mu[iota_ind, e_ind] && internalMatchingValue_e < 

        if !ieInternalMatch
            # external match has a no smaller outside value, if internal match is not possible, external match neither         
            # arr_Πi[i_ind, e_ind, iota_ind] = vi_val
            arr_Πe[i_ind, e_ind, iota_ind] = ve_val

        elseif ieInternalMatch && eiotaInternalMatch && internalMatchingValue_e < ve_val          
            # need to compute the contract if (i,e) can form an internal match, (e, iota) can form an internal match, and (i,e) internal match value for e is lower than e's outside value

            externalContract = externalContractFunc(i_ind, e_ind, vi_val, ve_val, sol_uc = sol_uc, bounds = bounds, para = para)

            if externalContract.flag 
            # successful match with higher ve_val, now update the contract
                arr_Πi[i_ind, e_ind, iota_ind] = externalContract.πi_star
                arr_Πe[i_ind, e_ind, iota_ind] = externalContract.πe_star
            else 
            # matching fails with higher ve_val, the pair should get their search value which is
                arr_Πi[i_ind, e_ind, iota_ind] = vi_val
                arr_Πe[i_ind, e_ind, iota_ind] = ve_val
            end
        end
    end

    return nothing
end

