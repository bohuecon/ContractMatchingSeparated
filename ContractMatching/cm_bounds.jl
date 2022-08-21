
function cUpperBoundGen(π_ss, cGrids)

    πGrids = π_ss.(cGrids)
    πMin = minimum(πGrids)
    πMax = maximum(πGrids)

    πGrids_sorted = sort(πGrids)
    cGrids_sorted = sort(cGrids, rev=true)

    cBoundLi = LinearInterpolation(πGrids_sorted, cGrids_sorted)

    function cUpperBound(v)

        v_bounded = min(max(v, πMin), πMax)

        return cBoundLi(v_bounded)

    end 

    return cUpperBound

end

function cLowerBoundGen(π_ss, cGrids)

    πGrids = π_ss.(cGrids)
    πMin = minimum(πGrids)
    πMax = maximum(πGrids)

    cBoundLi = LinearInterpolation(πGrids, cGrids)

    function cLowerBound(v)

        v_bounded = min(max(v, πMin), πMax)

        return cBoundLi(v_bounded)

    end 

    return cLowerBound
end


function cBoundsFunc(; para = para_de, sol_uc = sol_uc_de, grid_length = 10000, external = false)

    @unpack vec_dummies, num_dcases, h, ρ = para
    β1, β2, vec_β, γ1, vec_γ = read_contract_para(para, external)
    @unpack vec_ci, vec_ce = sol_uc

    vec_ciLowerBounds = Array{Function}(undef, num_dcases)
    vec_ciUpperBounds = Array{Function}(undef, num_dcases)
    vec_ceLowerBounds = Array{Function}(undef, num_dcases)
    vec_ceUpperBounds = Array{Function}(undef, num_dcases)

    for di in 1:num_dcases

        dummies = vec_dummies[di]
        ci_uc = vec_ci[di] # ci_star corresponds to dummies
        ce_uc = vec_ce[di] # ce_star corresponds to dummies
        πi_ss(c) = πi(c, dummies, β1, β2, vec_β, γ1, vec_γ)
        πe_ss(c) = πe(c, dummies, β1, β2, vec_β, γ1, vec_γ)


        if ci_uc < 1e-5

            vec_ciLowerBounds[di] = x -> 0.0
            ciUpperGrids = range(ci_uc, 1.0, length = grid_length)
            vec_ciUpperBounds[di] = cUpperBoundGen(πi_ss, ciUpperGrids)

        elseif ci_uc > 1.0 - 1e-5

            vec_ciUpperBounds[di] = x -> 1.0
            ciLowerGrids = range(ci_uc, 1.0, length = grid_length)
            vec_ciLowerBounds[di] = cLowerBoundGen(πi_ss, ciLowerGrids)

        else
            # so the optimal value is interior
            ciUpperGrids = range(ci_uc, 1.0, length = grid_length)
            vec_ciUpperBounds[di] = cUpperBoundGen(πi_ss, ciUpperGrids)
            ciLowerGrids = range(0.0, ci_uc, length = grid_length)
            vec_ciLowerBounds[di] = cLowerBoundGen(πi_ss, ciLowerGrids)

        end

        if ce_uc < 1e-5

            vec_ceLowerBounds[di] = x -> 0.0
            ceUpperGrids = range(ce_uc, 1.0, length = grid_length)
            vec_ceUpperBounds[di] = cUpperBoundGen(πe_ss, ceUpperGrids)

        elseif ce_uc > 1.0 - 1e-5

            vec_ceUpperBounds[di] = x -> 1.0
            ceLowerGrids = range(ce_uc, 1.0, length = grid_length)
            vec_ceLowerBounds[di] = cLowerBoundGen(πe_ss, ceLowerGrids)

        else
            # so the optimal value is interior
            ceUpperGrids = range(ce_uc, 1.0, length = grid_length)
            vec_ceUpperBounds[di] = cUpperBoundGen(πe_ss, ceUpperGrids)
            ceLowerGrids = range(0.0, ce_uc, length = grid_length)
            vec_ceLowerBounds[di] = cLowerBoundGen(πe_ss, ceLowerGrids)

        end
    end 

    return (vec_ciLowerBounds = vec_ciLowerBounds, vec_ciUpperBounds= vec_ciUpperBounds, vec_ceLowerBounds = vec_ceLowerBounds, vec_ceUpperBounds = vec_ceUpperBounds)

end


