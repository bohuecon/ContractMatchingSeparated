
# -------------------------------
#
# Description:
# This is the model of internal candidates
# In the model there is one continuous variable and two dummies
#
# -------------------------------
#
# Author: Bo Hu
# Date: 2022 June 16
#
# -------------------------------

module InternalCandi
export genMoments, para

using Parameters
using Distributions
using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations 
# using JLD, HDF5
using LinearAlgebra
using Interpolations
# using DataFrames
# using CSV

# using Plots
# theme(:default)
# using LaTeXStrings

# using Roots
# using Statistics
# using LoopVectorization
# using BenchmarkTools
# using ProfileView

include("internal_candi_parameters.jl")
include("internal_candi_solvemodel.jl")
include("internal_candi_bounds.jl")
include("internal_candi_valueupdate.jl")
include("internal_candi_contracting.jl")
include("internal_candi_policy.jl")
include("internal_candi_funcs.jl")
# include("internal_candi_plots.jl")
include("internal_candi_moments.jl")

function est_para2model_para(est_para)

    λi, λe, γi, γe, ai, bi, ae, be, ρ, β1, β2, β3, β4, β5, γ1, γ3, γ4, γ5, κ0, κ1, β1_ex, β2_ex, β3_ex, β4_ex, β5_ex, γ1_ex, γ3_ex, γ4_ex, γ5_ex, κ0_ex, κ1_ex = est_para

    vec_β = [β3, β4, β5]
    vec_γ = [γ3, γ4, γ5]    

    vec_β_ex = [β3_ex, β4_ex, β5_ex]
    vec_γ_ex = [γ3_ex, γ4_ex, γ5_ex]

    para = model_para(λi = λi, λe = λe, γi = γi, γe = γe, ai = ai, bi = bi, ae = ae, be = be, ρ = ρ, β1 = β1, β2 = β2, vec_β = vec_β, γ1 = γ1, vec_γ = vec_γ, κ0 = κ0, κ1 = κ1, β1_ex = β1_ex, β2_ex = β2_ex, vec_β_ex = vec_β_ex, γ1_ex = γ1_ex, vec_γ_ex = vec_γ_ex, κ0_ex = κ0_ex, κ1_ex = κ1_ex)

    return para
end 


function genMoments(;est_para = est_para, diagnosis = false)

    # transform est para to deep model para

    para = est_para2model_para(est_para)

    # est_para_initial = [3.65803, 2.72272, 4.94808, 17.0181, 4.11957, 5.3761, 4.04063, 2.58149, -0.547694, 4.59406, -6.87669, 0.290423, -0.793224, -0.240003, 0.479697, -2.84134, 4.54637, 1.99139, -153.041, 3.6081]
    # est_para_initial = [9.2937, 13.0476, 15.8459, 9.2559, 6.3981, 9.2277, 8.5521, 5.9541, -0.9136, 4.0767, -6.8068, 1.1419, -2.5603, 3.7307, 3.097, 4.2259, 4.332, 4.0677, 70.3971, 41.4369]
    # para = est_para2model_para(est_para_initial)


    # check if h() and γ() are quasicancave
    # i_flag = true means IT IS QUASICONCAVE

    i_quasiconcave_in, e_quasiconcave_in = quasiconcave_objects(para external = false)
    i_quasiconcave_ex, e_quasiconcave_ex = quasiconcave_objects(para external = true)

    # if diagnosis
    #     println()
    #     println("       >>> Internal objectives quasiconcave? i: $i_quasiconcave_in, e: $e_quasiconcave_in.")
    #     println("       >>> Internal objectives quasiconcave? i: $i_quasiconcave_ex, e: $e_quasiconcave_ex.")
    # end

    quasiconcave = i_quasiconcave_in && e_quasiconcave_in && i_quasiconcave_ex && e_quasiconcave_ex

    if quasiconcave # both objectives are quasiconcave for internal matching and external matching

        # solve the model
        # sol, not_convergent = solve_main(para = para, diagnosis = true)
        sol, not_convergent = solve_main(para = para, diagnosis = diagnosis)
        
        # record not convergent cases 

        if not_convergent
            open("log_err", "a") do f
            write(f, "\n _____________________________________________ \n")
            write(f, "parameters: $est_para; \n \n")
            write(f, "error: not convergent; \n \n ")
            end
        end

        # check if mat_Mu has a row of zeros
        # that is, some type of firms will not be matched

        @unpack mat_Mu = sol

        firm_not_match = minimum(sum(mat_Mu, dims = 2)) == 0

        if firm_not_match
            return zeros(20), true, not_convergent
        else
            # generate moments
            modelMoment = compute_moments(sol, para)
            return modelMoment, false, not_convergent
        end

        # if minimum(sum(mat_Mu, dims = 1)) * minimum(sum(mat_Mu, dims = 2)) == 0
        #     error_flag = true
        # end

        # if error_flag # does not converge
        #     return zeros(20), error_flag

        # else
        #     # generate moments
        #     modelMoment = compute_moments(sol, para)
        #     return modelMoment, false
        # end

    else
        not_convergent = false
        return zeros(20), true, not_convergent
    end
 
end

end



