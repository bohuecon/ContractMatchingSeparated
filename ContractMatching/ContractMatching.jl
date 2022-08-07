
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

module ContractMatching
export genMoments, para

using Parameters
using Distributions
using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations 
using JLD, HDF5
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

include("cm_parameters.jl")
include("cm_solvemodel.jl")
include("cm_bounds.jl")
include("cm_valueupdate.jl")
include("cm_contracting.jl")
include("cm_policy.jl")
include("cm_funcs.jl")
# include("cm_plots.jl")
include("cm_moments.jl")

function genMoments(;est_para = est_para, diagnosis = false)

    # transform est_para to model para
    para = est_para2model_para(est_para)

    # est_para_initial = [3.65803, 2.72272, 4.94808, 17.0181, 4.11957, 5.3761, 4.04063, 2.58149, -0.547694, 4.59406, -6.87669, 0.290423, -0.793224, -0.240003, 0.479697, -2.84134, 4.54637, 1.99139, -153.041, 3.6081]
    # est_para_initial = [9.2937, 13.0476, 15.8459, 9.2559, 6.3981, 9.2277, 8.5521, 5.9541, -0.9136, 4.0767, -6.8068, 1.1419, -2.5603, 3.7307, 3.097, 4.2259, 4.332, 4.0677, 70.3971, 41.4369]
    # para = est_para2model_para(est_para_initial)

    # check if h() and γ() are quasicancave
    i_quasiconcave_in, e_quasiconcave_in = quasiconcave_objects(para, external = false)
    i_quasiconcave_ex, e_quasiconcave_ex = quasiconcave_objects(para, external = true)

    if diagnosis
        println()
        println("       >>> Internal objectives quasiconcave? i: $i_quasiconcave_in, e: $e_quasiconcave_in.")
        println("       >>> External objectives quasiconcave? i: $i_quasiconcave_ex, e: $e_quasiconcave_ex.")
    end

    quasiconcave = i_quasiconcave_in && e_quasiconcave_in && i_quasiconcave_ex && e_quasiconcave_ex

    if quasiconcave 
    # both objectives are quasiconcave for internal matching and external matching

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

        # check if mat_Mu has a row of zeros, i.e., some types of firms are not matched

        @unpack mat_Mu, mat_Mu_bm = sol

        # there exists some type of firms that can not be matched
        firm_not_match = (minimum(sum(mat_Mu, dims = 2)) == 0) || (minimum(sum(mat_Mu_bm, dims = 2)) == 0)

        # firm_not_match = minimum(sum(mat_Mu .+ mat_Mu_bm, dims = 2)) == 0

        if firm_not_match
            return zeros(35), true, not_convergent
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
        return zeros(35), true, not_convergent
    end
 
end

end



