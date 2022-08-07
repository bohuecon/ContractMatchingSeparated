model_para = @with_kw (
    # agent distributions
    ai = 1.927, 
    bi = 3.602,
    Fi = Beta(ai, bi),
    ae = 3.142,
    be = 4.152,
    Fe = Beta(ae, be),
    # grids
    num_i = 30, # grid point number of investors
    num_e = 50, # grid point number of executives
    upper_i = 10.0, # the upper bound of investor types
    upper_e = 10.0, # the upper bound of executive types
    # discrete distributions
    vec_i = gen_typegrid(num_i, upper_i, Fi),
    vec_prob_i = gen_typeprob(num_i, upper_i, Fi),
    vec_e = gen_typegrid(num_e, upper_e, Fe),
    vec_prob_e = gen_typeprob(num_e, upper_e, Fe),
    mat_ie = gen_mixedgrid(vec_i, vec_e, vec_prob_i, vec_prob_e, num_i, num_e),
    mat_prob_ie = vec_prob_i * vec_prob_e', # the accompanied probability matrix of mat_ie
    # contract dummies
    num_dummies = 2,
    arr_dummies = [[d1, d2, d1*d2] for d1 in [0,1], d2 in [0,1]],
    num_dcases = length(arr_dummies),
    vec_dummies = reshape(arr_dummies, (num_dcases,1)), 
    # num_dummies: how many dummy variables
    # num_dcases: how many cases the dummies can construct, 2^num_dummies
    # dummies: a vector with each dummy pick a value
    # vec_dummies: a vector and each element is a dummies
    # meeting technologies
    λi = 8.0,
    λe = 3.2,
    γi = λi * 1.2847, # so that internal probability matches
    γe = λe, 
    r = 0.1,
    # contract parameters
    β1 = 1.5,
    β2 = - β1/0.68,
    vec_β = [0.41, 0.29, -0.001],
    γ1 = 0.151,
    vec_γ = [-1.66, -2.12, 0.001],  
    β1_ex = 1.5,
    β2_ex = - β1_ex/0.68,
    vec_β_ex = [0.41, 0.29, -0.001],
    γ1_ex = 0.151,
    vec_γ_ex = [-1.66, -2.12, 0.001], 
    # outcome parameters 
    κ0 = -455.56,
    κ1 = 14.5,
    κ0_ex = -455.56,
    κ1_ex = 14.5,
    # production technologies
    ρ = -1.370,
    g = g,
    h = h,
    γ = γ,
    πi = πi,
    πe = πe,
    # vectors of contract terms used in computation
    mat_g = gen_matg(vec_i, vec_e, g, ρ),
)

function est_para2model_para(est_para)

    λi, λe, γi, γe, ai, bi, ae, be, ρ, Δβ1, β2, Δβ3, Δβ4, Δβ5, γ1, γ3, γ4, γ5, κ0, κ1, Δβ1_ex, Δβ2_ex, Δβ3_ex, Δβ4_ex, Δβ5_ex, Δγ1_ex, Δγ3_ex, Δγ4_ex, Δγ5_ex, κ0_ex, κ1_ex = est_para

    β1 = -β2 * Δβ1
    β3 = β2 + Δβ3
    β4 = β2 + Δβ4
    β5 = β2 + β3 + β4 + Δβ5

    vec_β = [β3, β4, β5]
    vec_γ = [γ3, γ4, γ5] 

    β2_ex = β2 * Δβ2_ex
    β1_ex = -β2_ex * Δβ1_ex

    β3_ex = β2_ex + Δβ3_ex
    β4_ex = β2_ex + Δβ4_ex
    β5_ex = β2_ex + β3_ex + β4_ex + Δβ5_ex
    vec_β_ex = [β3_ex, β4_ex, β5_ex]

    γ1_ex = γ1*Δγ1_ex
    vec_γ_ex = [γ3*Δγ3_ex, γ4*Δγ4_ex, γ5*Δγ5_ex]

    para = model_para(λi = λi, λe = λe, γi = γi, γe = γe, ai = ai, bi = bi, ae = ae, be = be, ρ = ρ, β1 = β1, β2 = β2, vec_β = vec_β, γ1 = γ1, vec_γ = vec_γ, κ0 = κ0, κ1 = κ1, β1_ex = β1_ex, β2_ex = β2_ex, vec_β_ex = vec_β_ex, γ1_ex =γ1_ex, vec_γ_ex = vec_γ_ex, κ0_ex = κ0_ex, κ1_ex = κ1_ex)

    return para
end 


# the default value
# para_de = model_para()
