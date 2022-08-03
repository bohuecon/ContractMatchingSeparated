para = @with_kw (
    ## set up of agent types
    ai = 1.927, # parameters in investor distribution
    bi = 3.602,
    Fi = Beta(ai, bi),
    ae = 3.142,
    be = 4.152,
    Fe = Beta(ae, be),
    num_i = 30, # grid point number of investors
    num_e = 50, # grid point number of executives
    upper_i = 10.0, # the upper bound of investor types
    upper_e = 10.0, # the upper bound of executive types
    ## discrite distribution of agent types
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
    ## meeting rates
    λi = 8.0,
    λe = 3.2,
    γi = λi * 1.2847, # so that internal probability matches
    γe = λe, 
    r = 0.1,
    ## parameters in functions
    ρ = -1.370,
    # β1 = 0.679,
    β1 = 1.5,
    β2 = - β1/0.68,
    # β3 = -0.163,
    vec_β = [0.41, 0.29, -0.001],
    γ1 = 0.151,
    # γ2 = -0.200,
    vec_γ = [-1.66, -2.12, 0.001],  
    κ0 = -455.56,
    κ1 = 14.5,
    ## deep functions
    g = g,
    h = h,
    γ = γ,
    # πt = πt,
    πi = πi,
    πe = πe,
    ## vectors of contract terms used in computation
    mat_g = gen_matg(vec_i, vec_e, g, ρ),
)


