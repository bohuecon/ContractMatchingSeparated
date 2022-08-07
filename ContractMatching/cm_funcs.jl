# define deep functions

g(i, e, ρ) = (0.5 * i^ρ + 0.5 * e^ρ)^(2 / ρ)
h(c, dummies, β1, β2, vec_β) = exp(β1 * c + β2 * c^2 + c * (1 - c) * dot(dummies, vec_β))
γ(c, dummies, γ1, vec_γ) = c * exp(γ1 * c + c * (1 - c) * dot(dummies, vec_γ))
# γ(c, dummies, γ1, γ2, vec_γ) = c * exp(γ1 * c + γ2* c^2 + c * (1 - c) * dot(dummies, vec_γ))
πi(c, dummies, β1, β2, vec_β, γ1, vec_γ) = h(c, dummies, β1, β2, vec_β) # the firm gets all 
πe(c, dummies, β1, β2, vec_β, γ1, vec_γ) = γ(c, dummies, γ1, vec_γ) * h(c, dummies, β1, β2, vec_β) # the executive obtains the rest

function read_contract_para(para, external)

    if external
        return para.β1_ex, para.β2_ex, para.vec_β_ex, para.γ1_ex, para.vec_γ_ex
    else
        return para.β1, para.β2, para.vec_β, para.γ1, para.vec_γ
    end

end

function quasiconcave_objects(para; external = false)

    # read model parameters 
    @unpack vec_dummies, num_dcases, h, ρ = para
    β1, β2, vec_β, γ1, vec_γ = read_contract_para(para, external)

    πi_s(c, dummies) = πi(c, dummies, β1, β2, vec_β, γ1, vec_γ)
    πe_s(c, dummies) = πe(c, dummies, β1, β2, vec_β, γ1, vec_γ)

    i_flag = true
    i_ind = 1

    while i_flag && (i_ind <= num_dcases)

        dummies = vec_dummies[i_ind]
        result_i = optimize(c -> πi_s(c, dummies), 0.0, 1.0)
        ci = result_i.minimizer

        if (ci < 1.0 - 1e-5) && (ci > 1e-5)
            i_flag = false
        end

        i_ind += 1 
    end

    e_flag = true
    e_ind = 1

    while e_flag && (e_ind <= num_dcases)

        dummies = vec_dummies[e_ind]
        result_e = optimize(c -> πe_s(c, dummies), 0.0, 1.0)
        ce = result_e.minimizer

        if (ce < 1.0 - 1e-5) && (ce > 1e-5)
            e_flag = false
        end

        e_ind += 1
    end

    return (i_flag, e_flag)
end


"""
    gen_typegrid(num_i, upper_i, Fi)

Generate grid of agent types and the vector of density probabilities.

### in
    num_i: scalar, number of agent types
    upper_i: scaller, upper type 
    Fi distribution

### out
    vec_i: vector, agent types
    vec_pro_i: vector, probability of each type
"""
function gen_typegrid(num_i, upper_i, Fi)
    interval = upper_i / num_i
    interval_half = 0.5 * interval
    vec_i = range(interval_half, upper_i - interval_half, length = num_i)
    # vec_intervals = range(0.0, upper_i, length = num_i + 1)
    # vec_prob_i = [cdf(Fi, vec_intervals[i+1] / upper_i) - cdf(Fi, vec_intervals[i] / upper_i) for i in 1:num_i]

    return vec_i
end

"""
    gen_typeprob(num_i, upper_i, Fi)

Generate grid of agent types and the vector of density probabilities.

### in
    num_i: scalar, number of agent types
    upper_i: scaller, upper type 
    Fi distribution

### out
    vec_i: vector, agent types
    vec_pro_i: vector, probability of each type
"""
function gen_typeprob(num_i, upper_i, Fi)
    interval = upper_i / num_i
    # interval_half = 0.5 * interval
    # vec_i = range(interval_half, upper_i - interval_half, length = num_i)
    vec_intervals = range(0.0, upper_i, length = num_i + 1)
    vec_prob_i = [cdf(Fi, vec_intervals[i+1] / upper_i) - cdf(Fi, vec_intervals[i] / upper_i) for i in 1:num_i]

    return vec_prob_i
end



"""
    gen_mixedgrid(num_i, upper_i, Fi)

Generate grid of agent types and the vector of density probabilities.

### in
    vec_i: a vector (length num_i) of agent i types
    vec_e: a vector (length num_e) of agent e types
    vec_prob_i: a vector (length num_i) of densities on each type i
    vec_prob_e: a vector (length num_e) of densities on each type e

### out
    mat_ie: matrix num_i x num_e matrix of of [i, e]
    mat_prob_ie: accompanied probability matrix of mat_ie
"""
function gen_mixedgrid(vec_i, vec_e, vec_prob_i, vec_prob_e, num_i, num_e)
    # generate a vector of length num_i x num_e of vector [i,e]
    vec_ie = [[i,e] for i in vec_i for e in vec_e]
    # reshape to obtain num_i x num_e matrix of of [i, e]
    mat_ie = transpose(reshape(vec_ie, num_e, num_i))
    # the accompanied probability matrix of mat_ie
    # mat_prob_ie = vec_prob_i * vec_prob_e'

    return mat_ie
end


"""
    gen_matg(vec_i, vec_e, g, ρ)

Compute the value of g(i,e,ρ) for each pair of i, e. 
    
### in
    vec_i, vec_e
    g: the function to be computed
    para: model parameters
   
### out
    mat_g: ii x ei
"""
function gen_matg(vec_i, vec_e, g, ρ)
    length_i = length(vec_i)
    length_e = length(vec_e)
    mat_g = fill(0.0, length_i, length_e)
    for ii in 1:length_i, ei in 1:length_e
        i = vec_i[ii]
        e = vec_e[ei]
        mat_g[ii, ei] = g(i, e, ρ)
    end
    return mat_g
end
















