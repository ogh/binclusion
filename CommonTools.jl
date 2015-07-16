module CommonTools

require("LoadEnvironment.jl")
require("DataTypes.jl")

using DataTypes

export tanh, tanh!, hard_tanh, ultra_fast_tanh
export square!, l2sq, l2sq_distance
export add!, subtract_vec_matrix!
export rmsPropAddGradScale!, pregrads_l2sq_distance, CVMIntermediates
export adaGradAddGradScale!
export cosine_distance_grad, pregrads_cosine_distance
export mult_by_0!, sigmoid, loopdot
export get_rand
export @nogc

using Corpora.Token

immutable CVMIntermediates
    ofull::Matrix{Float};
    result::Vector{Float};
    local1::Vector{Float};
end

CVMIntermediates(wvd::Int64, max_len::Int64) = CVMIntermediates(
                                 zeros(Float, wvd, max_len),
                                 zeros(Float, wvd),
                                 zeros(Float, wvd));

sigmoid(z::Float) = const_one/(const_one + exp(-z));

# Disable garbage collection in the block this macro is applied to
# and perform one large garbage collection at the end.
# Prevents the garbage collection from being triggered constantly 
# and instead performs one big sweep.
macro nogc(ex)
    quote
        local val
        try
            gc_disable()
            val = $(esc(ex))
        finally
            gc_enable()
            gc()
        end
        val
    end
end

# Get random integer in range between 1 and l
function get_rand(l)
    return 1+int64(trunc(rand() * l))
end

# Dot product of two vectors
function loopdot(result::DenseArray{Float,1}, ip::DenseArray{Float,1})
    o = 0.0
    @simd for i=1:size(result,1)
        @inbounds o += result[i] * ip[i]
    end
    return o
end
                            
# Multiply all entries of a matrix by zero
function mult_by_0!(x::Matrix{Float})
    @simd for i=1:length(x)
        @inbounds x[i] *= const_zero
    end
end

# Multiply all entries of a vector by zero
# TODO: write one generic function that works for all array-type objects
function mult_by_0!(x::Vector{Float})
    @simd for i=1:length(x)
        @inbounds x[i] *= const_zero
    end
end
                                 
# Apply tanh to a float 
function tanh(x::Float)
    ex = exp(2*x)
    return (ex-1)/(ex+1)
end

# Apply tanh in place to all entries of a given matrix
function tanh!(x::DenseArray{Float,2})
    @simd for i=1:length(x)
        x[i] = tanh(x[i])
    end
end

# Crude approximation of tanh (inspired by Theano)
function hard_tanh(x::Float64)
    z = x
    if x < -1.0
        z = -0.995054
    elseif x > 1.0
        z = 0.995054
    end
    return z
end

# Apply approximation of tanh in place to all entries of given array
function hard_tanh!(x::DenseArray{Float})
    @simd for i=1:length(x)
        x[i] = hard_tanh(x[i])
    end
end

# Fast approximation of tanh, inspired by Theano
function ultra_fast_tanh(x::Float64)
    z = 0.0
    if x >= 0.0
        if x < 1.7
            z = (1.5 * x / (1.0 + x))
        elseif x < 3.0
            z = (0.935409070603099 + 0.0458812946797165 * (x - 1.7))
        else
            z = 0.99505475368673
        end
    else
        xx = -x
        if xx < 1.7
            z = (1.5 * xx / (1.0 + xx))
        elseif xx < 3
            z = (0.935409070603099 + 0.0458812946797165 * (xx - 1.7))
        else
            z = 0.99505475368673
        end
        z = -z
    end
    return z
end

# Apply approximation of tanh in place to all entries of given array
function ultra_fast_tanh!(x::DenseArray{Float})
    @simd for i=1:length(x)
        x[i] = ultra_fast_tanh(x[i])
    end
end

# Square all entries of given matrix in place
function square!(m::Matrix{Float})
    for y=1:size(m,2)
        @simd for x=1:size(m,1)
            m[x,y] *= m[x,y]
        end
    end
end

# Compute squared l2 norm of given vector
function l2sq(v::Vector{Float})
    result = 0.0
    @simd for i=1:size(v,1)
        result += v[i] * v[i]
    end
    return result
end

# Compute squared l2 norm of given matrix
# TODO: Write generic function that works for all array-type objects
function l2sq(m::Matrix{Float})
    result = 0.0
    @simd for i=1:length(m)
        @inbounds result += m[i] * m[i]
    end
    return result
end

# Computes cosine distance between two vectors
cosine_distance(w0,w1) = -(dot(w0,w1))/norm(w0)/norm(w1)
# Computes cosine distance gradient
function cosine_distance_grad(w0,w1)
    c = dot(w0,w1)
    s0 = norm(w0)
    s1 = norm(w1)
    sb = s0*s1
    sbsq = sb*sb
    
    grad_w0 = (w1 * sb - c * w0/s0 * s1) / sbsq * (-1)
    grad_w1 = (w0 * sb - c * w1/s1 * s0) / sbsq * (-1)
    
    return grad_w0, grad_w1
end

function pregrads_cosine_distance(v1::Vector{Float}, v2::Vector{Float},
                                vn::Vector{Float})
    # We are still below the margin
    ## precompute 
    grad_v1_1, grad_v2 = cosine_distance_grad(v1, v2)
    grad_v1_2, grad_vn = cosine_distance_grad(v1, vn)
    
    # gradient of the contrastive criterion w.r.t. 
    # the valid pair difference and the noise pair
    # difference
    return vn-v2, v2-v1, v1-vn
end

function l2sq_distance(v1::DenseArray{Float,1}, v2::DenseArray{Float,1})
    @assert length(v1) == length(v2)
    result = 0.0
    @simd for i=1:length(v1)
        @inbounds result += (v1[i]-v2[i])^2
    end
    return result
end

function pregrads_l2sq_distance(v1::DenseArray{Float,1}, v2::DenseArray{Float,1},
                                vn::DenseArray{Float,1})
    # gradient of the contrastive criterion w.r.t. 
    # the valid pair difference and the noise pair
    # difference
    return vn-v2, v2-v1, v1-vn
end
# Adds two vectors v and o up and store the result in o
function add!(v::DenseArray{Float}, o::DenseArray{Float})
    """
        Inplace add function that adds up two vectors.
        :param v: vector 1
        :param o: vector 2, will be modified to contain the result
                  of the addition
    """
    @assert length(v) == length(o)
    @simd for i=1:length(v)
        @inbounds o[i] += v[i]
    end
end

# Subtract vector from given column in matrix
# (in place)
function subtract_vec_matrix!(v::DenseArray{Float,1}, idx::Int,
                                        m::DenseArray{Float,2})
    @simd for i=1:size(v,1)
        m[i,idx] -= v[i]
    end
end

end
