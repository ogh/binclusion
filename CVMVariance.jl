module Variance

require("Corpora.jl")
require("CommonTools.jl")
require("CVMAbstract.jl")
require("DataTypes.jl")

## Imports
using DataTypes
using Base.LinAlg.BLAS.axpy!
using Corpora.Token
using Corpora.text
using CommonTools: tanh, square!, l2sq, l2sq_distance, mult_by_0!
using CommonTools: add!, subtract_vec_matrix!
using Base.LinAlg.BLAS.scal!
using CommonTools.CVMIntermediates

using CVMAbstractModel
import CVMAbstractModel: compose!, store_gradients!

## Exports
export VarianceModel
export compose
export compose!
export store_gradients!
export VarianceParameters

immutable VarianceModel <: AbstractModel
    wordvecdim::Int64;
    margin::Float; 
    regularizer::Float;
    lr_init::Float;
    lr_min::Float;
    niter::Int64;
    gamma::Float;
    nnoise::Int64;
    batchsize::Int64;
end

type VarianceParameters <: AbstractParameters
    wordvecs::Matrix{Float};
    gradients::Matrix{Float};
end

VarianceParameters() = VarianceParameters(Array(Float,0,0), Array(Float,0,0))

function l2sq!(v1::DenseArray{Float,1}, v2::DenseArray{Float,1},
                    o::DenseArray{Float,1}) 
    @simd for i=1:size(v1,1)
        o[i] += (v1[i]-v2[i])^2
    end
end

function compose!(model::VarianceModel, local1::Vector{Float},
                    s::Vector{Token}, o::Vector{Float})
    # Calculate the mean
    m = local1
    mult_by_0!(m)
    #= m = zeros(Float, model.wordvecdim) =#
    for i=1:size(s,1)
        #= Base.LinAlg.BLAS.axpy!(const_one, s[i].vector, m) =#
        add!(s[i].vector, m)
        #@simd for j=1:model.wordvecdim
        #    m[j] += s[i].vector[j]
        #end
    end
    scal!(model.wordvecdim, 1.0/size(s,1), m, 1)
    mult_by_0!(o);
    tmp = 0.0
    for i=1:size(s,1)
        l2sq!(s[i].vector, m, o)
        #= @simd for j=1:model.wordvecdim =#
        #=     tmp = s[i].vector[j] - m[j] =#
        #=     o[j] += tmp * tmp =# 
        #=     #o[j] += (s[i].vector[j] - m[j])^2 =#  
        #= end =#
    end
    scal!(model.wordvecdim, 1.0/size(s,1), o, 1)

    return o
end

function compose!(model::VarianceModel, s::Vector{Token}, 
                                intermediate::CVMIntermediates)
    return compose!(model, intermediate.local1, s, intermediate.result)
end

function compose(model::VarianceModel, s::Vector{Token})
    # Calculate the mean
    m = zeros(Float, model.wordvecdim)
    for i=1:size(s,1)
        add!(s[i].vector,m)
    end
    
    # Calculate the output
    o = zeros(Float, model.wordvecdim)
    for i=1:size(s,1)
        @simd for j=1:model.wordvecdim
            o[j] += (s[i].vector[j] .- m[j])^2  
        end
    end
    return o
end

function multstore!(scalar::Float, vec::DenseArray{Float,1},
                    idx, m::DenseArray{Float,2})
    @simd for j=1:size(vec,1)
        m[j,idx] = scalar
        m[j,idx] *= vec[j]
    end
end

function store_gradients!(model::VarianceModel,
                    pre_grad::Vector{Float},
                    s::Vector{Token}, intermediate::CVMIntermediates;
                    intermediates_precomputed::Bool=false)

    #= result = Array(Float64,(model.wordvecdim,size(s,1))) =#
    result = intermediate.ofull
    tmp_scaling = 2.0/size(s,1)
    for i=1:size(s,1)
        multstore!(tmp_scaling, s[i].vector, i, result)
    end
    m = zeros(model.wordvecdim)
    for i=1:size(s,1)
        #= Base.LinAlg.BLAS.axpy!(const_one, s[i].vector, m) =#
        add!(s[i].vector,m)
        #@simd for j=1:model.wordvecdim
        #    m[j] += s[i].vector[j]
        #end
    end
    factor = 2.0/(size(s,1)^2)
    scal!(model.wordvecdim, factor, m,1)
    for i=1:size(s,1)
        subtract_vec_matrix!(m, i, result)
        #= @simd for j=1:model.wordvecdim =#
        #=     result[j,i] -= m[j] =#
        #= end =#
    end
    
    tmp = intermediate.local1
    #= tmp = Array(Float,model.wordvecdim) =#
    for i=1:size(s,1)
        @simd for j=1:model.wordvecdim
            tmp[j] = result[j,i] * pre_grad[j]
        end
        add!(tmp, s[i].gradient)
    end
end

end
