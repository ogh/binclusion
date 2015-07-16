module BigramMean

require("Corpora.jl")
require("CommonTools.jl")
require("CVMAbstract.jl")
require("DataTypes.jl")

## Imports
using DataTypes
using Base.LinAlg.BLAS.axpy!
using Corpora.Token
using Corpora.text
using CommonTools: tanh, tanh!, ultra_fast_tanh, ultra_fast_tanh!
using CommonTools: hard_tanh, hard_tanh!
using CommonTools.CVMIntermediates

using CVMAbstractModel
import CVMAbstractModel: compose!, store_gradients!

## Exports
export BigramMeanModel
export BigramMeanParameters
export compose, step!, compose_w_l2
export compose!
export store_gradients!

immutable BigramMeanModel <: AbstractModel
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

# Stores intermediate values
type BigramMeanParameters <: AbstractParameters
    wordvecs::Matrix{Float};
    gradients::Matrix{Float};
    rms::Matrix{Float};
    pivot_vecs_l::Matrix{Float};
    pivot_gradients_l::Matrix{Float};
    pivot_rms_l::Matrix{Float};
    pivot_vecs_r::Matrix{Float};
    pivot_gradients_r::Matrix{Float};
    pivot_rms_r::Matrix{Float};
end

function BigramMeanParameters(wordvecs::Matrix{Float},
                            gradients::Matrix{Float},
                            rms::Matrix{Float})
    return BigramMeanParameters(wordvecs, gradients, rms,
                              Array(Float,0,0), Array(Float,0,0),
                              Array(Float,0,0),
                              Array(Float,0,0), Array(Float,0,0),
                              Array(Float,0,0))
end

BigramMeanParameters() = BigramMeanParameters(Array(Float,0,0), Array(Float,0,0),
                                                         Array(Float,0,0))



function constrastive_update1!(o::Matrix{Float64}, idx::Int64, 
                                 pre_grad::Vector{Float64},
                                 target::DenseArray{Float64,1}, 
                                 scaling::Float64)
    """
        Propagates the given ``pre_grad`` gradient through the first
        or last vector bigram of the a sentence.
    """
    @simd for i=1:size(target,1)
        # We can compute the derivative of of tanh(x) = o as 1-o^2
        # Since o is elementwise squared already, it reduces to (1-o)
        # o gets elementwise square in the function that calls this method!
        target[i] += (1.0 - o[i,idx]) * pre_grad[i] * scaling;
    end                          
end

function constrastive_update2!(o::Matrix{Float64}, idx1::Int64, idx2::Int64, 
                                 pre_grad::Vector{Float64},
                                 target::DenseArray{Float64,1}, 
                                 scaling::Float64)
    """
        Propagates the given ``pre_grad`` gradient through a vector bigram 
        in the middle of a sentence.
    """
    @simd for i=1:size(target,1)
        target[i] += (2.0 - o[i,idx1] - o[i,idx2]) * pre_grad[i] * scaling;
    end                          
end

function compose_w_l2(model::BigramMeanModel, s::Vector{Token},
                        o::Matrix{Float64}, meanvec::Vector{Float64})
    """
        Compose a vector representation for a sentence while calculating
        the l2 norm of the sentence's word vectors at the same time.
        The sentence is given as a list of tokens.
    """
    l2reg = l2(s[1].vector)
    for iword=2:size(s,1)
        iw1 = iword-1
        v1 = s[iword-1].vector
        v2 = s[iword].vector
        @simd for io=1:model.wordvecdim
            @inbounds o[io,iw1] = tanh(v1[io] + v2[io])
        end
        #l2reg += sum(v2.^2)
        l2reg += l2(v2)
    end

    # Get column mean of o
    size_o_2 = size(s,1)-1
    size_o_1 = model.wordvecdim
    Base.LinAlg.BLAS.scal!(model.wordvecdim,0.0,meanvec,1);
    for y=1:size_o_2
        @simd for x=1:size_o_1
            @inbounds meanvec[x] += o[x,y]
        end
    end
    return meanvec, o, l2reg
end

function compose_w_l2(model::BigramMeanModel, s::Vector{Token},
                        o::Matrix{Float64})
    return compose_w_l2(model, s, o, zeros(Float64, model.wordvecdim));
end

function compose_w_l2(model::BigramMeanModel, s::Vector{Token})
    return compose_w_l2(model, s, zeros(Float64, model.wordvecdim, size(s,1)-1));
end

function addToColTanh!(icol, v1, v2, o)
    """
        High-speed function that sums up two vectors and
        applies tanh componentwise to the result.
    """
    @simd for i=1:length(v1)
        @inbounds o[i, icol] = v1[i]
        @inbounds o[i, icol] += v2[i]
        @inbounds o[i, icol] = ultra_fast_tanh(o[i,icol])
    end
end

function compose!(model::BigramMeanModel, s::Vector{Token}, 
                                                    o::Matrix{Float64},
                                                    result::Vector{Float64})
    """
        Compose a sentence vector from a list of tokens while
        storing intermediates results.
    """
    # BOTH: o and result are modified in this function
                                                    
    # Create the intermediary result matrix "o"
    for iword=2:size(s,1)
        iw1 = iword-1
        v1 = s[iw1].vector
        v2 = s[iword].vector
        addToColTanh!(iw1, v1, v2, o)
    end

    # Add the columns of o up to get the final result
    size_o_2 = size(s,1)-1
    size_o_1 = model.wordvecdim
    Base.LinAlg.BLAS.scal!(model.wordvecdim,0.0,result,1);
    for y=1:size_o_2
        @simd for x=1:size_o_1
            result[x] += o[x,y]
        end
    end
    Base.LinAlg.BLAS.scal!(model.wordvecdim,1.0/size_o_2,result,1);
    return result
end

function compose(model::BigramMeanModel, s::Vector{Token})
    return compose!(model, s, zeros(model.wordvecdim, size(s,1)),
                    zeros(Float64, model.wordvecdim))
end

function compose!(model::BigramMeanModel, s::Vector{Token}, 
                                    intermediate::CVMIntermediates)
    return compose!(model, s, intermediate.ofull, intermediate.result)
end


function store_gradients!(model::BigramMeanModel, pre_grad::Vector{Float64},
                    s::Vector{Token}, intermediate::CVMIntermediates;
                    intermediates_precomputed::Bool=false)
    """
        Given a gradient from a higher level in the neural network,
        back-propagate the gradient through to the word representations.
    """
    
    #o = copy(intermediate.ofull)
    o = intermediate.ofull
    size_o = size(s,1)-1
    # This line makes the difference between addition and mean
    scaling = 1.0/size_o
    
    if !intermediates_precomputed
        # Square the output (because tanh'(x) = 1 - tanh(x)^2)
        for y=1:size_o
            @simd for x=1:size(o,1)
                @inbounds o[x,y] *= o[x,y]
            end
        end
    end
    
    constrastive_update1!(o, 1, pre_grad, s[1].gradient,
                            scaling);
    for iword=2:size_o
        constrastive_update2!(o, iword-1, iword, pre_grad, s[iword].gradient,
                                scaling)
    end
    constrastive_update1!(o, size_o, pre_grad, s[end].gradient,
                            scaling);
end

end
