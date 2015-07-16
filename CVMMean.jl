module Mean

require("Corpora.jl")
require("CommonTools.jl")
require("CVMAbstract.jl")
require("DataTypes.jl")

## Imports
using DataTypes
using Base.LinAlg.BLAS.axpy!
using Corpora.Token
using Corpora.text
using CommonTools: tanh, square!, l2sq, l2sq_distance, mult_by_0!, add!
using CommonTools.CVMIntermediates

using CVMAbstractModel
import CVMAbstractModel: compose!, store_gradients!

## Exports
export MeanModel
export compose, step!, compose_w_l2
export compose!
export store_gradients!
export MeanParameters

immutable MeanModel <: AbstractModel
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

type MeanParameters <: AbstractParameters
    wordvecs::Matrix{Float};
    gradients::Matrix{Float};
end

MeanParameters() = MeanParameters(Array(Float,0,0), Array(Float,0,0))

function compose_w_l2(model::MeanModel, s::Vector{Token},
                            meanvec::Vector{Float})
    """
        Compose a vector representation for a sentence while calculating
        the l2 norm of the sentence's word vectors at the same time.
        The sentence is given as a list of tokens.
    """
    
    Base.LinAlg.BLAS.scal!(model.wordvecdim,0.0,meanvec,1);
    l2reg = 0.0
    for iword=1:size(s,1)
        v1 = s[iword].vector
        axpy!(1.0, v1, meanvec)
        l2reg += l2sq(v1)*0.5
    end
    Base.LinAlg.BLAS.scal!(model.wordvecdim,1.0/size(s,1),meanvec,1);
    return meanvec, l2reg
end

function compose_w_l2(model::MeanModel, s::Vector{Token})
    return compose_w_l2(model, s, zeros(Float, model.wordvecdim));
end

function compose!(model::MeanModel, s::Vector{Token}, o::Vector{Float})
    """
        Compose a sentence vector from a list of tokens while
        storing intermediates results.
        In-place
    """
    mult_by_0!(o);
    for iword=1:size(s,1)
        add!(s[iword].vector, o)
    end
    Base.LinAlg.BLAS.scal!(model.wordvecdim,1.0/size(s,1),o,1);
    return o
end

function compose!(model::MeanModel, s::Vector{Token}, 
                                intermediate::CVMIntermediates)
    return compose!(model, s, intermediate.result)
end

function compose(model::MeanModel, s::Vector{Token})
    """
        Compose a sentence vector from a list of tokens while
        storing intermediates results.
    """
    o = zeros(Float, model.wordvecdim)
    for iword=1:size(s,1)
        o += s[iword].vector
    end
    Base.LinAlg.BLAS.scal!(model.wordvecdim,1.0/size(s,1),o,1);
    return o
end


function store_gradients!(model::MeanModel,
                    pre_grad::Vector{Float},
                    s::Vector{Token}, intermediate::CVMIntermediates;
                    intermediates_precomputed::Bool=false)
    """
        Given a gradient from a higher level in the neural network,
        back-propagate the gradient through to the word representations.
    """
    scaling = 1.0/size(s,1)
    for token in s
        # Contrastive
        #add!(pre_grad, token.gradient)
        axpy!(scaling, pre_grad, token.gradient)
    end
end

end
