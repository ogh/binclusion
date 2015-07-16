module Addition

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
export AdditionModel
export compose, step!, compose_w_l2
export compose!
export store_gradients!
export AdditionParameters

immutable AdditionModel <: AbstractModel
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

type AdditionParameters <: AbstractParameters
    wordvecs::Matrix{Float};
    gradients::Matrix{Float};
end

AdditionParameters() = AdditionParameters(Array(Float,0,0), Array(Float,0,0))


## Composes the given sentence (s) and stores the result in o
function compose_w_l2!(model::AdditionModel, s::Vector{Token},
                            o::Vector{Float})
    """
        Compose a vector representation for a sentence while calculating
        the l2 norm of the sentence's word vectors at the same time.
        The sentence is given as a list of tokens.
    """
    
    # Make sure that o is initialized to 0.0 (assumes that o does
    # not contains NaN or Inf values
    Base.LinAlg.BLAS.scal!(model.wordvecdim,0.0,o,1);
    l2reg = 0.0
    for iword=1:size(s,1)
        v = s[iword].vector
        add!(v, o)
        l2reg += l2sq(v)*0.5
    end
    return o, l2reg
end

function compose_w_l2(model::AdditionModel, s::Vector{Token})
    return compose_w_l2(model, s, zeros(Float, model.wordvecdim));
end

# Composes the given sentence s and store the result in o.
# Composing meas simply adding up all the word vectors
function compose!(model::AdditionModel, s::Vector{Token}, o::Vector{Float})
    """
        Compose a sentence vector from a list of tokens while
        storing intermediates results.
        In-place
    """
    # Make sure that the result is initialized to all zeros
    # Undefined behavior if o contains NaN or Inf
    mult_by_0!(o);
    for iword=1:size(s,1)
        add!(s[iword].vector, o)
    end
    return o
end

function compose!(model::AdditionModel, s::Vector{Token}, 
                                intermediate::CVMIntermediates)
    return compose!(model, s, intermediate.result)
end

# Compose the given sentence and return the result
function compose(model::AdditionModel, s::Vector{Token})
    """
        Compose a sentence vector from a list of tokens while
        storing intermediates results.
    """
    o = zeros(Float, model.wordvecdim)
    for iword=1:size(s,1)
        o += s[iword].vector
    end
    return o
end


# Propagate the given gradient down to the word representations
function store_gradients!(model::AdditionModel,
                    pre_grad::Vector{Float},
                    s::Vector{Token}, intermediate::CVMIntermediates;
                    intermediates_precomputed::Bool=false)
    """
        Given a gradient from a higher level in the neural network,
        back-propagate the gradient through to the word representations.
    """
    # Since the composition function is addition, we simply
    # need to add the given gradient to the word repr gradients
    for token in s
        add!(pre_grad, token.gradient)
    end
end

end
