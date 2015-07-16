
module CVMAbstractModel

require("CommonTools.jl")

using DataTypes
using Corpora.Token
using CommonTools.CVMIntermediates
using CommonTools: mult_by_0!, sigmoid, loopdot, get_rand
using CommonTools: tanh, square!, l2sq, l2sq_distance

export AbstractModel
export AbstractParameters
export accumulate_gradients_batch!
export accumulate_closeness_gradients_batch!
export objective_gradients_batch
export regularize_l2!

abstract AbstractModel;
abstract AbstractParameters;


# Whenever we create a new model, these two methods need to be implemented
# (compose! and store_gradients!)

# Given the gradient of the error function w.r.t. the phrase representation,
# backpropagate this gradient to the parameters of the model
function store_gradients!(model::AbstractModel, 
                    pre_grad::Vector{Float},
                    s::Vector{Token}, intermediate::CVMIntermediates;
                    intermediates_precomputed::Bool=false)
    error("This is an abstract method");
end

# Given a list of tokens, compose a phrase vector representation
function compose!(model::AbstractModel, 
                                    s::Vector{Token}, 
                                    intermediate::CVMIntermediates)
    error("This is an abstract method");
end

# Adds l2 regularization to the given parameters
function regularize_l2!(model::AbstractModel, params::AbstractParameters;
                            sparse_updates::Bool=false)
    acc_reg = const_zero;
    if model.regularizer > const_zero
        if sparse_updates
            # Only add regularization to the gradients that are not 0
            for i=1:length(params.wordvecs)
                if params.gradients[i] != 0
                    params.gradients[i] += params.wordvecs[i] * model.regularizer
                    acc_reg += model.regularizer * params.wordvecs[i]^2
                end
            end
        else
            # Add l2 regularization
            Base.LinAlg.BLAS.axpy!(model.regularizer, params.wordvecs, params.gradients)
            # Compute l2 error
            acc_reg += l2sq(params.wordvecs) * model.regularizer
        end
    end
    return acc_reg*0.5f0;
end

# Accumulates the gradients of |s1-s2|^2
function accumulate_closeness_gradients_batch!(model::AbstractModel, 
                s1::Vector{Token}, s2::Vector{Token}, 
                intermediates::Vector{CVMIntermediates}; scaling=1.0)
    err = 0.0
    # Compose vector representation sentence1 and sentence2
    o1 = compose!(model, s1, intermediates[1])
    o2 = compose!(model, s2, intermediates[2])
    
    # Cache distance of the pair of aligned sentences (valid pair)
    dist_o1_o2 = l2sq_distance(o1, o2)
    # Update error
    err += scaling * dist_o1_o2
    # Calculate the gradient of |o1-o2|^2 w.r.t. o1
    pre_grad_closeness = (o1-o2)
    # Propagate that gradient to the individual word representations
    # of sentence1
    Base.LinAlg.BLAS.scal!(model.wordvecdim,
                                    scaling,
                                    pre_grad_closeness, 1);
    store_gradients!(model, pre_grad_closeness, s1, intermediates[1])
    # The gradient of |o1-o2|^2 w.r.t. o2 is just the negative o1 gradient 
    # => invert the sign
    Base.LinAlg.BLAS.scal!(model.wordvecdim,
                                    -1.0,
                                    pre_grad_closeness, 1);
    # Propagage the gradient down to the word representations of sentence 2
    store_gradients!(model, pre_grad_closeness, s2, intermediates[2])
    
    return err;
end 

# Calculates the error of the hinge loss and the closeness
function objective_gradients_batch(model::AbstractModel, 
                s1::Vector{Token}, s2::Vector{Token}, 
                sns::Vector{Vector{Token}},
                intermediates::Vector{CVMIntermediates};
                scaling::Float=const_one)

    # Calculate the representations of sentence1 and sentence2
    # In case of the inclusion criterion, sentence1 is the outer window
    # and sentence 2 is the inner window.
    o1 = compose!(model, s1, intermediates[1])
    o2 = compose!(model, s2, intermediates[2])
    
    # Cache distance
    dist_o1_o2 = l2sq_distance(o1, o2)
    # Update error
    err_closeness = dist_o1_o2 * 0.5
    
    # Calculate the contrastive error (hinge loss) for each noise sample
    err_contrastive = 0.0
    inv_len_sns = 1.0/size(sns,1)
    cn_scaling = scaling * model.gamma * inv_len_sns  
    for sn in sns
        # Compose the vector representation of the noise sample
        on = compose!(model, sn, intermediates[3])

        # Compute the contastive part of the hinge loss
        contrastive = 0.5*(dist_o1_o2 - l2sq_distance(o1, on))
        
        # Calculate the hinge loss
        contrastive_err = max(0.0, model.margin + contrastive)
        
        # If hinge loss greater than 0: Update the accumulative error
        if  contrastive_err > 0
            err_contrastive += contrastive_err * cn_scaling;
        end
    end
    
    return err_contrastive + err_closeness;
end 


# Accumulates the gradients of the hinge loss and the closeness function
function accumulate_gradients_batch!(model::AbstractModel, 
                s1::Vector{Token}, s2::Vector{Token}, 
                sns::Vector{Vector{Token}},
                intermediates::Vector{CVMIntermediates};
                scaling::Float=const_one)
    
    # Calculate the representations of sentence1 and sentence2
    # In case of the inclusion criterion, sentence1 is the outer window
    # and sentence 2 is the inner window.
    o1 = compose!(model, s1, intermediates[1])
    o2 = compose!(model, s2, intermediates[2])
    
    # Cache distance of the pair of aligned sentences (valid pair)
    dist_o1_o2 = l2sq_distance(o1, o2)
    err_closeness = dist_o1_o2 * 0.5
    
    # Calculate the gradient of |o1-o2|^2 w.r.t. o1
    pre_grad_closeness = (o1-o2)
    Base.LinAlg.BLAS.scal!(model.wordvecdim,
                                    scaling,
                                    pre_grad_closeness, 1);
    # Propagage the gradient down to the word representations of sentence 2
    store_gradients!(model, pre_grad_closeness, s1, intermediates[1])
    # The gradient of |o1-o2|^2 w.r.t. o2 is just the negative o1 gradient 
    # => invert the sign
    Base.LinAlg.BLAS.scal!(model.wordvecdim,
                                    -1.0, 
                                    pre_grad_closeness, 1);
    # Propagage the gradient down to the word representations of sentence 2
    store_gradients!(model, pre_grad_closeness, s2, intermediates[2])
    
    err_contrastive = 0.0
    # Scale the number contribution of each individual noise sample
    # by 1/their number
    inv_len_sns = 1.0/size(sns,1)
    cn_scaling = scaling * model.gamma * inv_len_sns
    # Apply the scaling to the |o1-o2|^2 w.r.t. o2 gradient 
    Base.LinAlg.BLAS.scal!(model.wordvecdim,
                                    model.gamma * inv_len_sns, 
                                    pre_grad_closeness, 1);
    for sn in sns
        on = compose!(model, sn, intermediates[3])
        contrastive = 0.5*(dist_o1_o2 - l2sq_distance(o1, on))
        
        # Calculate the hinge loss
        contrastive_err = max(0.0, model.margin + contrastive)
        
        # If hinge loss greater than 0: Minimize the contrastive distances
        if  contrastive_err > 0

            # Compute gradients of |o1,o2|^2-|o1,on|^2 w.r.t. on
            pre_gradn = o1-on
            Base.LinAlg.BLAS.scal!(model.wordvecdim,
                                    cn_scaling, 
                                    pre_gradn, 1);
            ### Dangerous! Overwriting "on" and reuising it is potentially
            # dangerous!!
            # Original, clean version
            #pre_grad1 = on-o2
            # hacky, high-speed version reusing "on"
            pre_grad1 = on
            Base.LinAlg.BLAS.axpy!(-1.0, o2, pre_grad1)
            Base.LinAlg.BLAS.scal!(model.wordvecdim, 
                                    cn_scaling, 
                                    pre_grad1, 1);
            
            # Add the gradients up in the gradient field of each token
            ## Hinge objective
            store_gradients!(model, pre_grad1, s1, intermediates[1],
                                    intermediates_precomputed=true)
            # Could optimize this by pulling it out of the loop
            store_gradients!(model, pre_grad_closeness, s2, intermediates[2],
                                    intermediates_precomputed=true)
            # Propagage the gradients down to the word representations
            # of the noise sentence
            store_gradients!(model, pre_gradn, sn, intermediates[3])
           
            # Update accumulative error
            err_contrastive += contrastive_err * model.gamma;
        end
    end
    
    return err_closeness, err_contrastive;
end 

end
