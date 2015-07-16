module Optimizers

require("LoadEnvironment.jl")
require("DataTypes.jl")

using DataTypes

export Optimizer;
export step!
export AbsAdaGrad
export AdaGrad
export RMSProp
export AdaDelta

abstract Optimizer;

###################   AdaGrad   ######################

immutable AbsAdaGrad <: Optimizer
    parameters::DenseArray{Float}
    gradients::DenseArray{Float}
    rms::DenseArray{Float}
end

function AbsAdaGrad(parameters::DenseArray{Float},
                 gradients::DenseArray{Float})
    @assert length(parameters) == length(gradients)
    return AbsAdaGrad(parameters, gradients,
                   zeros(size(parameters)))
end

function step!(opt::AbsAdaGrad, lr::Float)
    step_absadagrad!(opt.parameters, opt.gradients, opt.rms, lr)
end

function step_absadagrad!(parameters::DenseArray{Float}, 
                        gradients::DenseArray{Float}, 
                        rms::DenseArray{Float}, lr::Float)
    @simd for i=1:length(parameters)
        # Update the sum of squared gradients
        @inbounds rms[i] += abs(gradients[i])^3;
        # Scale gradients using rms
        @inbounds gradients[i] /= rms[i]^(1.0/3.0)
        # Update parameters
        @inbounds parameters[i] -= lr * gradients[i]
        # Reset the batch gradients to zero
        @inbounds gradients[i] = 0.0
    end
end


###################   AdaGrad   ######################

immutable AdaGrad <: Optimizer
    parameters::DenseArray{Float}
    gradients::DenseArray{Float}
    rms::DenseArray{Float}
end

function AdaGrad(parameters::DenseArray{Float},
                 gradients::DenseArray{Float})
    @assert length(parameters) == length(gradients)
    return AdaGrad(parameters, gradients,
                   zeros(size(parameters)))
end

function step!(opt::AdaGrad, lr::Float)
    step_adagrad!(opt.parameters, opt.gradients, opt.rms, lr)
end

function step_adagrad!(parameters::DenseArray{Float}, 
                        gradients::DenseArray{Float}, 
                        rms::DenseArray{Float}, lr::Float)
    @simd for i=1:length(parameters)
        # Update the sum of squared gradients
        @inbounds rms[i] += gradients[i]^2;
        # Scale gradients using rms
        @inbounds gradients[i] /= sqrt(rms[i])
        # Update parameters
        @inbounds parameters[i] -= lr * gradients[i]
        # Reset the batch gradients to zero
        @inbounds gradients[i] = 0.0
    end
end


###################   RMSProp ######################

immutable RMSProp <: Optimizer
    parameters::DenseArray{Float}
    gradients::DenseArray{Float}
    rms::DenseArray{Float}
    decay::Float
end

function RMSProp(parameters::DenseArray{Float},
                 gradients::DenseArray{Float},
                 decay::Float)
    @assert length(parameters) == length(gradients)
    return RMSProp(parameters, gradients,
                   ones(size(parameters)),
                   decay)
end

function step!(opt::RMSProp, lr::Float)
    step_rmsprop!(opt.parameters, opt.gradients, opt.rms, 
                    opt.decay, lr)
end

function step_rmsprop!(parameters::DenseArray{Float}, 
                        gradients::DenseArray{Float}, 
                        rms::DenseArray{Float}, 
                        decay::Float, lr::Float)
    invdecay = const_one - decay
    @simd for i=1:length(parameters)
        # Update the sum of squared gradients
        @inbounds rms[i] *= decay
        @inbounds rms[i] += gradients[i]^2 * invdecay;
        # Scale gradients using rms
        @inbounds gradients[i] /= sqrt(rms[i])
        # Update parameters
        @inbounds parameters[i] -= lr * gradients[i]
        # Reset the batch gradients to zero
        @inbounds gradients[i] = 0.0
    end
end

############## AdaDelta ################

immutable AdaDelta <: Optimizer
    parameters::DenseArray{Float}
    gradients::DenseArray{Float}
    rms::DenseArray{Float}
    s::DenseArray{Float}
    decay::Float
end

function AdaDelta(parameters::DenseArray{Float},
                 gradients::DenseArray{Float},
                 decay::Float)
    @assert length(parameters) == length(gradients)
    return AdaDelta(parameters, gradients,
                   zeros(size(parameters)),
                   zeros(size(parameters)),
                   decay)
end

function step!(opt::AdaDelta, lr::Float)
    step_adadelta!(opt.parameters, opt.gradients, opt.rms, 
                    opt.s, opt.decay, lr)
end

function step_adadelta!(parameters::DenseArray{Float}, 
                        gradients::DenseArray{Float}, 
                        rms::DenseArray{Float}, s::DenseArray{Float},
                        decay::Float, lr::Float)
    invdecay = const_one - decay
    _step = const_zero
    @simd for i=1:length(parameters)
        # Update the sum of squared gradients
        @inbounds rms[i] *= decay
        @inbounds rms[i] += gradients[i]^2 * invdecay;
        # Scale gradients using rms
        # The gradients therfore become update steps
        # Note: We are modifying the values stored in the
        # gradient array!
        _step = sqrt(s[i]+1e-8)/sqrt(rms[i]+1e-8) * gradients[i]
        # Update parameters
        #= @inbounds parameters[i] -= gradients[i] =#
        @inbounds parameters[i] -= _step
        # Update the sum of squared steps
        @inbounds s[i] *= decay
        @inbounds s[i] += _step^2 * invdecay;
        # Reset the gradients
        @inbounds gradients[i] = 0.0
    end
end

end
