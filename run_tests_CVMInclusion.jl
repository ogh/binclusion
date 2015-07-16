require("CVMMean.jl")
require("CVMAddition.jl")
require("CVMVariance.jl")
require("CVMBigramAddition.jl")
require("CVMBigramMean.jl")
require("Corpora.jl")
require("CommonTools.jl")

using Variance
using BigramMean
using BigramAddition
using Addition
using Mean
using ArrayViews
using CVMAbstractModel
using Corpora
using CommonTools.CVMIntermediates
using Calculus

srand(300)

for (ModelModule, Model, Parameters) in [
        (Variance, Variance.VarianceModel, Variance.VarianceParameters),
        # The bigram based models will fail the gradient check
        # since we are using an approximate version of tanh for speed reasons
        #(BigramMean, BigramMean.BigramMeanModel, BigramMean.BigramMeanParameters),
        #(BigramAddition, BigramAddition.BigramAdditionModel, BigramAddition.BigramAdditionParameters),
        (Mean, Mean.MeanModel, Mean.MeanParameters),
        (Addition, Addition.AdditionModel, Addition.AdditionParameters)]
    println("ModelModule: ", ModelModule,
                " Model: ", Model, " Parameters: ", Parameters)

    model = Model(5, 5.0, 0.0, 0.0, 0.0, 0, 1.0, 10, 1)

    function init_params(nvocab::Int64, params::AbstractParameters)
        params.wordvecs = randn(model.wordvecdim, nvocab) * 0.1
        params.gradients = zeros(model.wordvecdim, nvocab)
    end


    function init_data_token(word, iword, params::AbstractParameters)
        return DataToken(word, iword, [1], 
                            view(params.wordvecs, :, iword),
                            view(params.gradients, :, iword))
    end

    intermediates = [CVMIntermediates(model.wordvecdim, 25),
                    CVMIntermediates(model.wordvecdim, 25),
                    CVMIntermediates(model.wordvecdim, 25)]

    params1 = Parameters()
    params2 = Parameters()

    init_params(20, params1)
    init_params(19, params2)

    tokens1 = Token[init_data_token("$i",i,params1) for i=1:20]
    tokens2 = Token[init_data_token("$i",i,params2) for i=1:19]

    s_bi_1 = Token[tokens1[rand(1:20)] for i=1:15]
    s_mono_outer_1 = Token[tokens1[rand(1:20)] for i=1:11]
    s_mono_inner_1 = s_mono_outer_1[3:8]
    s_bi_2 = Token[tokens2[rand(1:19)] for i=1:17]
    s_mono_outer_2 = Token[tokens2[rand(1:19)] for i=1:13]
    s_mono_inner_2 = s_mono_outer_2[3:8]

    get_rand(range) = 5

    function set_copy!(a, b)
        @assert length(a) == length(b)
        @simd for i=1:length(a)
            b[i] = a[i]
        end
    end

    # Accumulate gradients (the analytical way)
    accumulate_gradients_batch!(model, s_mono_outer_1, 
                        s_mono_inner_1, Vector{Token}[s_bi_1], intermediates, 
                            scaling=1.0)


    function joggle_words(wordvecs)
        reshape(wordvecs, size(params1.wordvecs))
        tmp = deepcopy(params1.wordvecs)
        set_copy!(wordvecs, params1.wordvecs)
        err = objective_gradients_batch(model, s_mono_outer_1, 
                        s_mono_inner_1, Vector{Token}[s_bi_1], intermediates, 
                            scaling=1.0)
        set_copy!(tmp, params1.wordvecs)
        return err
    end

    function calc_fd_words()
        return Calculus.finite_difference(joggle_words,
                    reshape(deepcopy(params1.wordvecs),length(params1.wordvecs)));
    end

    ana_words = reshape(params1.gradients,length(params1.gradients))
    fd_words = calc_fd_words()
    fd_words_shaped = reshape(fd_words, size(params1.gradients))
    @test_approx_eq_eps ana_words fd_words 1e-6
end

println("All tests were successful")
