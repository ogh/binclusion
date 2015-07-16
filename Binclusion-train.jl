# vim:set ft=julia ts=4 sw=4 sts=4 autoindent:

include("LoadEnvironment.jl")

require("Corpora.jl")
require("CVMAbstract.jl")
require("CVMAddition.jl")
require("CVMVariance.jl")
require("CVMMean.jl")
require("CVMBigramAddition.jl")
require("CVMBigramMean.jl")
require("CommonTools.jl")
require("DataTypes.jl")
require("Optimizers.jl")

# using ProfileView
using DataTypes
using ArrayViews
using CVMAbstractModel
using Addition
using Mean
using BigramAddition
using BigramMean
using Variance
using CommonTools.CVMIntermediates
using Optimizers
using CommonTools.@nogc


using Corpora: Token, Sentence
using Corpora: DummyToken, DataToken
using Corpora: text, read_linecorpus, replace_vocab!
using JSON
using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "path_l1"
            help = "Path for l1 training data (sentence aligned to l2!!)"
            required = true
        "path_l2"
            help = "Path for l2 training data (sentence aligned to l1!!)"
            required = true
        "path_l1_mono"
            help = "Path for monolingual l1 training data"
            required = true
        "path_l2_mono"
            help = "Path for monolingual l2 training data"
            required = true
        "path_output"
            help = "Path to store the model and representations after training."
            required = true
        "--pre_trained_l1"
            help = "Path to pre-trained word representations for language 1."
            arg_type = String
            default = ""
        "--pre_trained_l2"
            help = "Path to pre-trained word representations for language 2."
            arg_type = String
            default = ""
        "--modeltype"
            help = "Type of model to train. (composition function)"
            arg_type = String
            default = "Addition"
        "--optimizer"
            help = "Optimizer: Choose from 'AdaGrad', 'RMSProp', 'AdaDelta'"
            arg_type = String
            default = "AdaGrad"
        "--wordvecdim"
            help = "Dimensionality of the word vectors"
            arg_type = Int
            default = 40
        "--margin"
            help = "Hinge Loss margin"
            arg_type = Float
            default = float(40.0)
        "--nnoise"
            help = "Number of noise samples"
            arg_type = Int
            default = 1
        "--batchsize"
            help = "Size of batches over which the gradient will be accumulated"
            arg_type = Int
            default = 40000
        "--regularizer"
            help = "L2 regularizer influence"
            arg_type = Float
            default = float(0.2)
        "--gamma"
            help = "Influence of the noise contrastive objective (in contrast to the 'keep it close' objective"
            arg_type = Float
            default = float(1.0)
        "--bi_closeness_scaling"
            help = "Influence of closeness objective"
            arg_type = Float
            default = float(1.0)
        "--initial_lr"
            help = "Initial learning rate (will be decreased during training)"
            arg_type = Float
            default = float(0.2)
        "--minimal_lr"
            help = "Minimal learning rate"
            arg_type = Float
            default = float(0.2)
        "--rms_decay"
            help = "Decay for RMSProp or AdaDelta(between 0.0 and 1.0)"
            arg_type = Float
            default = float(0.99)
        "--niter"
            help = "Number of iterations"
            arg_type = Int
            default = 100
        "--limit_lines_mono"
            help = "Only read in the specified number of lines in the monolingual case"
            arg_type = Int
            default = 0
        "--unk"
            help = "Unk token"
            default = "_UNK_"
        "--skipunk"
            help = "Skip unknown tokens instead of replacing them by the unk"
            action = :store_true
        "--sparse_updates"
            help = "Only add regularization to gradients that are non-zero"
            action = :store_true
        "--bilingual_contrastive"
            help = "Instead of bilingual closeness, apply the contrastive criterion as used in Hermann et al. (2014)"
            action = :store_true
        "--debug"
            help = "Turn on debug mode (working with less data)"
            action = :store_true
    end

    return parse_args(s)
end


function dump_to_file(model, vocab1, vocab2, path, args)
    """
        Dumps the given model to disk
    """

    # Save model
    open(joinpath(path,"model"),"w") do fout
        d = JSON.parse(JSON.json(model))
        d["modeltype"] = args["modeltype"]
        d["path_l1"] = args["path_l1"]
        d["path_l2"] = args["path_l2"]
        d["path_l1_mono"] = args["path_l1_mono"]
        d["path_l2_mono"] = args["path_l2_mono"]
        write(fout,JSON.json(d))
    end
    # Save vocabulary l1
    open(joinpath(path,"wordvecs.l1"),"w") do fout
        println(fout , model.wordvecdim, " ", length(vocab1))
        for tok in sort(collect(values(vocab1)), by=x->x.count[1], rev=true)
            string_vec = join(map(x->(@sprintf "%0.8f" x),tok.vector), " ")
            println(fout ,tok.word, " ", string_vec)
        end
    end
    # Save vocabulary l2
    open(joinpath(path,"wordvecs.l2"),"w") do fout
        println(fout , model.wordvecdim, " ", length(vocab2))
        for tok in sort(collect(values(vocab2)), by=x->x.count[1], rev=true)
            string_vec = join(map(x->(@sprintf "%0.8f" x),tok.vector), " ")
            println(fout ,tok.word, " ", string_vec)
        end
    end
end


function load_wordvecs(path_wordvecs::String)
    """
        Load pre-trained word vectors from disk
    """
    d = Dict{String,Token}()
    open(path_wordvecs) do fin
        dim, nvecs = map(int64, split(strip(readline(fin))," "))
        for line in eachline(fin)
            s = split(strip(line)," ")
            d[s[1]] = Token(s[1], map(float,s[2:end]),
                                ones(dim), [0])
        end
    end
    return d;
end

function main()
    # Parse Command line
    args = parse_commandline()
    

    # Create folder and log file
    mkpath(args["path_output"])
    logfile = open(joinpath(args["path_output"],"progress.log"),"w")
    
    # Dump arguments
    open(joinpath(args["path_output"],"args"),"w") do fout
        write(fout,JSON.json(args))
    end

    # Save the current state of the code together with the model
    dump_code_path = abspath(joinpath(args["path_output"],"code"))
    mkpath(dump_code_path)
    current_dir = splitdir(@__FILE__)[1]
    # Note: This will only work on systems that have find installed
    # (=> not Windows)
    run(`find $current_dir -maxdepth 1 -name "*\.jl"` |> 
                `xargs -i cp {} $dump_code_path`)


    # Initialize model according to the user's selection
    if args["modeltype"] == "BigramMean"
       global ModelModule = BigramMean
       typealias Model BigramMean.BigramMeanModel
       typealias Parameters BigramMean.BigramMeanParameters
    elseif args["modeltype"] == "BigramAddition"
       global ModelModule = BigramAddition
       typealias Model BigramAddition.BigramAdditionModel
       typealias Parameters BigramAddition.BigramAdditionParameters
    elseif args["modeltype"] == "Addition"
       global ModelModule = Addition
       typealias Model Addition.AdditionModel
       typealias Parameters Addition.AdditionParameters
    elseif args["modeltype"] == "Mean"
       global ModelModule = Mean
       typealias Model Mean.MeanModel
       typealias Parameters Mean.MeanParameters
    elseif args["modeltype"] == "Variance"
       global ModelModule = Variance
       typealias Model Variance.VarianceModel
       typealias Parameters Variance.VarianceParameters
    end

    ## Some statics
    # RMSProp/AdaDelta decay
    rms_decay = float(args["rms_decay"])
    @assert 0.0 <= rms_decay <= 1.0
    # Influence of bilingual closeness objective
    # The bigger, the more emphasis we put on the bilingual objective
    bi_closeness_scaling = args["bi_closeness_scaling"]
    # Only regularize parameters that have non-zero gradients
    sparse_updates = args["sparse_updates"]
    # Instead of applying bilingual closeness, use the 
    # bilingual noise contrastive criterion in Hermann et al. (2014)
    bilingual_contrastive = args["bilingual_contrastive"]
    # Debug mode with smaller number of samples
    line_limit = args["debug"] ? 10000 : 0
    line_limit_mono = args["debug"] ? 10000 : 0
    if args["limit_lines_mono"] > 0
        line_limit_mono = args["limit_lines_mono"]
    end

    
    # Initialize model
    model = Model(args["wordvecdim"], # word vector dimensionality
                    args["margin"], # Margin
                    args["regularizer"], #l2 regularizer influence
                    args["initial_lr"], # initial learning rate
                    args["minimal_lr"], # minimal learning rate
                    args["niter"], # number of iterations
                    args["gamma"], # weighting of the hinge objective
                    args["nnoise"], # number of noise samples
                    args["batchsize"]) # size of gradient batches
                    
    # If the user has specified a path to pre-trained vectors, load them
    vocab_l1 = Dict{String,Token}()
    if !isempty(args["pre_trained_l1"])
        vocab_l1 = load_wordvecs(args["pre_trained_l1"])
    end
    vocab_l2 = Dict{String,Token}()
    if !isempty(args["pre_trained_l2"])
        vocab_l2 = load_wordvecs(args["pre_trained_l2"])
    end
    
    # Initializes a large parameter array for the given vocabulary size
    function init_params(nvocab::Int64, params::AbstractParameters)
        params.wordvecs = randn(model.wordvecdim, nvocab) * 0.1
        params.gradients = zeros(model.wordvecdim, nvocab)
    end
   
    # Creates a DataToken from a given DataToken
    # Usually needed when we load pre-trained word vectors
    function init_from_data(iword::Int64, tok::DataToken, 
                                        params::AbstractParameters)
        return DataToken(tok.word, iword, tok.count, tok.vector, tok.gradient)
    end
   
    ## tricking Julia here a bit
    # Create a vector that doesn't hold a copy but only a reference
    # to the big parameter array
    # This way, we have everything nicely encapsulated in the
    # Token struct and don't need to fiddle with indices anymore
    function vec_from_pointer(p::Ptr{Float}, iword, width)
        byteshift = sizeof(Float) * (iword-1) * width
        return pointer_to_array(p+byteshift, (width,))
    end
    
    # Initializes a data token from a dummy token
    # Dummy tokens are used a placeholders when we load all the data.
    # After loading the whole data we know the vocabulary sizes and 
    # can initialize the large arrays holding all the parameters.
    # Once that is done, we replace DummyTokens by DataTokens which 
    # then hold references to the actual word vectors
    function init_from_dummy(iword::Int64, tok::DummyToken,
                                            params::AbstractParameters)
        return DataToken(tok.word, iword, tok.count, 
                            vec_from_pointer(pointer(params.wordvecs), 
                                                    iword, model.wordvecdim),
                            vec_from_pointer(pointer(params.gradients), 
                                                    iword, model.wordvecdim))
    end
    
    ##### Load language 1 part of the bilingual corpus    
    params_l1 = Parameters()
    (data_l1, vocab_l1_bi, 
        ntokens_l1, max_len_l1) = read_linecorpus(args["path_l1"], 
                                    (x, y) -> init_from_data(x,y,params_l1), 
                                    (x, y) -> init_from_dummy(x,y,params_l1),
                                    x -> init_params(x, params_l1),
                                    limit_lines=line_limit, 
                                    skiprows=0, dummy_vocab=vocab_l1,
                                    dummies_only=true);
    
    ##### Load language 2 part of the bilingual corpus    
    params_l2 = Parameters()
    (data_l2, vocab_l2_bi, 
        ntokens_l2, max_len_l2) = read_linecorpus(args["path_l2"], 
                                    (x, y) -> init_from_data(x,y,params_l2), 
                                    (x, y) -> init_from_dummy(x,y,params_l2),
                                    x -> init_params(x, params_l2),
                                    limit_lines=line_limit, 
                                    skiprows=0, dummy_vocab=vocab_l2,
                                    dummies_only=true);

    println(logfile, "[Only Bilingual]: [L1] Vocab size: $(length(vocab_l1))")
    println(logfile, "[Only Bilingual]: [L2] Vocab size: $(length(vocab_l2))")

    ##### Load monolingual data for language 1
    tic()
    (data_mono_l1, vocab_l1, 
        ntokens_mono_l1, max_len_mono_l1) = read_linecorpus(args["path_l1_mono"], 
                                    (x, y) -> init_from_data(x,y,params_l1), 
                                    (x, y) -> init_from_dummy(x,y,params_l1),
                                    x -> init_params(x, params_l1),
                                    limit_lines=line_limit_mono,
                                    dummy_vocab=vocab_l1_bi,
                                    skiprows=0);                                   
                                    
    ##### Load monolingual data for language 2
    (data_mono_l2, vocab_l2, 
        ntokens_mono_l2, max_len_mono_l2) = read_linecorpus(args["path_l2_mono"], 
                                    (x, y) -> init_from_data(x,y,params_l2), 
                                    (x, y) -> init_from_dummy(x,y,params_l2),
                                    x -> init_params(x, params_l2),
                                    limit_lines=line_limit_mono, 
                                    dummy_vocab=vocab_l2_bi,
                                    skiprows=0);
    println("Time to read corpus: $(toq()) seconds")
    tic()
    ##### Replace the dummy tokens by real data tokens
    replace_vocab!(data_l1, vocab_l1)
    replace_vocab!(data_l2, vocab_l2)
    println("Time to replace vocab: $(toq()) seconds")


    # Check if the data from both languages has the same size
    # If it doesn't, it can't be sentence aligned
    @assert size(data_l1) == size(data_l2)
    nsentences = size(data_l1,1)
    nsentences_mono_l1 = size(data_mono_l1,1)
    nsentences_mono_l2 = size(data_mono_l2,1)

    # Vocabulary sizes
    n_vocab_l1 = length(vocab_l1)
    n_vocab_l2 = length(vocab_l2)

    # Vocabulary lists
    words_l1 = collect(values(vocab_l1))
    words_l2 = collect(values(vocab_l2))

    # Create Optimizer for l1 and l2
    (opt_l1, opt_l2) = if args["optimizer"] == "AdaGrad"
        (AdaGrad(params_l1.wordvecs, params_l1.gradients),
         AdaGrad(params_l2.wordvecs, params_l2.gradients))
    elseif args["modeltype"] == "RMSProp"
        (RMSProp(params_l1.wordvecs, params_l1.gradients, args["rms_decay"]),
         RMSProp(params_l2.wordvecs, params_l2.gradients, args["rms_decay"]))
    elseif args["modeltype"] == "AdaDelta"
        (AdaDelta(params_l1.wordvecs, params_l1.gradients, args["rms_decay"]),
         AdaDelta(params_l2.wordvecs, params_l2.gradients, args["rms_decay"]))
    end


    # Initialize objectes for intermedate values
    # => since garbage collection is slow, we don't want to re-create them
    # for every run of the inner loop.
    # These structures simply store intermediate values (forward pass results).
    # CAREFUL: In case of implementing parallelism later these intermediate
    # values need to be taken into account. Each thread/process needs its own
    # set of intermediates.
    intermediates = CVMIntermediates[CVMIntermediates(model.wordvecdim,
                                 max(max_len_l1, max_len_l2,
                                     max_len_mono_l1, max_len_mono_l2)),
                     CVMIntermediates(model.wordvecdim,
                                 max(max_len_l1, max_len_l2,
                                     max_len_mono_l1, max_len_mono_l2)),
                     CVMIntermediates(model.wordvecdim,
                                 max(max_len_l1, max_len_l2,
                                    max_len_mono_l1, max_len_mono_l2))]
                                 
    println(logfile, "Loading Data completed")
    println(logfile, "Bilingual")
    println(logfile, "\tL1")
    println(logfile, "\t\tNumber of sentences: $(size(data_l1,1))")
    println(logfile, "\t\tNumber of tokens: $(ntokens_l1)")
    println(logfile, "\t\tVocabulary size: $(length(vocab_l1))")
    println(logfile, "\tL2")
    println(logfile, "\t\tNumber of sentences: $(size(data_l2,1))")
    println(logfile, "\t\tNumber of tokens: $(ntokens_l2)")
    println(logfile, "\t\tVocabulary size: $(length(vocab_l2))")
    println(logfile, "Monolingual")
    println(logfile, "\tL1")
    println(logfile, "\t\tNumber of sentences: $(size(data_mono_l1,1))")
    println(logfile, "\t\tNumber of tokens: $(ntokens_mono_l1)")
    println(logfile, "\tL2")
    println(logfile, "\t\tNumber of sentences: $(size(data_mono_l2,1))")
    println(logfile, "\t\tNumber of tokens: $(ntokens_mono_l2)")
    
    #########################################################################
    ######################## Training #######################################
    #########################################################################
    
    
    println(logfile, "##############")
    println(logfile, "Starting training")
    # Outer and inner windows of the inclusion criterion will always be between
    # the specified boundaries
    minmaxwindow_outer = (10, 50);
    minwindow_inner = 3;
    minmaxwindown = (10, 50);
    ### Perform the training
    # The total number of samples that we process is calculated
    # as the number of sentences in the largest data collection (mono_l1,
    # mono_l2 or bi) times the number of iterations specified by the user
    max_nsentences = max(nsentences, nsentences_mono_l1, nsentences_mono_l2)
    nsentences_total = max_nsentences * model.niter
    isentence_total = 0
    # Some varibles to store accumulated errors
    acc_total = const_zero;
    acc_closeness_bi = const_zero;
    acc_contr_mono_l1 = const_zero;
    acc_closeness_mono_l1 = const_zero;
    acc_contr_mono_l2 = const_zero;
    acc_closeness_mono_l2 = const_zero;
    acc_reg = const_zero;
    isentence_tic = const_zero;
    tic();
    # Learning rate (initially) (will be decreased according to specified
    # parameters)
    lr = model.lr_init;

    ################## START THE TRAINING #######################
    println("Starting training")
    #= Profile.clear() =#
    while isentence_total < nsentences_total
        # Since we always iterate over a full batch, 
        # we might end up training for slightly more than the specified
        # iterations.
        # Turn off the garbage collection while processing a batch
        @nogc for ibatch=1:model.batchsize
            # Calculate the index of the current bilingual sentence pair
            isentence = (isentence_total % nsentences) + 1
            # Adaptive learning rate
            lr = float(max(model.lr_min, 
                        model.lr_init * 
                            (const_one - isentence_total / nsentences_total)))
                            
            # Select the text of the current sentence and its translation
            s1 = data_l1[isentence].text
            s2 = data_l2[isentence].text
            # Make sure all sentences have a certain minimal length
            if (size(s1,1) < 5 || size(s2,1) < 5)
                isentence_total += 1
                continue 
            end
            
            # Collect the gradients for the bilingual criterion.
            # All the gradients are stored in the corresponding field
            # of each token. This field is a reference to the large gradient
            # array the contains all gradients.
            if bilingual_contrastive
                sn = data_l2[rand(1:nsentences)].text
                err_closeness_bi, err_contr_bi = accumulate_gradients_batch!(
                                                model, 
                                                s1, s2, Vector{Token}[sn], 
                                                intermediates,
                                                scaling=bi_closeness_scaling)
                acc_closeness_bi += err_closeness_bi + err_contr_bi
            else
                acc_closeness_bi += accumulate_closeness_gradients_batch!(model, 
                                                s1, s2, 
                                                intermediates,
                                                scaling=bi_closeness_scaling)
            end


            #### Monolingual
            for (data, ns, lbl) in ((data_mono_l1, nsentences_mono_l1, 1), 
                            (data_mono_l2, nsentences_mono_l2, 2))
                # Calculate the running index for the current
                # sentence in the current corpus
                icursentence = (isentence_total%ns)+1
                # Get the sentence length and check if it 
                # is longer than the specified minimum
                lensen = size(data[icursentence].text,1)
                if lensen < minmaxwindow_outer[1]+2
                    continue 
                end



                ### Outer window
                # Calculate the offset for the valid sentence
                offset_outer = rand(1:lensen-1-minmaxwindow_outer[1])
                # Get the end of the window
                # |-----|...............|.........|------------------|
                #    offset         min_winend   max_winend    sentence_end
                min_winend_outer = offset_outer+minmaxwindow_outer[1]
                window_end_outer = rand(min_winend_outer:
                                            min(min_winend_outer + 
                                                minmaxwindow_outer[2],
                                                lensen))
                ### Inner window
                # Length of the sub sentence
                lensubsen = window_end_outer - offset_outer
                # Get the offset for the inner window
                offset_inner = rand(1:lensubsen-1-minwindow_inner)
                # Earliest end of inner window
                min_winend_inner = offset_inner+minwindow_inner
                # Get a random inner window end that is bigger than the minsize
                window_end_inner = rand(min_winend_inner:lensubsen)
                

                ### Noise (same procedure as for outer window)
                # Calculate the offset for the valid sentence
                noise_samples = Vector{Token}[]
                while size(noise_samples,1) < model.nnoise
                    # Get the index for the noise sentence
                    inoise = icursentence
                    # Make sure the noise sentence is not equivalent
                    # to the current valid sentence
                    while inoise == icursentence
                        inoise = rand(1:ns)
                    end
                    # Check if the length of the noise sentence
                    # is above the specified threshold
                    lensen_noise = size(data[inoise].text,1)
                    if lensen_noise < minmaxwindown[1]+2
                        continue
                    end

                    offsetn = rand(1:lensen_noise-1-minmaxwindown[1])
                    min_winendn = offsetn+minmaxwindown[1]
                    window_endn = rand(min_winendn:
                                                min(min_winendn + 
                                                    minmaxwindown[2],
                                                    lensen_noise))
                    sn = data[inoise].text[offsetn:window_endn]
                    push!(noise_samples,sn)
                end
                
                ### Retrieve the sentences with the previously calculated indices
                s1 = data[icursentence].text[offset_outer:window_end_outer]
                s2 = s1[offset_inner:window_end_inner]
                # Calculate the overlap ratio of the inner and outer window
                overlap_ratio = size(s2,1)/size(s1,1);
                # Accumulate the gradients for the inclusion objective
                err_closeness, err_contr = accumulate_gradients_batch!(model, 
                                                s1, s2, noise_samples, 
                                                intermediates,
                                                scaling=overlap_ratio)
                # Update the accumulative error
                if lbl == 1
                    acc_contr_mono_l1 += err_contr
                    acc_closeness_mono_l1 += err_closeness
                elseif lbl == 2
                    acc_contr_mono_l2 += err_contr
                    acc_closeness_mono_l2 += err_closeness
                else
                    throw(ErrorException("Unknown label given for acc errors."))
                end
            end

            # Increase the sentence counter
            isentence_total += 1
        end
        
        # Apply the squared l2 regularizer
        acc_reg += regularize_l2!(model, params_l1, 
                                        sparse_updates=sparse_updates)
        acc_reg += regularize_l2!(model, params_l2, 
                                        sparse_updates=sparse_updates)
        # Execute one update step of the optimizers
        # The optimizer also resets all the gradients accumulated in the
        # last batch to zero
        step!(opt_l1, lr)
        step!(opt_l2, lr)
        
        # Calculate the total accumulated error
        acc_total = (acc_closeness_bi + acc_reg + acc_contr_mono_l1 + 
                        acc_closeness_mono_l1 + acc_contr_mono_l2 + 
                        acc_closeness_mono_l2)
        
        # Print some progress statistics                        
        time_elapsed = toq();
        @printf logfile "Progress: %0.2f%% " (isentence_total)/nsentences_total*100
        @printf logfile "lr: %0.5f " lr
        @printf logfile "error: total %0.1f " (acc_total/model.batchsize) 
        @printf logfile "closeness %0.3f " (acc_closeness_bi/model.batchsize) 
        @printf logfile "cl-mono-l1 %0.3f " (acc_closeness_mono_l1/model.batchsize) 
        @printf logfile "co-mono-l1 %0.3f " (acc_contr_mono_l1/model.batchsize/model.nnoise) 
        @printf logfile "cl-mono-l2 %0.3f " (acc_closeness_mono_l2/model.batchsize) 
        @printf logfile "co-mono-l2 %0.3f " (acc_contr_mono_l2/model.batchsize/model.nnoise) 
        @printf logfile "reg %0.3f " (acc_reg/model.batchsize) 
        @printf logfile "sents/s: %6d \n" ((isentence_total-isentence_tic)/time_elapsed)
        flush(logfile)
        isentence_tic = isentence_total
        # Reset the accumulative errors 
        acc_total = const_zero;
        acc_contr_mono_l1 = const_zero;
        acc_closeness_mono_l1 = const_zero;
        acc_contr_mono_l2 = const_zero;
        acc_closeness_mono_l2 = const_zero;
        acc_closeness_bi = const_zero;
        acc_reg = const_zero;
        # Rest the timer that we use to estimate number-sentences/sec
        tic()
    end
    
    ##################### TRAINING FINISHED #####################
    
    # Create the output path if it doesn't exist
    dump_to_file(model, vocab_l1, vocab_l2, args["path_output"],
                    args);
    
   
    # Close the log file
    close(logfile)

end

main()
