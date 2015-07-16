module Corpora

require("LoadEnvironment.jl")
require("DataTypes.jl")

## Using
using ArrayViews
using DataTypes

## Exports
export Token, DataToken, DummyToken
export Sentence
export read_linecorpus
export text
export get_context, initialize_contexts
export replace_vocab!

abstract Token;

immutable DummyToken <: Token
    word::String;
    count::Vector{CInt};
end

DummyToken(word::String) = DummyToken(word, [0])

immutable DataToken <: Token
    word;
    idx::Int64;
    count::Vector{Int64};
    vector::DenseArray{Float,1}
    gradient::DenseArray{Float,1}
end

immutable Sentence
    text::Vector{Token} 
end
Sentence() = Sentence(Token[])

text(s::Vector{Token}) = join(map(x->x.word, s), " ")
text(sentence::Sentence) = text(sentence.text)


function readuntil_ws_eol(s::IO)
    out = IOBuffer()
    lineend = false
    firstchar = true
    while !eof(s)
        c = read(s, Char)
        if c == ' ' || c == '\n'
            if !firstchar
                if c == '\n'
                    lineend = true 
                end
                break
            end
        else
            firstchar = false
            write(out, c)
        end
    end
    return (lineend,takebuf_string(out))
end


function read_linecorpus(path::String, init_from_data, init_from_dummy,
                            init_params; 
                            dummy_vocab=Dict{String,Token}(), unk="_UNK_",
                            limit_lines=0, skiprows=0, skipunk=true, 
                            no_new_vocab=false, lock_tokens=false,
                            dummies_only=false)
    # Estimate number of lines
    lcounter = 0
    open(path) do fin
        for (iline, line) in enumerate(eachline(fin))
            if iline < skiprows
                continue
            end
            lcounter += 1
            if limit_lines > 0 && lcounter >= limit_lines
                break
            end
        end
    end
    data = Array(Sentence, lcounter)
    n_words_total = 0
    lcounter = 1
    max_lensen = -1
    open(path) do fin
        iline = 1
        sentence = Sentence()
        data[lcounter] = sentence
        lensen = 0
        while !eof(fin)
            (eol, token) = readuntil_ws_eol(fin)

            if iline < skiprows
                continue
            end
            
            ## For every token in the text
            # Check if it is already in the vocabulary
            # If not: add it with count 1
            # If it is: increase the counters
            if !haskey(dummy_vocab, token)
                if !no_new_vocab
                    #vocab[token] = inittoken(token, lock_tokens);
                    dummy_vocab[token] = DummyToken(token);
                else
                    if skipunk
                        n_words_total += 1
                        continue
                    else
                        token = unk
                    end
                end
            end
            dummytok = dummy_vocab[token]
            push!(sentence.text,dummytok)
            lensen += 1
            dummytok.count[1] += 1
            n_words_total += 1
            
            
            if eol
                if lensen > max_lensen
                    max_lensen = lensen
                end
            end


            if eol
                if (limit_lines > 0 && lcounter >= limit_lines)
                    break
                end
                if eof(fin)
                    break
                end
                lcounter += 1
                sentence = Sentence()
                data[lcounter] = sentence
                lensen = 0
            end


        end
    end
    
    
    vocab = if dummies_only
        dummy_vocab    
    else
        v = Dict{String,Token}()
        # Convert DummyTokens to real tokens
        ## Signal the model that the vocabulary has been acquired

        init_params(length(dummy_vocab))
        ## Initialize token types
        for (iword,(word, tok)) in enumerate(dummy_vocab)
            if typeof(tok) == DataToken
                v[word] = init_from_data(iword,tok)
            else
                v[word] = init_from_dummy(iword,tok)
            end
        end
        
        ## Replace the dummies by data tokens
        for isentence in 1:size(data,1)
            for itoken in 1:size(data[isentence].text,1)
                data[isentence].text[itoken] = v[data[isentence].text[itoken].word]
            end
        end
        v
    end

    return data, vocab, n_words_total, max_lensen;
end

function replace_vocab!(data::Vector{Sentence}, vocab::Dict{String,Token})
    for isentence in 1:size(data,1)
        for itoken in 1:size(data[isentence].text,1)
            data[isentence].text[itoken] = vocab[data[isentence].text[itoken].word]
        end
    end
end

function get_context(sentence::Vector{Token}, pos::Int64, windowsize::Int64)
    lensen = size(sentence,1)
    windowsize_half = int64(windowsize/2)
    window_start = max(1, pos - windowsize_half)
    window_end = min(lensen, pos + windowsize_half)
    window = Token[]
    for i=window_start:window_end
        if i != pos
            push!(window, sentence[i])
        end
    end
    return window
end

function initialize_contexts(data::Vector{Sentence}, windowsize::Int64,
                                minlensen=5)
    for sentence in data
        if size(sentence.text,1) < 5
            continue
        end
        for pos=1:size(sentence.text,1)
            empty!(sentence.text[pos].context)
            append!(sentence.text[pos].context,
                    get_context(sentence.text, pos, windowsize))
        end
    end
end

end
