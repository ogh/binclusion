# Binclusion #

This repository contains an implementation of the method proposed in
Soyer et al. (2015).

The code is written for the [Julia][juliaweb] programming language.
It is quite optimized, however, it currently can only run on a single core.
Training of the largest model mentioned in Soyer et al. (2015) 
(EuroFullReuters) takes about 6 hours on a modern Desktop computer where
loading the data constitutes a big chunk of this time.

## Instructions ##

In order to run the code you will need to install julia.
For instructions on how to install julia please refer to 
[the julia website][juliaweb].
Note that julia is a young language which changes rapidly.
You should be able to run this code with julia 0.3.x.x.
Even though this code was developed using an early build of
julie 0.4.0 it possible that julia that it will not run 
with the release version of 0.4.0 since it introduces several
breaking changes.

After installing julia you have to install several packages

    - ArrayViews
    - JSON
    - ArgParse
    - Calculus
    
You can install packages in julia by opening the interactive console
(type `julia` in bash) and using the package manager with

    Pkg.add("<package name>")

The environment that you need for running the code should now be set up.
To get help on the parameters that you can specify run:

    julia Binclusion-train.jl --help
    
To run the program with default parameters for 30 iterations:

    julia Binclusion-train.jl \
       --niter 30 \
       parallel_corpus_en \
       parallel_corpus_de \
       monolingual_en \
       monolingual_de \
       resultfolder &
    
where `parallel_corpus_en` and `parallel_corpus_de` are files that contain one
sentence in each line and the sentences are parallel. This means that
line 1 in `parallel_corpus_en` is the translation of line 1 in
`parallel_corpus_de`.
`monolingual_en` and `monolingual_de` should also contain one sentence per line
but the text does not have to be parallel, the files can even have different
numbers of lines.

The program creates a copy of its source files for archival purposes and
stores it together with the model description and word representation files.
This copy uses the unix `find` tool and will therefore most likely only
run on unix systems. You can just comment out that part of the code if you
are on windows, it is not essential.

## Citing ##

If you make use of this code for scientific research, please cite
the following publication:

    @inproceedings{soyer2014leveraging,
        title={Leveraging Monolingual Data for Crosslingual Compositional Word Representations},
        author={Soyer, Hubert and Stenetorp, Pontus and Aizawa, Akiko},
        booktitle={International Conference on Learning Representations},
        year={2015}
    }
    
    
[juliaweb]: http://julialang.org/
