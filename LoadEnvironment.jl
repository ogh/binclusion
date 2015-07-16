import Base.source_path

using Base.Test

srcpath = source_path()
srcdir  = dirname(srcpath)

# Inject the main source directory into the load path
push!(LOAD_PATH, "$srcdir/") 
