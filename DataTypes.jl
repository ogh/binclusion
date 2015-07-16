module DataTypes

export Float, float
export CInt, cint
export CUint, cuint
export const_zero, const_one

# Define custom datatypes to make it easier to switch between 32 bit and 64
# bit.

typealias Float Float64
float = float64
typealias CInt Int64
cint = int64
typealias CUint Uint64
cuint = uint64

const const_zero = float(0.0)
const const_one = float(1.0)

end
