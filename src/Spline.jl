
"""
    test_are_numbers(y...)

Return `true` if the inputs are all valid numbers, `false` otherwise (i.e. there
is at least one `NaN` or `Inf` or `-Inf`).
"""
test_are_numbers(y...) = any(x -> isnan(x) || isinf(x), [y...]) ? false : true



"""
    struct MySpline
        xs::Vector{Float64}
        coeffs::Vector{Tuple{Float64,Float64,Float64,Float64}}
        N::Int64
    end

Struct that stores the data necessary to define a cubic spline interpolation.

The implementation of this function is based on:
- Parviz Moin, _"Fundamentals of Engineering Numerical Analysis"_ (2010), 
  Cambridge University Press: Second edition, Chapter 1.2, "Cubic Spline Interpolation"
- 

Have a look at the **Spline Theory** section in the online documentation of GaPSE.jl 
([https://foglienimatteo.github.io/GaPSE.jl/stable/](https://foglienimatteo.github.io/GaPSE.jl/stable/))
to have a better understanding of the mathematical procedure here employed.

## Arguments

- `xs::Vector{Float64}`: input x-axis data points.
- `coeffs::Vector{Tuple{Float64,Float64,Float64,Float64}}`: vector of tuples containing 
   `(y, B, C, D)`, respectively the input y-axis data points and the linear, quadriatic and
  cubic coefficients of the polynomial, i.e.:
  ```math
    f(x) = y_i + B \\,(x - x_i) + C \\,(x - x_i)^2 + D \\,(x-x_i)^3 \\; , \\quad 
    \\forall x \\in [x_i, x_{i+1}] \\; , \\quad \\forall i = 1,...,N-1 
  ```
- `N::Int64`: number of input data point couples ``(x_i, y_i)``

## Constructor

    MySpline(
        xs::Vector{T1}, ys::Vector{T2}; 
        bc::String="Error", ic::String="Natural"
    ) where {T1<:AbstractFloat, T2<:AbstractFloat}
        
It supports specifying Boundary Conditions (BC) and Initial Conditions (ic) for the 
spline interpolation.

- `xs::Vector{T1}`: input x-axis data points.
- `xs::Vector{T2}`: input y-axis data points.
- `bc::String="Error"`: Boundary Condition option, i.e. what to do if the point 
  where to evaluate the spline is outside the range ``[x_1, x_N]``. 
  Only "Error" or "error" are allowed.
- `ic::String="Natural"`: Initial Condition option. 
  Supported options are 
  - `"Natural"` : also known as Free Run-Out, the splines gets flat at the extremes;
  - `"Parabolic"`: Parabolic run-out, i.e. the edge cubic are set to be parabolas;
  - `"ThirdDerivative"` : set the continuity of the Third Derivative at the extremes.

## Example of use

Once you create a `MyStruct` spline, you can apply it as a normal function to
any input number/vector:

```julia
julia> xs = [1.0*x for x in 1:10]; # MySpline doesn't accept Int64

julia> ys = [2.0*x for x in 1:10]; 

julia> spl = GaPSE.MySpline(xs, ys; ic="Natural");

julia> spl(3)
6.0

julia> spl.([4 5])
1×2 Matrix{Float64}:
 8.0  10.0

julia> spl.([4.3, 5.1])
2-element Vector{Float64}:
  8.6
 10.2
```

See also: [`derivative`](@ref)
"""
struct MySpline
    xs::Vector{Float64}
    coeffs::Vector{Tuple{Float64,Float64,Float64,Float64}}
    N::Int64

    function MySpline(xs::Vector{T1}, ys::Vector{T2}; bc::String="Error", ic::String="Natural") where {T1<:AbstractFloat,T2<:AbstractFloat}
        @assert length(xs) == length(ys) "xs and ys have different lengths!"
        @assert length(xs) > 3 "cannot interpolate with less than 4 pairs of data"

        # Boundary conditions
        @assert bc == "Error" || bc=="error" "Only bc=\"Error\"/\"error\" is allowed at the moment, bc=$bc is not valid."


        N, Δ = length(xs), diff(xs)
        γ, δ = similar(Δ), similar(xs)
        B, C, D = similar(xs), similar(xs), similar(xs)
        # B, C and D should be "similar(Δ)", but we want to use C[i] = g''(x[i]) to compute B and D, 
        # and their last terms B[N] & D[N] require g''(x[N]), that would not be included in C if
        # it would span from 1 to N-1; we will cut them at the end

        # Initial conditions
        b_1, b_N, c_1, a_N, d_1, d_N = begin
                tmp = if ic == "Natural"
                        1, 1, 0, 0, 0, 0
                    elseif ic == "Parabolic"
                        1, 1, -1, -1, 0, 0
                    elseif ic == "ThirdDerivative"
                        η_in = [(ys[i+1] - ys[i]) / (xs[i+1] - xs[i]) for i in 1:3]
                        η_in_2 = [(η_in[i+1] - η_in[i]) / (xs[i+2] - xs[i]) for i in 1:2]
                        η_in_3 = (η_in_2[2] - η_in_2[1]) / (xs[4] - xs[1])

                        η_end = [(ys[N-3+i] - ys[N-4+i]) / (xs[N-3+i] - xs[N-4+i]) for i in 1:3]
                        η_end_2 = [(η_end[i+1] - η_end[i]) / (xs[N-2+i] - xs[N-4+i]) for i in 1:2]
                        η_end_3 = (η_end_2[2] - η_end_2[1]) / (xs[N] - xs[N-3])

                        -Δ[1] / 6, -Δ[N-1] / 6, Δ[1] / 6, Δ[N-1] / 6, Δ[1]^2 * η_in_3, -Δ[N-1]^2 * η_end_3
                    else
                        vic = ["Natural", "Parabolic", "ThirdDerivative"]
                        throw(AssertionError("ic=$ic is not valid. Valid inital conditions are: $vic"))
                    end
                test_are_numbers(tmp...) ? tmp : (1, 1, 0, 0, 0, 0)
            end
        

        γ[1], δ[1] = c_1 / b_1, d_1 / b_1
        for i in 2:N-1
            ζ = (ys[i+1] - ys[i]) / Δ[i] - (ys[i] - ys[i-1]) / Δ[i-1]
            den = 2 * (Δ[i] + Δ[i-1]) - Δ[i-1] * γ[i-1]
            γ[i] = Δ[i] / den
            δ[i] = (6 * ζ - Δ[i-1] * δ[i-1]) / den
        end
        δ[N] = (d_N - a_N * δ[N-1]) / (b_N - a_N * γ[N-1])

        @assert ic != "Natural" || δ[N] ≈ 0 " δ[N] = $(δ[N]) !=0  with ic=$ic"

        #β[N] = b_N * β[N-1] - a_N * γ[N-1]
        #δ[N] = (d_N * β[N-1] - a_N * δ[N-1])/β[N]

        C[N] = δ[N] / 2
        for i in N-1:-1:1
            C[i] = (δ[i] - 2 * γ[i] * C[i+1]) / 2
            B[i] = (ys[i+1] - ys[i]) / Δ[i] - Δ[i] * (C[i+1] + 2 * C[i]) / 3
            D[i] = (C[i+1] - C[i]) / (3 * Δ[i])
        end
        
        B[N], C[N], D[N] = 0.0, 0.0, 0.0
        # We need the N index for ys[N] , so for compatibility we keep them all

        new(xs, [(ys[i], B[i], C[i], D[i]) for i in 1:N], N)
    end
end

"""
    (S::MySpline)(x)

Return the value of this cubic spline struct `MySpline` in the input point `x`,
using the equation:

```math
    f(x) = y_i + B \\,(x - x_i) + C \\,(x - x_i)^2 + D \\,(x-x_i)^3 \\; , \\quad 
    \\forall x \\in [x_i, x_{i+1}] \\; , \\quad \\forall i = 1,...,N-1 
```

where ``x_i`` is the biggest value stored in `S.xs` smaller than `x` and 
``y, ``B``, ``C``, ``D`` are its corresponding coefficients stored in `S.coeff[i]` 
(with `x_i = S.xs[i]`).

See also: [`MySpline`](@ref), [`derivative`](@ref)
"""
function (S::MySpline)(x)
    @assert S.xs[1] ≤ x ≤ S.xs[end] "BC Error: $(S.xs[1]) ≤ $x ≤ $(S.xs[end]) does not hold!"
    i = searchsortedlast(S.xs, x)
    #u = x - S.xs[i]
    u, a, b, c, d = (i == length(S.xs)) ? (x - S.xs[i-1], S.coeffs[i-1]...) : (x - S.xs[i], S.coeffs[i]...)
    (u ≈ 0.0) && (return S.coeffs[i][1])

    return a + b * u + c * u^2 + d * u^3
    #return @evalpoly(u, S.coeffs[i]...)
end

function derivative(S::MySpline, x::T; nu::Int=1) where {T<:Real}
    @assert nu > 0 "The derivative order nu must be >0; nu=$nu is not valid!"
    @assert S.xs[1] ≤ x ≤ S.xs[end] "BC Error: $(S.xs[1]) ≤ $x ≤ $(S.xs[end]) does not hold!"
    i = searchsortedlast(S.xs, x)
    u, _, b, c, d = (i == length(S.xs)) ? (x - S.xs[i-1], S.coeffs[i-1]...) : (x - S.xs[i], S.coeffs[i]...)

    if nu == 1
        return b + u * (2*c + 3*d*u)
    elseif nu == 2
        return 2*c + 6*d * u
    elseif nu == 3
        return 6*d
    else
        return 0.0
    end

    # The following general expression is way less efficient than the if-else statements
    #if nu > 3
    #    return 0
    #else
    #    coeffs = S.coeffs[i]
    #    return @evalpoly(x - x_i, coeffs[nu+1:end] .* [factorial(i) / factorial(i - nu) for i in nu:length(coeffs)-1]...)
    #end
end

function derivative(S::MySpline, x::Vector{T}; nu::Int=1) where {T<:Real}
    return [derivative(S, p; nu=nu) for p in x]
end


"""
    derivative(S::MySpline, x::T; nu::Int=1) where {T<:Real}
    derivative(S::MySpline, x::Vector{T}; nu::Int=1) where {T<:Real}

Evaluate the derivative of order `nu` in the input point(s) `x` for the cubic
spline object ``S::MySpline``.
Due to the fact that this spline is piecewise cubic, the following equations hold
``\\forall x \\in [x_i, x_{i+1}] \\; , \\quad \\forall i = 1,...,N-1``:
- ``\\nu=1``:
  ```math
    f^{'}(x) = B  + 2 \\, C \\,(x - x_i) + 3 \\, D \\,(x-x_i)^2 
  ```
- ``\\nu=2``:
  ```math
    f^{''}(x) = 2 \\, C + 6 \\, D \\,(x-x_i)
  ```
- ``\\nu=3``:
  ```math
    f^{'''}(x) = 6 \\, D
  ```
- ``\\nu \\geq 4``:
  ```math
    f^{(n)}(x) = 0 \\; , \\quad \\forall n \\geq 4
  ```
where ``x_i`` is the biggest value stored in `S.xs` smaller than `x` and 
``y, ``B``, ``C``, ``D`` are its corresponding coefficients stored in `S.coeff[i]` 
(with `x_i = S.xs[i]`).


See also: [`MySpline`](@ref)
"""
derivative
