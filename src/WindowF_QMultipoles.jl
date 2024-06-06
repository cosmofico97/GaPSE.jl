# -*- encoding: utf-8 -*-
#
# This file is part of GaPSE
# Copyright (C) 2022 Matteo Foglieni
#
# GaPSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GaPSE is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GaPSE. If not, see <http://www.gnu.org/licenses/>.
#


"""
    WindowFIntegrated_multipole(
            s, windowfint::GaPSE.WindowFIntegrated;
            s_min, s_max,
            L::Int=0, alg::Symbol=:lobatto,
            N_lob::Int=100, N_trap::Int=200,
            atol_quad::Float64=0.0, rtol_quad::Float64=1e-2,
            enhancer::Float64=1e6,
            )

Evaluate the multipole of order `L` of the input Integrated Window Function `windowfint` in the 
input comoving distance `s`. 
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of the Integrated Window Function multipole ``Q_{\\ell_1}`` is the following:

```math
Q_{\\ell_1}(s) = 
    \\int_{-1}^{1} \\mathrm{d}\\mu \\; \\mathcal{L}_{\\ell_1}(\\mu) 
    \\; \\mathcal{F}(s, \\mu)
```

where ``\\mathcal{L}_{\\ell_1}`` is the Legendre polynomial of order ``\\ell_1``, ``\\mu`` the 
cosine angle, 

```math
\\mathcal{F}(s, \\mu) = 
    \\int_0^\\infty \\mathrm{d}s_1 \\, \\phi(s_1) \\,  
    \\phi\\left(\\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}\\right) 
    \\, F\\left(\\frac{s}{s_1}, \\mu \\right)
```

the integrated window function associated to the window function ``F\\left(\\frac{s}{s_1}, \\mu \\right)``
(check the docstring of `WindowF` for its definition) and ``\\phi`` the radial window function, obtained by 
`ϕ`.


## Optional arguments

- `s_min` and `s_max` (mandatory keyword arguments) : min and max comoving distance of the survey;
  their values will be internally used by 

- `L::Int = 0`: order of the Legendre polynomial to be used

- `alg::Symbol = :lobatto` : algorithm to be used for the integration; the valid options 
  are (other values will lead to `AssertionError`):
  - `:quad` -> the integration over ``\\mu`` will be preformed through the Julia function `quadgk` 
  from the [`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl) Julia package, that uses an adaptive 
  Gauss-Kronrod quadrature.
  - `:trap` -> the integration over ``\\mu`` will be preformed through the Julia function `trapz` 
  from the [`Trapz.jl`](https://github.com/francescoalemanno/Trapz.jl) Julia package, that uses the
  simple trapezoidal rulae.
  - `:lobatto` -> the integration over ``\\mu`` will be preformed through the Julia function `gausslobatto` 
  from the [`FastGaussQuadrature.jl`](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) Julia package, 
  that uses the Gauss-Lobatto quadrature. 

- `N_lob::Int = 100` : number of points to be used in the sampling made by the function `trapz`.
  Note that these options will have an effect only if you se `alg = :quad`.

- `N_trap::Int = 200` : number of points to be used in the sampling made by the function `trapz`.
  Note that these options will have an effect only if you se `alg = :quad`.

- `atol_quad::Float64 = 0.0` and `rtol_quad::Float64 = 1e-2`: absolute and relative tolerance
  to be passed to the function `quadgk`; it's recommended not to set `rtol_quad < 1e-2` 
  because the time for evaluation increase quickly.
  Note that these options will have an effect only if you se `alg = :quad`.

- `enhancer::Float64 = 1e6`: just a float number used in order to deal better with small numbers; 
  the returned value is NOT modified by this value, because after a multiplication
  the internal result is divided by `enhancer`.

See also: [`WindowFIntegrated`](@ref), [`WindowF`](@ref), [`ϕ`](@ref)
"""
function WindowFIntegrated_multipole(
    s, windowfint::GaPSE.WindowFIntegrated;
    s_min, s_max,
    L::Int=0, alg::Symbol=:lobatto,
    N_lob::Int=100, N_trap::Int=200,
    atol_quad::Float64=0.0, rtol_quad::Float64=1e-2,
    enhancer::Float64=1e6)

    @assert alg ∈ GaPSE.VALID_INTEGRATION_ALGORITHM ":$alg is not a valid Symbol for \"alg\"; they are: \n\t" *
                                                    "$(":".*string.(VALID_INTEGRATION_ALGORITHM) .* vcat([" , " for i in 1:length(VALID_INTEGRATION_ALGORITHM)-1], " .")... )"

    @assert N_trap > 2 "N_trap must be >2,  N_trap = $N_trap is not!"
    @assert N_lob > 2 "N_lob must be >2,  N_lob = $N_lob is not!"
    @assert atol_quad ≥ 0.0 "atol_quad must be ≥ 0.0,  atol_quad = $atol_quad is not!"
    @assert rtol_quad ≥ 0.0 "rtol_trap must be ≥ 0.0,  rtol_quad = $rtol_quad is not!"
    @assert L ≥ 0 "L must be ≥ 0, L = $L is not!"


    orig_f(μ) = enhancer * GaPSE.spline_integrF(s, μ, windowfint) * Pl(μ, L)

    int = if alg == :lobatto
        xs, ws = gausslobatto(N_lob)
        dot(ws, orig_f.(xs))

    elseif alg == :quad
        quadgk(μ -> orig_f(μ), -1.0, 1.0; atol=atol_quad, rtol=rtol_quad)[1]

    elseif alg == :trap
        μs = union(
            range(-1.0, -0.98, length=Int(ceil(N_trap / 3) + 1)),
            range(-0.98, 0.98, length=Int(ceil(N_trap / 3) + 1)),
            range(0.98, 1.0, length=Int(ceil(N_trap / 3) + 1))
        )
        #μs = range(-1.0 + 1e-6, 1.0 - 1e-6, length=N_trap)
        orig_fs = orig_f.(μs)
        trapz(μs, orig_fs)

    else
        throw(AssertionError("how did you arrive here?"))
    end

    return int / enhancer
end


##########################################################################################92


function print_map_WindowFIntegrated_multipole(
    ss::Vector{Float64},
    windowFint::Union{String,GaPSE.WindowFIntegrated}, out::String;
    s_min, s_max,
    pr::Bool=true, L_max::Int=4, kwargs...)

    GaPSE.check_parent_directory(out)
    GaPSE.check_namefile(out)

    @assert L_max ≥ 0 "L_max must be ≥ 0!"
    @assert haskey(kwargs, :L)==false "you must not provide here the keyword L, use L_max instead!"
    @assert 0.0 < s_min < s_max " 0.0 < s_min < s_max must hold!"
    @assert all(ss .≥ 0.0) "All ss must be ≥ 0.0!"
    #@assert ss[begin] ≈ 0.0 "Why don't you start sampling from s=0 instead from s=$(ss[begin])?"
    @assert all([ss[i+1] > ss[i] for i in 1:(length(ss)-1)]) "ss must be a float vector of increasing values!"

    WINDOWFINT = typeof(windowFint) == String ? GaPSE.WindowFIntegrated(windowFint) : windowFint

    t1 = time()
    PTFs = [zeros(length(ss)) for L in 0:L_max]

    if pr == true
        for L in 0:L_max
            PTFs[L+1] = @showprogress "calculating PhiTimesF L=$L: " [
                begin
                        res = WindowFIntegrated_multipole(s, WINDOWFINT;
                            s_min=s_min, s_max=s_max, L=L, kwargs...)

                        #println("s1 = $s1, s=$s, res = $res")
                        res
                end for s in ss]
        end
    else
        for L in 0:L_max
            PTFs[L+1] = [
                WindowFIntegrated_multipole(s, WINDOWFINT;
                        s_min=s_min, s_max=s_max, L=L, kwargs...)
                for s in ss]
        end
    end


    t2 = time()

    (pr) && println("\ntime needed for print_map_WindowFIntegrated_multipole " *
                    "[in s] = $(@sprintf("%.5f", t2-t1)) \n")


    open(out, "w") do io

        println(io, GaPSE.BRAND)
        println(io, "# This is an integration map of the Q_{l_1} multipoles, defined as:")
        println(io, "#      Q_{l_1}(s_1, s \\mu) = \\int_{-1}^{+1} \\mathrm{d}\\mu \\mathcal{L}_{l_1}(\\mu) \\mathcal{F}(s, \\mu)")
        println(io, "#      \\mathcal{F}(s, \\mu) = \\int_0^{\\infty} \\mathrm{d}s_1 s_1^2 \\phi(s_1) \\phi(\\sqrt(s_1^2 + s^2 + 2 s_1 s \\mu)) F(s/s_1, \\mu)")
        println(io, "# where \\mathcal{L}_{l_1}(\\mu) is tre Legendre polynomial if order l1 and")
        println(io, "# F(x, \\mu) is the window function considered (for its analytical definition, check the code).\n#")

        println(io, "#\n# Time needed for this computation [in s]: $(t2-t1)")
        println(io, "# The keyword arguments were:")

        if !isempty(kwargs)
            for key in keys(kwargs)
                println(io, "# \t\t$(key) = $(kwargs[key])")
            end
        end

        println(io, "#\n# s [h_0^{-1} Mpc] \t " *
                    join(["Q_{l_1=$L} \t " for L in 0:L_max]))
        for (i, s) in enumerate(ss)
            println(io, "$s \t " *
                        join(["$(PTFs[L+1][i]) \t " for L in 0:L_max]))
        end

    end

end


function print_map_WindowFIntegrated_multipole(
    s_zs::Vector{Float64},
    windowFint::Union{String,GaPSE.WindowFIntegrated}, out::String,
    file_data::String; z_min, z_max,
    names_bg=GaPSE.NAMES_BACKGROUND, h_0=0.7, kwargs...)

    @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"

    @assert all(s_zs .≥ 0.0) "All s_zs must be ≥ 0.0!"
    #@assert s_zs[begin] ≈ 0.0 "Why don't you start sampling from z=0 instead from z=$(s_zs[begin])?"
    @assert all([s_zs[i+1] > s_zs[i] for i in 1:(length(s_zs)-1)]) "s_zs must be a float vector of increasing values!"


    BD = GaPSE.BackgroundData(file_data, 1000.0; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    ss = s_zs[1] ≈ 0.0 ? union([0.0], s_of_z.(s_zs[begin+1:end])) : s_of_z.(s_zs)

    print_map_WindowFIntegrated_multipole(ss,
        windowFint, out; s_min=s_of_z(z_min), s_max=s_of_z(z_max), kwargs...)
end

function print_map_WindowFIntegrated_multipole(
    windowFint::Union{String,GaPSE.WindowFIntegrated}, out::String,
    file_data::String; z_min, z_max,
    names_bg=GaPSE.NAMES_BACKGROUND, h_0=0.7, N::Int=100, m::Float64=2.1, st::Float64=0.0, kwargs...)

    @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"
    @assert N > 9 "N > 9 must hold!"
    @assert 0.0 < m < 10.0 "0.0 < m < 10.0 must hold!"
    @assert st ≥ 0.0 "st must be ≥ 0.0, not $st !"
    BD = GaPSE.BackgroundData(file_data, z_max; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    s_min, s_max = s_of_z(z_min), s_of_z(z_max)


    ss = st ≈ 0.0 ? begin
        union([0.0], [s for s in range(0.0, m * s_max, length=N)][begin+1:end])
    end : [s for s in range(st, m * s_max, length=N)]

    print_map_WindowFIntegrated_multipole(ss,
        windowFint, out; s_min=s_min, s_max=s_max, kwargs...)
end


"""
    print_map_WindowFIntegrated_multipole(
        ss::Vector{Float64},
        windowFint::Union{String,GaPSE.WindowFIntegrated}, out::String;
        s_min, s_max,
        pr::Bool=true, L_max::Int=4, alg::Symbol=:lobatto,
        N_lob::Int=100, N_trap::Int=200,
        atol_quad::Float64=0.0, rtol_quad::Float64=1e-2,
        enhancer::Float64=1e6)

    print_map_WindowFIntegrated_multipole(
        s_zs::Vector{Float64},
        windowFint::Union{String,GaPSE.WindowFIntegrated}, out::String,
        file_data::String; z_min, z_max,
        names_bg=GaPSE.NAMES_BACKGROUND, h_0=0.7, kwargs...))

    print_map_WindowFIntegrated_multipole(
        windowFint::Union{String,GaPSE.WindowFIntegrated}, out::String,
        file_data::String; z_min, z_max,
        names_bg=GaPSE.NAMES_BACKGROUND, h_0=0.7, N::Int=100, 
        m::Float64=2.1, st::Float64=0.0, kwargs...)

Evaluate the integrated window function multipoles ``Q_{\\ell_1}(s)`` in a vector of ``s`` values for all
the multipoles ``0 \\leq \\ell_1 \\leq L_\\mathrm{max}``, and print the results in the `out` file.
The computation of the multipole is performed through `WindowFIntegrated_multipole`.

The first method takes as input:
- `ss::Vector{Float64}` :  the vector of s points where to 
  sample the integrated window function multipoles ``Q_{\\ell_1}``.`ss` must be a float vector of 
  increasing comoving distance values (so each element must be ≥ 0); the first and last values 
  ARE NOT RELATED to `s_min` and `s_max`.
- `windowFint::Union{String,WindowFIntegrated}`, i.e. the integrated window function itself; it can be 
  passed as the namefile where the integrated window is stored in (that will be opened with `WindowFIntegrated`) 
  or as a `WindowFIntegrated` struct directly.
- `out::String` : the name of the output file
- `s_min` and `s_max` (keyword arguments) : min and max comoving distance of the survey;
  their values will be internally used by 

The second method takes as input the min and max redshifts of the survey (`z_min`and `z_max`),
the vector of redshifts `zs::Vector{Float64}` for the integrated window function sampling, `windowFint` 
as before and the `file_data` where can be found the association ``z \\rightarrow s(z)``. 
Such file must have the structure of the 
background data produced by the CLASS (link: https://github.com/lesgourg/class_public) code.
Note that also `zs` musyt be a float vector of increasing redshift values (so each element must be ≥ 0).
This method internally recalls the first one, so the other `kwargs...` are in common.

The third method takes as input the min and max redshifts of the survey (`z_min`and `z_max`) and the same 
input as the second method (`windowF`, `out` and `file_data`) but NOT THE REDSHIFT SAMPLING VECTOR `zs`.
The sampling will be internally made linearly from ``s = \\mathrm{st}``(where `st::Float64 = 0.0` is a 
keyword argument) to ``s = m \\, s_{\\mathrm{max}}``, where `s_max` is the comoving distance associated to 
`z_max` (for the data stored in `file_data`) and `m::Float64 = 2.1` a coefficient that we 
suggest to set equals to `2 < m < 3`.
`N::Int = 100` is the number of `s` values used for the sampling in the interval 
``[0, m \\, s_{\\mathrm{max}}]``.
This method internally recalls the first one, so the other `kwargs...` are in common.

The analytical expression for the integrated window function is the following:

```math
Q_{\\ell_1}(s) = 
    \\int_{-1}^{1} \\mathrm{d}\\mu \\; \\mathcal{L}_{\\ell_1}(\\mu) 
    \\; \\mathcal{F}(s, \\mu)
```

where ``s`` is the comoving distance, ``\\mu`` the cosine angle, ``\\mathcal{L}_{\\ell_1}`` the
Legendre polynomial of order ``\\ell_1`` and ``\\mathcal{F}(x, μ)`` the 
integrated window function. Check the documentation of `WindowFIntegrated` for its definition.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

## Example

julia> windowfint = GaPSE.WindowFIntegrated(PATH_TO_GAPSE*"data/IntegrF_REFERENCE_pi2_z115.txt");
julia> GaPSE.print_map_WindowFIntegrated_multipole(windowfint, "my_Ql_multipoles.txt", 
    PATH_TO_GAPSE*"data/WideA_ZA_background.dat"; z_min = 1.0, z_max=1.5, st = 1.0, N=500, pr=false)
julia> run(`head -n 20  \$(DIR*"my_Ql_multipoles.txt")`)
###############
#    GaPSE    #
############### 
#
# This is an integration map of the Q_{l_1} multipoles, defined as:
#      Q_{l_1}(s_1, s \\mu) = \\int_{-1}^{+1} \\mathrm{d}\\mu \\mathcal{L}_{l_1}(\\mu) \\mathcal{F}(s, \\mu)
#      \\mathcal{F}(s, \\mu) = \\int_0^{\\infty} \\mathrm{d}s_1 s_1^2 \\phi(s_1) \\phi(\\sqrt(s_1^2 + s^2 + 2 s_1 s \\mu)) F(s/s_1, \\mu)
# where \\mathcal{L}_{l_1}(\\mu) is tre Legendre polynomial if order l1 and
# F(x, \\mu) is the window function considered (for its analytical definition, check the code).
#
#
# Time needed for this computation [in s]: 27.256186962127686
# The keyword arguments were:
#
# s [h_0^{-1} Mpc]      Q_{l_1=0}      Q_{l_1=1}      Q_{l_1=2}      Q_{l_1=3}      Q_{l_1=4}      
1.0      4.1857800000750543e11      -4.377435879373042e7      -5.084259164821501e8      1.2380785453994218e6      -3.641597411371149e8      
13.852533751787348      4.1473071900503394e11      -6.063857839848524e8      -1.342493986435839e9      1.7150523594728626e7      -2.0225033264194965e8
...            ...            ...            ...            ...            ...


## Optional arguments

As optional arguments of the first method:

- `s_min` and `s_max` (mandatory keyword arguments) : min and max comoving distance of the survey;
  their values will be internally used by 

- `pr::Bool = true` : do you want to see the progress-bar of the computation?

- `L_max::Int64 = 4` : maximum multipole order to be computed

- `alg::Symbol = :lobatto` : algorithm to be used for the integration; the valid options 
  are (other values will lead to `AssertionError`):
  - `:quad` -> the integration over ``\\mu`` will be preformed through the Julia function `quadgk` 
  from the [`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl) Julia package, that uses an adaptive 
  Gauss-Kronrod quadrature.
  - `:trap` -> the integration over ``\\mu`` will be preformed through the Julia function `trapz` 
  from the [`Trapz.jl`](https://github.com/francescoalemanno/Trapz.jl) Julia package, that uses the
  simple trapezoidal rulae.
  - `:lobatto` -> the integration over ``\\mu`` will be preformed through the Julia function `gausslobatto` 
  from the [`FastGaussQuadrature.jl`](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) Julia package, 
  that uses the Gauss-Lobatto quadrature. 

- `N_lob::Int = 100` : number of points to be used in the sampling made by the function `trapz`.
  Note that these options will have an effect only if you se `alg = :quad`.

- `N_trap::Int = 200` : number of points to be used in the sampling made by the function `trapz`.
  Note that these options will have an effect only if you se `alg = :quad`.

- `atol_quad::Float64 = 0.0` and `rtol_quad::Float64 = 1e-2`: absolute and relative tolerance
  to be passed to the function `quadgk`; it's recommended not to set `rtol_quad < 1e-2` 
  because the time for evaluation increase quickly.
  Note that these options will have an effect only if you se `alg = :quad`.

- `enhancer::Float64 = 1e6`: just a float number used in order to deal better with small numbers; 
  the returned value is NOT modified by this value, because after a multiplication
  the internal result is divided by `enhancer`.

The optional arguments given to the second method will be directly given to the first one.
The only two exceptions are:

- `s_min` and `s_max` (mandatory keyword arguments) : min and max redshift of the survey;
  their values will be internally coverted to comoving distances and passed to the first method

- `names = NAMES_BACKGROUND` : the column names of the `file_data`. If the colum order change from
  the default one `NAMES_BACKGROUND`, you must set as input the vector of string with the correct
  one, with the SAME names. They are, with the default order:\n
  $(NAMES_BACKGROUND)

- `h = 0.7` : the adimensional hubble constant. By default, CLASS background data are measured with
  it numerically expressed (so distances are measured in `Mpc`, for example), while this code works
  with `h` in the unit of measure (so distances are measured in `Mpc/h`, for example).
  Change this value to `1.0` if the input data do not have this issue, or to your value of interest 
  (`0.67`, `0.5`, ...).


The optional arguments given to the third method will be directly given to the first one again.
The only two exceptions are:

- `names_bg=GaPSE.NAMES_BACKGROUND` and `h_0=0.7` : same as for the second method

- `N::Int=100` : number of points to be used in the liearly spaced comoving distance vector

- `st::Float64=0.0` : starting comoving distance of the vector

- `m:Float64 = 2.1` : coefficient that set the maximum comoving distance of the vector, equals to ``m * s_max``,
  where `s_max` is the comoving distance associated to the redhsift `z_max`

See also:  [`WindowFIntegrated_multipole`](@ref), [`WindowFIntegrated`](@ref), 
[`WindowF`](@ref), [`ϕ`](@ref), [`BackgroundData`](@ref)
"""
print_map_WindowFIntegrated_multipole


#=
function PhiTimesWindowF(s1, s, μ, windowf::WindowF; s_min, s_max)
     return ϕ(√(s1^2 + s^2 + 2 * s1 * s * μ), s_min, s_max) * spline_F(s / s1, μ, windowf)
end

function PhiTimesWindowF_multipole(
     s1, s, windowf::WindowF;
     s_min, s_max,
     L::Int=0, alg::Symbol=:lobatto,
     N_lob::Int=100, N_trap::Int=200,
     atol_quad::Float64=0.0, rtol_quad::Float64=1e-2,
     enhancer::Float64=1e6,
     kwargs...)

     @assert alg ∈ VALID_INTEGRATION_ALGORITHM ":$alg is not a valid Symbol for \"alg\"; they are: \n\t" *
                                               "$(":".*string.(VALID_INTEGRATION_ALGORITHM) .* vcat([" , " for i in 1:length(VALID_INTEGRATION_ALGORITHM)-1], " .")... )"

     @assert N_trap > 2 "N_trap must be >2,  N_trap = $N_trap is not!"
     @assert N_lob > 2 "N_lob must be >2,  N_lob = $N_lob is not!"
     @assert atol_quad ≥ 0.0 "atol_quad must be ≥ 0.0,  atol_quad = $atol_quad is not!"
     @assert rtol_quad ≥ 0.0 "rtol_trap must be ≥ 0.0,  rtol_quad = $rtol_quad is not!"
     @assert L ≥ 0 "L must be ≥ 0, L = $L is not!"


     orig_f(μ) = enhancer * PhiTimesWindowF(s1, s, μ, windowf; s_min=s_min, s_max=s_max) * Pl(μ, L)

     int = if alg == :lobatto
          xs, ws = gausslobatto(N_lob)
          dot(ws, orig_f.(xs))

     elseif alg == :quad
          quadgk(μ -> orig_f(μ), -1.0, 1.0; atol=atol_quad, rtol=rtol_quad)[1]

     elseif alg == :trap
          μs = union(
               range(-1.0, -0.98, length=Int(ceil(N_trap / 3) + 1)),
               range(-0.98, 0.98, length=Int(ceil(N_trap / 3) + 1)),
               range(0.98, 1.0, length=Int(ceil(N_trap / 3) + 1))
          )
          #μs = range(-1.0 + 1e-6, 1.0 - 1e-6, length=N_trap)
          orig_fs = orig_f.(μs)
          trapz(μs, orig_fs)

     else
          throw(AssertionError("how the hell did you arrive here?"))
     end

     return int / enhancer
end



##########################################################################################92



function print_map_PhiTimesWindowF_multipole(
     s1_ss::Vector{Float64}, s_ss::Vector{Float64},
     windowF::Union{String,WindowF}, out::String;
     s_min, s_max,
     pr::Bool=true, L_max::Int=4, kwargs...)

     check_parent_directory(out)
     check_namefile(out)

     @assert L_max ≥ 0 "L_max must be ≥ 0!"
     @assert 0.0 < s_min < s_max " 0.0 < s_min < s_max must hold!"
     @assert all(s_ss .≥ 0.0) "All s_ss must be ≥ 0.0!"
     @assert s_ss[begin] ≈ 0.0 "Why don't you start sampling from s=0 instead from s=$(s_ss[begin])?"
     @assert all([s_ss[i+1] > s_ss[i] for i in 1:(length(s_ss)-1)]) "s_ss must be a float vector of increasing values!"
     @assert all(s1_ss .≥ 0.0) "All s1_ss must be ≥ 0.0!"
     @assert s1_ss[begin] > 0.0 "The vector s1_ss must start from a value > 0, not $(s1_ss[begin])!"
     @assert all([s1_ss[i+1] > s1_ss[i] for i in 1:(length(s1_ss)-1)]) "s1_ss must be a float vector of increasing values!"

     WINDOWF = typeof(windowF) == String ? WindowF(windowF) : windowF

     s1_ss_grid = [s1 for s1 in s1_ss for s in s_ss]
     s_ss_grid = [s for s in s1_ss for s in s_ss]

     t1 = time()
     PTFs = [zeros(length(s1_ss_grid)) for L in 0:L_max]

     if pr == true
          for L in 0:L_max
               PTFs[L+1] = @showprogress "calculating PhiTimesF L=$L: " [
                    begin
                         res = PhiTimesWindowF_multipole(
                              s1, s, WINDOWF; s_min=s_min, s_max=s_max,
                              L=L, kwargs...)

                         #println("s1 = $s1, s=$s, res = $res")
                         res
                    end for (s1, s) in zip(s1_ss_grid, s_ss_grid)]
          end
     else
          for L in 0:L_max
               PTFs[L+1] = [
                    PhiTimesWindowF_multipole(
                         s1, s, WINDOWF; s_min=s_min, s_max=s_max,
                         L=L, kwargs...)
                    for (s1, s) in zip(s1_ss_grid, s_ss_grid)]
          end
     end


     t2 = time()

     (pr) && println("\ntime needed for print_map_PhiTimesWindowF_multipole " *
                     "[in s] = $(@sprintf("%.5f", t2-t1)) \n")


     open(out, "w") do io

          println(io, BRAND)
          println(io, "# This is an integration map of the F_{l_1} multipoles, defined as:")
          println(io, "#      F_{l_1}(s_1, s \\mu) = \\int_{-1}^{+1} \\mathrm{d}\\mu \\mathcal{L}_{l_1}(\\mu) F(s_1, s, \\mu)")
          println(io, "#      F(s_1, s \\mu) =  \\phi(\\sqrt(s_1^2 + s^2 + 2 s_1 s \\mu)) F(s/s_1, \\mu)")
          println(io, "#                    =  \\sum_{l_1=0}^{\\infty} (2 l_1 + 1) \\mathcal{L}_{l_1}(\\mu) F_{l_1}(s_1, s) / 2 ")
          println(io, "# where \\mathcal{L}_{l_1}(\\mu) is thre Legendre polynomial if order l1 and")
          println(io, "# F(x, \\mu) is stored in a WindowF struct (for its analytical definition, check the code).\n#")

          println(io, "#\n# Time needed for this computation [in s]: $(t2-t1)")
          println(io, "# Range of interest:")
          println(io, "# \t s_min = $s_min h_0^{-1} Mpc")
          println(io, "# \t s_max = $s_max h_0^{-1} Mpc")
          println(io, "# The keyword arguments were:")

          if !isempty(kwargs)
               for key in keys(kwargs)
                    println(io, "# \t\t$(key) = $(kwargs[key])")
               end
          end

          println(io, "#\n# s1 [h_0^{-1} Mpc] \t s [h_0^{-1} Mpc] \t " *
                      join(["F_{l_1=$L} \t " for L in 0:L_max]))
          for (i, (s1, s)) in enumerate(zip(s1_ss_grid, s_ss_grid))
               println(io, "$s1 \t $s \t " *
                           join(["$(PTFs[L+1][i]) \t " for L in 0:L_max]))
          end

     end

end

function print_map_PhiTimesWindowF_multipole(
     s1_zs::Vector{Float64}, s_zs::Vector{Float64},
     windowF::Union{String,WindowF}, out::String,
     file_data::String; z_min, z_max,
     names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)

     @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"
     @assert all(s1_zs .≥ 0.0) "All s1_zs must be ≥ 0.0!"
     @assert s1_zs[begin] > 0.0 "The vector z1_ss must start from a value > 0, not $(z1_ss[begin])!"
     @assert all([s1_zs[i+1] > s1_zs[i] for i in 1:(length(s1_zs)-1)]) "s1_zs must be a float vector of increasing values!"

     @assert all(s_zs .≥ 0.0) "All s_zs must be ≥ 0.0!"
     @assert s_zs[begin] ≈ 0.0 "Why don't you start sampling from z=0 instead from z=$(s_zs[begin])?"
     @assert all([s_zs[i+1] > s_zs[i] for i in 1:(length(s_zs)-1)]) "s_zs must be a float vector of increasing values!"


     BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
     s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
     s1_ss = s_of_z.(s1_zs)
     s_ss = union([0.0], s_of_z.(s_zs[begin+1:end]))

     print_map_PhiTimesWindowF_multipole(s1_ss, s_ss,
          windowF, out; s_min=s_of_z(z_min), s_max=s_of_z(z_max), kwargs...)
end

function print_map_PhiTimesWindowF_multipole(
     windowF::Union{String,WindowF}, out::String,
     file_data::String; z_min, z_max,
     names_bg=NAMES_BACKGROUND, h_0=0.7, N_s1_ss::Int=100, N_s_ss::Int=100,
     m_s1::Float64=2.1, m_s::Float64=2.1, st::Float64=1.0, kwargs...)

     @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"
     @assert N_s1_ss > 9 "N_s1_ss > 9 must hold!"
     @assert 0.0 < m_s1 < 10.0 "0.0 < m_s1 < 10.0 must hold!"
     @assert N_s_ss > 9 "N_s_ss > 9 must hold!"
     @assert 0.0 < m_s < 10.0 "0.0 < m_s < 10.0 must hold!"
     BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
     s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
     s_min, s_max = s_of_z(z_min), s_of_z(z_max)

     s1_ss = [s1 for s1 in range(st, m_s1 * s_max, length=N_s1_ss)]
     s_ss = union([0.0], [s for s in range(0.0, m_s * s_max, length=N_s_ss)][begin+1:end])

     print_map_PhiTimesWindowF_multipole(s1_ss, s_ss,
          windowF, out; s_min=s_min, s_max=s_max, kwargs...)
end






##########################################################################################92

##########################################################################################92

##########################################################################################92



function Q_multipole(
    s, windowf::WindowF;
    s_min, s_max,
    L::Int=0, alg::Symbol=:quad,
    llim=nothing, rlim=nothing,
    N_trap::Int=200,
    atol_quad::Float64=0.0, rtol_quad::Float64=1e-2,
    enhancer::Float64=1e6,
    in_alg::Symbol=:lobatto,
    in_N_lob::Int=100, in_N_trap::Int=200,
    in_atol_quad::Float64=0.0, in_rtol_quad::Float64=1e-2,
    in_enhancer::Float64=1e6, in_st::Float64=1.0,
    kwargs...)


    @assert N_trap > 2 "N_trap must be >2,  N_trap = $N_trap is not!"
    @assert atol_quad ≥ 0.0 "atol_quad must be ≥ 0.0,  atol_quad = $atol_quad is not!"
    @assert rtol_quad ≥ 0.0 "rtol_trap must be ≥ 0.0,  rtol_quad = $rtol_quad is not!"
    @assert L ≥ 0 "L must be ≥ 0, L = $L is not!"
    @assert isnothing(llim) || llim ≥ 0.0 "llim must be nothing or ≥ 0.0!"
    @assert isnothing(rlim) || rlim > 0.0 "rlim must be nothing or > 0.0!"
    @assert isnothing(llim) || isnothing(rlim) || rlim > llim "rlim must be > llim!"


    LLIM = isnothing(llim) ? 0.95 * s_min : llim
    RLIM = isnothing(rlim) ? 1.05 * s_max : isinf(rlim) ? 3.0 * s_max : rlim
    f(s1) = ϕ(s1, s_min, s_max) > 0 ? begin
        enhancer * s1^2 * PhiTimesWindowF_multipole(s1, s, windowf;
            s_min=s_min, s_max=s_max, L=L, alg=in_alg, N_lob=in_N_lob, N_trap=in_N_trap,
            atol_quad=in_atol_quad, rtol_quad=in_rtol_quad,
            enhancer=in_enhancer, st=in_st) * ϕ(s1, s_min, s_max)
    end : 0.0


    res = if alg == :trap
        ps = range(LLIM, RLIM, length=N_trap)
        trapz(ps, f.(ps))

    elseif alg == :quad
        quadgk(s -> f(s), LLIM, RLIM; rtol=rtol_quad, atol=atol_quad)[1]
    else
        throw(AssertionError("The value 'alg = :$alg' is not a valid algorithm; you must " *
                            "choose between ':trap' and ':quad' . "))
    end

    return res / enhancer
end



##########################################################################################92



function print_map_Q_multipole(
    s_ss::Vector{Float64},
    windowF::Union{String,WindowF}, out::String;
    s_min, s_max,
    pr::Bool=true, L_max::Int=4, kwargs...)

    check_parent_directory(out)
    check_namefile(out)

    @assert L_max ≥ 0 "L_max must be ≥ 0!"
    @assert 0.0 < s_min < s_max " 0.0 < s_min < s_max must hold!"
    @assert all(s_ss .≥ 0.0) "All s_ss must be ≥ 0.0!"
    #@assert s_ss[begin] ≈ 0.0 "Why don't you start sampling from s=0 instead from s=$(s_ss[begin])?"
    @assert all([s_ss[i+1] > s_ss[i] for i in 1:(length(s_ss)-1)]) "s_ss must be a float vector of increasing values!"

    WINDOWF = typeof(windowF) == String ? WindowF(windowF) : windowF

    t1 = time()
    Qs = [zeros(length(s_ss)) for L in 0:L_max]

    if pr == true
        for L in 0:L_max
            Qs[L+1] = @showprogress "calculating Q L=$L: " [
                begin
                        res = Q_multipole(
                            s, WINDOWF; s_min=s_min, s_max=s_max,
                            L=L, kwargs...)

                        #println("s1 = $s1, s=$s, res = $res")
                        res
                end for s in s_ss]
        end
    else
        for L in 0:L_max
            Qs[L+1] = [
                Q_multipole(
                        s, WINDOWF; s_min=s_min, s_max=s_max,
                        L=L, kwargs...)
                for s in s_ss]
        end
    end


    t2 = time()

    (pr) && println("\ntime needed for print_map_Q_multipole " *
                    "[in s] = $(@sprintf("%.5f", t2-t1)) \n")


    open(out, "w") do io

        println(io, BRAND)
        println(io, "# This is an integration map of the Q_{l_1} multipoles, defined as:")
        println(io, "#      Q_{l_1}(s_1, s \\mu) = \\int_0^{\\infty} \\mathrm{d}s_1 s_1^2 \\phi(s_1) F_{l_1}(s_1, \\mu)")
        println(io, "#      F_{l_1}(s_1, s \\mu) = \\int_{-1}^{+1} \\mathrm{d}\\mu \\mathcal{L}_{l_1}(\\mu) F(s_1, s, \\mu)")
        println(io, "#      F(s_1, s \\mu) =  \\phi(\\sqrt(s_1^2 + s^2 + 2 s_1 s \\mu)) F(s/s_1, \\mu)")
        println(io, "#                    =  \\sum_{l_1=0}^{\\infty} (2 l_1 + 1) \\mathcal{L}_{l_1}(\\mu) F_{l_1}(s_1, s) / 2 ")
        println(io, "# where \\mathcal{L}_{l_1}(\\mu) is thre Legendre polynomial if order l1 and")
        println(io, "# F(x, \\mu) is stored in a WindowF struct (for its analytical definition, check the code).\n#")

        println(io, "#\n# Time needed for this computation [in s]: $(t2-t1)")
        println(io, "# Range of interest:")
        println(io, "# \t s_min = $s_min h_0^{-1} Mpc")
        println(io, "# \t s_max = $s_max h_0^{-1} Mpc")
        println(io, "# The keyword arguments were:")

        if !isempty(kwargs)
            for key in keys(kwargs)
                println(io, "# \t\t$(key) = $(kwargs[key])")
            end
        end

        println(io, "#\n# s [h_0^{-1} Mpc] \t " *
                    join(["Q_{l_1=$L} \t " for L in 0:L_max]))
        for (i, s) in enumerate(s_ss)
            println(io, "$s \t " *
                        join(["$(Qs[L+1][i]) \t " for L in 0:L_max]))
        end

    end

end


function print_map_Q_multipole(
    s_zs::Vector{Float64},
    windowF::Union{String,WindowF}, out::String,
    file_data::String; z_min, z_max,
    names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)

    @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"
    @assert all(s_zs .≥ 0.0) "All s_zs must be ≥ 0.0!"
    #@assert s_zs[begin] ≈ 0.0 "Why don't you start sampling from z=0 instead from z=$(s_zs[begin])?"
    @assert all([s_zs[i+1] > s_zs[i] for i in 1:(length(s_zs)-1)]) "s_zs must be a float vector of increasing values!"


    BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    s_ss = s_of_z.(s1_zs)
    #s_ss = union([0.0], s_of_z.(s_zs[begin+1:end]))

    print_map_Q_multipole(s_ss,
        windowF, out; s_min=s_of_z(z_min), s_max=s_of_z(z_max), kwargs...)
end

function print_map_Q_multipole(
    windowF::Union{String,WindowF}, out::String,
    file_data::String; z_min, z_max,
    names_bg=NAMES_BACKGROUND, h_0=0.7, N::Int=100,
    st::Float64=1.0,
    m::Float64=2.1, kwargs...)

    @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"
    @assert N > 9 "N_s_ss > 9 must hold!"
    @assert 0.0 < m < 10.0 "0.0 < m < 10.0 must hold!"
    BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    s_min, s_max = s_of_z(z_min), s_of_z(z_max)

    s_ss = [s1 for s1 in range(st, m * s_max, length=N)]
    #s_ss = union([0.0], [s for s in range(0.0, m_s * s_max, length=N)][begin+1:end])

    print_map_Q_multipole(s_ss,
        windowF, out; s_min=s_min, s_max=s_max, kwargs...)
end
=#
