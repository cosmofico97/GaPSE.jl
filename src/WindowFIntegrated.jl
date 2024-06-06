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
    integrated_F_quadgk(s, μ, s_min, s_max, windowF::WindowF;
        llim=0.0, rlim=Inf, rtol=1e-2, atol=0.0)

Computes the Integrated Window Function fron the input Window Function `windowF`, 
through the `quadgk` function of the [QuadGK](https://github.com/JuliaMath/QuadGK.jl) Julia
pagkage, in the point `(s,μ)` (where s is the comoving distance and μ the angle cosine).
`s_min` and `s_max` are the min and max value for the radial part of the survey window function.

The analytical expression of the Integrated window function is the following:
```math
\\mathcal{F}(s, \\mu) = 
    \\int_0^\\infty \\mathrm{d}s_1 \\, \\phi(s_1) \\,
    \\phi\\left(\\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}\\right) 
    \\, F\\left(\\frac{s}{s_1}, \\mu \\right)
```

where ``s`` is the comoving distance, ``\\mu`` the cosine angle,
``\\phi`` is the angular part of the survey window function and ``F(x, μ)`` is the 
window function. Check the documentation of `WindowF` for its definition.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.
     
## Optional arguments

- `llim=nothing` and `rlim=nothing` : integration limits for `quadgk`;
  if `llim=nothing` it will be set to `0.95 * s_min`; if `rlim=nothing`, it will be set to `1.05*s_max`,
  while if `rlim=Inf` it will be set to `3 * s_max`
- `rtol=1e-2` and `atol=0.0` : relative and absoute tolerance for `quadgk`


See also: [`WindowF`], [`ϕ`](@ref)
"""
function integrated_F_quadgk(s, μ, s_min, s_max, windowF::WindowF;
    llim=nothing, rlim=nothing, rtol=1e-2, atol=0.0)

    LLIM = isnothing(llim) ? 0.95 * s_min : llim
    RLIM = isnothing(rlim) ? 1.05 * s_max : isinf(rlim) ? 3.0 * s_max : rlim
    f(p) = ϕ(p, s_min, s_max) > 0 ? begin
        spline_F(s / p, μ, windowF) * ϕ(p, s_min, s_max) * ϕ(s2(p, s, μ), s_min, s_max) * p^2
    end : 0.0
    quadgk(p -> f(p), LLIM, RLIM; rtol=rtol, atol=atol)[1]
end


#=
"""
    integrated_F_quadgk(s, μ, z_min, z_max, windowF::WindowF, file_data::String;
        names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)

Same as the othert method, but this one takes as input REDSHIFTS and not comoving distances.
In order to convert from one to the other, you must provide a `file_data` with the appropriate
data; it is expected that such file is an output of CLASS.
"""
function integrated_F_quadgk(s, μ, z_min, z_max, windowF::WindowF, file_data::String;
    names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)
    BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    integrated_F_quadgk(s, μ, s_of_z(z_min), s_of_z(z_max), windowF::WindowF; kwargs...)
end
=#


"""
    integrated_F_trapz(s, μ, s_min, s_max, windowF::WindowF;
        llim=nothing, rlim=nothing, N::Int=1000)

Computes the Integrated Window Function fron the input Window Function `windowF`, 
through the `trapz` function of the [Trapz](https://juliapackages.com/p/trapz) Julia
pagkage, in the point `(s,μ)` (where s is the comoving distance and μ the angle cosine).
`s_min` and `s_max` are the min and max value for the radial part of the survey window function.

The analytical expression of the Integrated window function is the following:
```math
\\mathcal{F}(s, \\mu) = 
    \\int_0^\\infty \\mathrm{d}s_1 \\, \\phi(s_1) \\,
    \\phi\\left(\\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}\\right) 
    \\, F\\left(\\frac{s}{s_1}, \\mu \\right)
```

where ``s`` is the comoving distance, ``\\mu`` the cosine angle,
``\\phi`` is the angular part of the survey window function and ``F(x, μ)`` is the 
window function. Check the documentation of `WindowF` for its definition.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.
     
## Optional arguments

- `llim=nothing` and `rlim=nothing` : limits of the sampling interval to be used for `trapz`;
  if `llim=nothing` it will be set to `0.95 * s_min`; if `rlim=nothing`, it will be set to `1.05*s_max`,
  while if `rlim=Inf` it will be set to `3 * s_max`
- `N::Int = 1000` : number of points to be used for the sampling


See also: [`WindowF`], [`ϕ`](@ref)
"""
function integrated_F_trapz(s, μ, s_min, s_max, windowF::WindowF;
    llim=nothing, rlim=nothing, N::Int=1000)

    LLIM = isnothing(llim) ? 0.95 * s_min : llim
    RLIM = isnothing(rlim) ? 1.05 * s_max : isinf(rlim) ? 3.0 * s_max : rlim
    f(p) = ϕ(p, s_min, s_max) > 0 ? begin
        spline_F(s / p, μ, windowF) * ϕ(p, s_min, s_max) * ϕ(s2(p, s, μ), s_min, s_max) * p^2
    end : 0.0
    ps = range(LLIM, RLIM, length=N)
    trapz(ps, f.(ps))
end


#=
"""
    integrated_F_trapz(s, μ, z_min, z_max, windowF::WindowF, file_data::String;
        names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)

Same as the othert method, but this one takes as input REDSHIFTS and not comoving distances.
In order to convert from one to the other, you must provide a `file_data` with the appropriate
data; it is expected that such file is an output of CLASS.
"""
function integrated_F_trapz(s, μ, z_min, z_max, windowF::WindowF, file_data::String;
    names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)
    BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    integrated_F_trapz(s, μ, s_of_z(z_min), s_of_z(z_max), windowF::WindowF; kwargs...)
end
=#

##########################################################################################92

#=
function print_map_IntegratedF(in::String, out::String, s_min, s_max,
    μs::Vector{Float64}; kwargs...)

    check_parent_directory(out)
    check_namefile(out)

    windowF = WindowF(in)
    windowFint = WindowFIntegrated(s_min, s_max, μs, windowF; kwargs...)
    print_map_IntegratedF(out, windowFint)
end
=#

function print_map_IntegratedF(s_min, s_max, ss::Vector{Float64},
    μs::Vector{Float64}, windowF::Union{String,WindowF}, out::String;
    alg::Symbol=:trap, llim=nothing, rlim=nothing,
    rtol=1e-2, atol=0.0, N::Int=1000, pr::Bool=true)

    check_parent_directory(out)
    check_namefile(out)

    @assert 0.0 < s_min < s_max " 0.0 < s_min < s_max must hold!"
    @assert 9 < N < 100001 " 10 < N < 100001 must hold!"
    @assert all(ss .≥ 0.0) "All ss must be ≥ 0.0!"
    @assert ss[begin] ≈ 0.0 "Why don't you start sampling from s=0ad from s=$(ss[begin])?"
    @assert all([ss[i+1] > ss[i] for i in 1:(length(ss)-1)]) "ss must be a float vector of increasing values!"
    @assert all(μs .≥ -1.0) "All μs must be ≥-1.0!"
    @assert all([μs[i+1] > μs[i] for i in 1:(length(μs)-1)]) "μs must be a float vector of increasing values!"
    @assert all(μs .≤ 1.0) "All μs must be ≤1.0!"
    @assert isnothing(llim) || llim ≥ 0.0 "llim must be nothing or ≥ 0.0!"
    @assert isnothing(rlim) || rlim > 0.0 "rlim must be nothing or > 0.0!"
    @assert isnothing(llim) || isnothing(rlim) || rlim > llim "rlim must be > llim!"

    #ss_step=21.768735478453323
    #@assert ss_start ≥ 0.0 " ss_start ≥ 0.0 must hold!"
    #@assert (iszero(ss_stop) && s_max ≥ ss_start + 3 * ss_step) || (ss_stop ≥ ss_start + 3 * ss_step)
    #" (ss_stop == 0 && s_max ≥ ss_start + 3 * ss_step) || ss_stop ≥ ss_start + 3 * ss_step musty hold!"
    #SS_STOP = iszero(ss_stop) ? 3.0 * s_max : ss_stop
    #ss = [s for s in ss_start:ss_step:SS_STOP]

    WINDOWF = typeof(windowF) == String ? WindowF(windowF) : windowF

    t1 = time()

    IFs = if alg == :trap
        pr ? begin
            @showprogress "calculating intF: " [
                integrated_F_quadgk(s, μ, s_min, s_max, WINDOWF;
                        llim=llim, rlim=rlim, rtol=rtol, atol=atol)
                for s in ss, μ in μs]
        end : begin
            [integrated_F_quadgk(s, μ, s_min, s_max, WINDOWF;
                llim=llim, rlim=rlim, rtol=rtol, atol=atol)
            for s in ss, μ in μs]
        end
    elseif alg == :quad
        pr ? begin
            @showprogress "calculating intF: " [
                integrated_F_trapz(s, μ, s_min, s_max, WINDOWF;
                        llim=llim, rlim=rlim, N=N)
                for s in ss, μ in μs]
        end : begin
            [integrated_F_trapz(s, μ, s_min, s_max, WINDOWF;
                llim=llim, rlim=rlim, N=N)
            for s in ss, μ in μs]
        end
    else
        throw(AssertionError("The value 'alg = :$alg' is not a valid algorithm; you must " *
                            "choose between ':trap' and ':quad' . "))
    end

    t2 = time()

    (pr) && println("\ntime needed for print_map_IntegratedF " *
                    "[in s] = $(@sprintf("%.5f", t2-t1)) \n")


    ss_grid = [s for s in ss for μ in μs]
    μs_grid = [μ for s in ss for μ in μs]
    IFs_grid = reshape(transpose(IFs), (:,))

    open(out, "w") do io
        println(io, BRAND)
        println(io, "# This is an integration map of the function \\mathcal{F}(s, \\mu), defined as:")
        println(io, "# \\mathcal{F}(s, \\mu) = \\int_0^\\infty dp p^2 \\phi(p) \\phi(\\sqrt{p^2 + s^2 + 2 p s \\mu}) F(s/p, \\mu)")
        println(io, "# where F(x, \\mu) is stored in a WindowF struct (for its analytical definition, check the code.\n#")

        println(io, "#\n# Time needed for this computation [in s]: $(t2-t1)")
        println(io, "# Range of interest:")
        println(io, "# \t s_min = $s_min h_0^{-1} Mpc")
        println(io, "# \t s_max = $s_max h_0^{-1} Mpc")
        println(io, "# The keyword arguments were:")
        println(io, "# \t alg = :$alg \t llim = $llim \t rlim = $rlim")
        println(io, "# \t rtol = $rtol \t atol = $atol \t N = $N \t pr = $pr")

        println(io, "#\n# s [h_0^{-1} Mpc] \t mu \t IF")
        for (s, μ, F) in zip(ss_grid, μs_grid, IFs_grid)
            println(io, "$s\t $μ \t $F")
        end
    end
end


function print_map_IntegratedF(z_min, z_max, zs::Vector{Float64},
    μs::Vector{Float64}, windowF::Union{String,WindowF}, out::String,
    file_data::String;
    names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)

    @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"
    @assert all(zs .≥ 0.0) "All zs must be ≥ 0.0!"
    @assert zs[begin] ≈ 0.0 "Why don't you start sampling from z=0 instead from z=$(zs[begin])?"
    @assert all([zs[i+1] > zs[i] for i in 1:(length(zs)-1)]) "zs must be a float vector of increasing values!"

    BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    SS = union([0.0], s_of_z.(zs[begin+1:end]))

    print_map_IntegratedF(s_of_z(z_min), s_of_z(z_max), SS,
        μs, windowF, out; kwargs...)
end

function print_map_IntegratedF(z_min, z_max,
    μs::Vector{Float64}, windowF::Union{String,WindowF}, out::String,
    file_data::String;
    names_bg=NAMES_BACKGROUND, h_0=0.7, N_ss::Int=100, m::Float64=2.1, kwargs...)

    @assert 0.0 ≤ z_min < z_max "0.0 ≤ z_min < z_max must hold!"
    @assert N_ss > 9 "N_ss > 9 must hold!"
    @assert 0.0 < m < 10.0 "0.0 < m < 10.0 must hold!"
    BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    s_min, s_max = s_of_z(z_min), s_of_z(z_max)
    SS = union([0.0], [s for s in range(0.0, m * s_max, length=N_ss)][begin+1:end])

    print_map_IntegratedF(s_min, s_max, SS,
        μs, windowF, out; kwargs...)
end


"""
    print_map_IntegratedF(
        s_min, s_max, 
        ss::Vector{Float64}, μs::Vector{Float64}, 
        windowF::Union{String,WindowF}, out::String;
        alg::Symbol=:trap, llim=nothing, rlim=nothing,
        rtol=1e-2, atol=0.0, N::Int=1000, pr::Bool=true)

    print_map_IntegratedF(
        z_min, z_max, 
        zs::Vector{Float64}, μs::Vector{Float64}, 
        windowF::Union{String,WindowF}, out::String,
        file_data::String; 
        names_bg = NAMES_BACKGROUND, h_0 = 0.7, kwargs...)

    print_map_IntegratedF(
        z_min, z_max,
        μs::Vector{Float64}, 
        windowF::Union{String,WindowF}, out::String,
        file_data::String;
        names_bg = NAMES_BACKGROUND, h_0 = 0.7, N_ss::Int = 100, 
        m::Float64 = 2.1, kwargs...)

Evaluate the integrated window function ``\\mathcal{F}(s,\\mu)`` in a rectangual grid 
of ``\\mu`` and ``s`` values, and print the results in the `out` file.


The first method takes as input:
- `s_min` and `s_max` : min and max comoving distance of the survey; their values will be internally
  used by the radial function `ϕ`
- `ss::Vector{Float64}` and `μs::Vector{Float64}` :  the vector of s and μ points where to  
  sample the integrated window function ``\\mathcal{F}``. They must be a float vector of 
  increasing values; more precisely:
  - `ss` must be a float vector of increasing comoving distance values (so each element must be ≥ 0);
    the first and last values ARE NOT RELATED to `s_min` and `s_max`.
  - `μs` must be a float vector of increasing cosine values (so each element x must be -1 ≤ x ≤ 1).

- `windowF::Union{String,WindowF}`, i.e. the window function itself; it can be passed as the namefile
  where the window is stored in (that will be opened with `WindowF`) or as a `WindowF` struct directly.
- `out::String` : the name of the output file

The second method takes as input the min and max redshifts of the survey (`z_min`and `z_max`),
the vector of redshifts `zs::Vector{Float64}` for the integrated window function sampling, `μs` and `windowF` 
as before and the `file_data` where can be found the association ``z \\rightarrow s(z)``. 
Such file must have the structure of the 
background data produced by the CLASS (link: https://github.com/lesgourg/class_public) code.
Note that also `zs` musyt be a float vector of increasing redshift values (so each element must be ≥ 0).
This method internally recalls the first one, so the other `kwargs...` are in common.

The third method takes as input the min and max redshifts of the survey (`z_min`and `z_max`) and the same 
input as the second method (`μs`, `widnowF`, `out` and `file_data`) but NOT THE REDSHIFT SAMPLING VECTOR `zs`.
The sampling will be internally made linearly from ``s = 0`` to ``s = m \\, s_{\\mathrm{max}}``, 
where `s_max` is the comoving distance associated to `z_max` (for the data stored in `file_data`) 
and `m::Float64 = 2.1` a coefficient that we suggest to set equals to `2 < m < 3`.
`N_ss::Int = 100` is the number of `s` values used for the sampling in the interval 
``[0, m \\, s_{\\mathrm{max}}]``.
This method internally recalls the first one, so the other `kwargs...` are in common.

The analytical expression for the integrated window function is the following:

```math
\\mathcal{F}(s, \\mu) = 
    \\int_0^\\infty \\mathrm{d}s_1 \\, \\phi(s_1) \\,  
    \\phi\\left(\\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}\\right) 
    \\, F\\left(\\frac{s}{s_1}, \\mu \\right)
```

where ``s`` is the comoving distance, ``\\mu`` the cosine angle,
``\\phi`` is the angular part of the survey window function and ``F(x, μ)`` is the 
window function. Check the documentation of `WindowF` for its definition.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

## Optional arguments

As optional arguments of the first method:

- `alg::Symbol = :trap` : algorithm to be used for the integration; the valid options are `:quad`
  (that will recall `integrated_F_quadgk`) and `:trap` (that will recall `integrated_F_trapz`);
  other values will lead to `AssertionError`
- `llim=nothing` and `rlim=nothing` : integration limits for quad/trap; 
  if `llim=nothing` it will be set to `0.95 * s_min`; if `rlim=nothing`, it will be set to `1.05*s_max`,
  while if `rlim=Inf` it will be set to `3 * s_max`.
- `N::Int = 1000` : number of points to be used for the sampling of `trapz`; it's useless if you set
  `alg = :quad`;
- `rtol=1e-2` and `atol=0.0` : relative and absoute tolerance for `quadgk`; they are useless if you set
  `alg = :trap`;
- `pr::Bool = true` : do you want to see the progress-bar of the computation?

The optional arguments given to the second method will be directly given to the first one.
The only two exceptions are options relative to the background data, managed internally by the struct
`BackgroundData`:

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

- `N_ss::Int=100` : number of points to be used in the liearly spaced comoving distance vector

- `st::Float64=0.0` : starting comoving distance of the vector

- `m:Float64 = 2.1` : coefficient that set the maximum comoving distance of the vector, equals to ``m * s_max``,
  where `s_max` is the comoving distance associated to the redhsift `z_max`

See also: [`integrated_F_quadgk`](@ref), [`integrated_F_trapz`](@ref),
[`ϕ`](@ref), [`WindowF`](@ref), [`WindowFIntegrated`](@ref),
[`BackgroundData`](@ref)
"""
print_map_IntegratedF


#=
function print_map_IntegratedF(in::String, out::String, z_min, z_max,
     μs::Vector{Float64}, file_data::String; kwargs...)

     check_parent_directory(out)
     check_namefile(out)

     windowF = WindowF(in)
     windowFint = WindowFIntegrated(z_min, z_max, μs, windowF, file_data; kwargs...)
     print_map_IntegratedF(out, windowFint)
end
=#


##########################################################################################92




"""
    WindowFIntegrated(
        ss::Vector{Float64}
        μs::Vector{Float64}
        IFs::Matrix{Float64}
        )

Struct containing ss, μs and IFs values of the integrated window function ``\\mathcal{F}(s, μ)``.
`ss` and `μs` are 1D vectors containing each value only once, while 
`IFs` values are contained in a matrix of size `(length(ss), length(μs))`, so:
- along a fixed column the changing value is `s`
- along a fixed row the changing value is `μ`

The analytical expression for the integrated window function is the following:

```math
\\mathcal{F}(s, \\mu) = 
    \\int_0^\\infty \\mathrm{d}s_1 \\, \\phi(s_1) \\,  
    \\phi\\left(\\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}\\right) 
    \\, F\\left(\\frac{s}{s_1}, \\mu \\right)
```

where ``s`` is the comoving distance, ``\\mu`` the cosine angle,
``\\phi`` is the angular part of the survey window function and ``F(x, μ)`` is the 
window function. Check the documentation of `WindowF` for its definition.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.


## Constructors

     WindowFIntegrated(file::String)

Read the IF map from the file `file`. Such a file might
be produced by `print_map_IntegratedF`, check its docstring. 

It does not matter if the pattern is

```data
# ss      μs      IFs
0.0       -1.0       ...
0.0       -0.9       ...
0.0       -0.8       ...
...       ...      ...
```

or 

```data
# ss      μs      IFs
0.0       -1.0       ...
0.1       -1.0       ...
0.2       -1.0       ...
...       ...      ...
```

because the constructor will recognise it. What does matter is the columns order:
`ss` first, then `μs` and finally `IFs`.


See also: [`integrated_F_trapz`](@ref), [`integrated_F_quadgk`](@ref),
[`spline_integrF`](@ref), [`WindowF`](@ref), [`ϕ`](@ref),
[`print_map_IntegratedF`](@ref)
"""
struct WindowFIntegrated
    ss::Vector{Float64}
    μs::Vector{Float64}
    IFs::Matrix{Float64}

    #=
    function WindowFIntegrated(s_min, s_max, ss::Vector{Float64},
        μs::Vector{Float64}, windowF::WindowF;
        alg::Symbol=:trap, llim=nothing, rlim=nothing,
        rtol=1e-2, atol=0.0, N::Int=1000, pr::Bool=true)

        @assert 0 < s_min < s_max " 0 < s_min < s_max must hold!"
        @assert ss_start ≥ 0.0 " ss_start ≥ 0.0 must hold!"
        @assert 9 < N < 100001 " 10 < N < 100001 must hold!"
        @assert 0.0 ≤ llim < rlim " 0.0 ≤ llim < rlim must hold!"
        @assert all(ss .≥ 0.0) "All ss must be ≥ 0.0!"
        @assert all([ss[i+1] > ss[i] for i in 1:(length(ss)-1)]) "ss must be a float vector of increasing values!"
        @assert all(μs .≥ -1.0) "All μs must be ≥-1.0!"
        @assert all([μs[i+1] > μs[i] for i in 1:(length(μs)-1)]) "μs must be a float vector of increasing values!"
        @assert all(μs .≤ 1.0) "All μs must be ≤1.0!"

        #ss_step=21.768735478453323
        #@assert (iszero(ss_stop) && s_max ≥ ss_start + 3 * ss_step) || (ss_stop ≥ ss_start + 3 * ss_step)
        #" (ss_stop == 0 && s_max ≥ ss_start + 3 * ss_step) || ss_stop ≥ ss_start + 3 * ss_step musty hold!"
        #SS_STOP = iszero(ss_stop) ? 3.0 * s_max : ss_stop
        #ss = [s for s in ss_start:ss_step:SS_STOP]

        IFs = if alg == :trap
            pr ? begin
                @showprogress "calculating intF: " [
                        integrated_F_quadgk(s, μ, s_min, s_max, windowF;
                            llim=llim, rlim=rlim, rtol=rtol, atol=atol)
                        for s in ss, μ in μs]
            end : begin
                [integrated_F_quadgk(s, μ, s_min, s_max, windowF;
                        llim=llim, rlim=rlim, rtol=rtol, atol=atol)
                    for s in ss, μ in μs]
            end
        elseif alg == :quad
            pr ? begin
                @showprogress "calculating intF: " [
                        integrated_F_trapz(s, μ, s_min, s_max, windowF;
                            llim=llim, rlim=rlim, N=N)
                        for s in ss, μ in μs]
            end : begin
                [integrated_F_trapz(s, μ, s_min, s_max, windowF;
                        llim=llim, rlim=rlim, N=N)
                    for s in ss, μ in μs]
            end
        else
            throw(AssertionError("The value 'alg = :$alg' is not a valid algorithm; you must " *
                                "choose between ':trap' and ':quad' . "))
        end

        new(ss, μs, IFs)
    end
    =#

    #=
    function WindowFIntegrated(z_min, z_max, μs::Vector{Float64}, windowF::WindowF,
        file_data::String; names_bg=NAMES_BACKGROUND, h_0=0.7, kwargs...)

        BD = BackgroundData(file_data, z_max; names=names_bg, h=h_0)
        s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
        WindowFIntegrated(s_of_z(z_min), s_of_z(z_max), μs, windowF; kwargs...)
    end
    =#

    function WindowFIntegrated(file::String)
        data = readdlm(file, comments=true)
        ss, μs, IFs = data[:, 1], data[:, 2], data[:, 3]
        @assert size(ss) == size(μs) == size(IFs) "ss, μs and IFs must have the same length!"

        new_ss = unique(ss)
        new_μs = unique(μs)
        new_IFs =
            if ss[2] == ss[1] && μs[2] ≠ μs[1]
                transpose(reshape(IFs, (length(new_μs), length(new_ss))))
            elseif ss[2] ≠ ss[1] && μs[2] == μs[1]
                reshape(IFs, (length(new_ss), length(new_μs)))
            else
                throw(ErrorException("What kind of convenction for the file $file" *
                                        " are you using? I do not recognise it."))
            end
        new(new_ss, new_μs, new_IFs)
    end
end


"""
    spline_integrF(s, μ, str::WindowFIntegrated)::Float64

Return the 2-dim spline value of ``\\mathcal{F}`` in the given `(s,μ)`, where
``\\mathcal{F}`` is defined in the input `WindowFIntegrated`.
The spline is obtained through the `interpolate` function of the 
[`GridInterpolations`](https://github.com/sisl/GridInterpolations.jl) Julia
package.

See also: [`WindowFIntegrated`](@ref)
"""
function spline_integrF(s, μ, str::WindowFIntegrated)
    grid = GridInterpolations.RectangleGrid(str.ss, str.μs)
    GridInterpolations.interpolate(grid, reshape(str.IFs, (:, 1)), [s, μ])
end



"""
    print_map_IntegratedF(out::String, windowFint::WindowFIntegrated)

Print the input Integrated Window Function `windowFint` in the file `out`.

See also: [`WindowFIntegrated`](@ref)
"""
function print_map_IntegratedF(out::String, windowFint::WindowFIntegrated)

    check_parent_directory(out)
    check_namefile(out)

    ss_grid = [s for s in windowFint.ss for μ in windowFint.μs]
    μs_grid = [μ for s in windowFint.ss for μ in windowFint.μs]
    IFs_grid = reshape(transpose(windowFint.IFs), (:,))

    open(out, "w") do io
        println(io, BRAND)
        println(io, "# This is an integration map of the function \\mathcal{F}(s, \\mu), defined as:")
        println(io, "# \\mathcal{F}(s, \\mu) = \\int_0^\\infty dp p^2 \\phi(p) \\phi(\\sqrt{p^2 + s^2 + 2 p s \\mu}) F(s/p, \\mu)")
        println(io, "# where F(x, \\mu) is stored in a WindowF struct (for its analytical definition, check the code.\n#")

        println(io, "#\n# s [h_0^{-1} Mpc] \t mu \t IF")
        for (s, μ, F) in zip(ss_grid, μs_grid, IFs_grid)
            println(io, "$s\t $μ \t $F")
        end
    end
end


##########################################################################################92
