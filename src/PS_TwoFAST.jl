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
    TwoFAST_PS_multipole(f_in;
        int_s_min::Float64 = 1e-1, int_s_max::Float64 = 1e3,
        L::Int = 0, N::Int = 1024, pr::Bool = true,
        k0::Union{Nothing,Float64} = nothing,
        right::Union{Float64,Nothing} = nothing
    ) ::Tuple{Vector{Float64}, Vector{Float64}}

Computes the Power Spectrum from the input spline `f_in` through the TwoFAST `xicalc` 
function of the [TwoFAST](https://github.com/hsgg/TwoFAST.jl) Julia package.
More precisely, it computes the `L`-order PS multipole through the
following Fast Fourier Transform and the effective redshift approximation

```math
P_L(k) = \\frac{2 L + 1}{A^{'}} (-i)^L \\, \\phi(s_\\mathrm{eff}) \\int_0^\\infty 
    \\mathrm{d} s \\; s^2 \\, j_L(ks) \\, f_\\mathrm{in}(s) \\; ,
    \\quad \\; A^{'} = \\frac{1}{4\\,\\pi}
```

where ``f_\\mathrm{in}`` is the inpunt spline.


## Optional arguments

- `pr::Bool=true` : want to print the automatic messages to the screen?
- `L::Int=0` : which multipole order should I use for this computation? IT MUST MATCH 
  THE MULTIPOLE ORDER OF THE INPUT TPCF!
- `N::Int = 1024` : number of points to be used in Fourier transform 
- `int_s_min::Float64 = 1e-1` and `int_s_max::Float64 = 1e3`: extremes of integration
- `k0::Union{Nothing,Float64} = nothing` : starting point for the `xicalc` function; if `nothing`, 
  it will be set `k0 = 1.0 / int_s_max`
- `right::Union{Float64,Nothing} = nothing` : do you want to cut the output elements with 
  `ks .> right`? if set to `nothing`, no cut will be done.

See also: [`PS_multipole`](@ref)
"""
function TwoFAST_PS_multipole(f_in;
    int_s_min::Float64=1e-1, int_s_max::Float64=1e3,
    L::Int=0, N::Int=1024, pr::Bool=true,
    k0::Union{Nothing,Float64}=nothing,
    right::Union{Float64,Nothing}=nothing)


    k_0 = isnothing(k0) ? 1.0 / int_s_max : k0

    t1 = time()
    ks, pks = xicalc(s -> 2 * π^2 * f_in(s), L, 0;
        N=N, kmin=int_s_min, kmax=int_s_max, r0=k_0)
    t2 = time()
    pr && println("\ntime needed for this Power Spectrum computation [in s] = $(t2-t1)\n")

    if isnothing(right)
        if iseven(L)
            return ks, (1 / A_prime * (-1)^(L / 2)) .* pks
        else
            return ks, (1 / A_prime * (-im)^L) .* pks
        end
    else
        if iseven(L)
            return ks[ks.<right], (1 / A_prime * (-1)^(L / 2)) .* pks[ks.<right]
        else
            return ks[ks.<right], (1 / A_prime * (-im)^L) .* pks[ks.<right]
        end
    end
end



"""
    TwoFAST_PS_multipole(ss, fs;
        int_s_min::Float64 = 1e-1, int_s_max::Float64 = 1e3,
        epl::Bool = true, pr::Bool = true, L::Int = 0,
        N_left::Int = 12, N_right::Int = 12,
        p0_left = [-2.0, 1.0], p0_right = [-2.0, 1.0],
        k0::Union{Nothing,Float64} = nothing
    ) ::Tuple{Vector{Float64}, Vector{Float64}}

Takes the input data vector `ss` and `fs` and creates a spline from them, passing it
as input tho the other `TwoFAST_PS_multipole` method. Depending on the options, it may
create also a power law epansions on the edges.


## Optional arguments

- `pr::Bool=true` : want to print the automatic messages to the screen?
- `L::Int=0` : which multipole order should I use for this computation? IT MUST MATCH 
  THE MULTIPOLE ORDER OF THE INPUT TPCF!
- `epl::Bool=true` : do you want to extend the edges of the input vectors using two fitted
  power-laws (obtained from `EPLs`)
- `N_left::Int = 12` and `N_right::Int = 12` : number of points from left
  right edges to be used for the power law fitting in `EPLs`. They matters only
  if in the given input file ξ is not defined until the extremes of integration
  `int_s_min` and `int_s_max`.
- `int_s_min::Float64 = 1e-1` and `int_s_max::Float64 = 1e3`: extremes of integration; if `epl`
  is set to `false`, their values will be automatically set to `min(ss...)` and `max(ss...)`
  respectively. Their values do matter only if `epl=true`. 
- `p0_left=[-2.0, 1.0]` and `p0_right=[-2.0, 1.0]`: vectors with the initial values for the left/right 
  power-law fitting of `EPLs`; the power-law is in the form ``y = f(x) = b * x^s``, so the first vector 
  value is the initial value of ``s`` (and of course the second is the one of ``b``).
- `k0::Union{Nothing,Float64} = nothing` : starting point for the `xicalc` function; if `nothing`, 
  it will be set `k0 = 1.0 / max(ss...)`
- `right::Union{Float64,Nothing} = nothing` : do you want to cut the output elements with 
  `ks .> right`? if set to `nothing`, no cut will be done.
- `N::Int = 1024` : number of points to be used in Fourier transform 
- `cut_first_n::Int=0` and `cut_last_n::Int=0` : you can cout the first and/or last n elements
  of the input data, if they are highly irregular.

See also: [`PS_multipole`](@ref)
"""
function TwoFAST_PS_multipole(SS, FS;
        int_s_min::Float64=1e-1, int_s_max::Float64=1e3,
        epl::Bool=true, pr::Bool=true, L::Int=0,
        N_left::Int=12, N_right::Int=12,
        p0_left=[-2.0, 1.0], p0_right=[-2.0, 1.0],
        k0::Union{Nothing,Float64}=nothing,
        cut_first_n::Int = 0, cut_last_n::Int=0
    )

    @assert length(SS) == length(FS) "length(ss) == length(fs) must hold!"
    @assert cut_first_n ≥ 0 "cut_first_n ≥ 0 must hold!"
    @assert cut_last_n ≥ 0 "cut_last_n ≥ 0 must hold!"
    @assert cut_first_n + cut_last_n < length(SS) "cut_first_n + cut_last_n < length(ss) must hold!"

    a, b = 1 + cut_first_n, length(SS) - cut_last_n
    ss, fs = SS[a:b], FS[a:b]

    N = length(ss)
    k_0 = isnothing(k0) ? 1.0 / max(ss...) : k0
    right = 1.0 / min(ss...)

    f_in, INT_s_min, INT_s_max =
        if epl == true
            if all(fs[begin:begin+5] .≈ 0.0) || all(fs[end-5:end] .≈ 0.0)
                spl = GaPSE.MySpline(ss, fs; bc="error")
                f(x) = ((x ≤ ss[1]) || (x ≥ ss[end])) ? 0.0 : spl(x)
                f, int_s_min, int_s_max
            else
                EPLs(ss, fs, p0_left, p0_right;
                        N_left=N_left, N_right=N_right), int_s_min, int_s_max
            end
        else
            GaPSE.MySpline(ss, fs; bc="error"), min(ss...), max(ss...)
        end

    return TwoFAST_PS_multipole(f_in; int_s_min=INT_s_min, int_s_max=INT_s_max,
        k0=k_0, right=right, N=N, pr=pr, L=L)
end



"""
    TwoFAST_all_PS_multipole(input::String,
        group::String=VALID_GROUPS[end];
        L::Int = 0, pr::Bool = true, 
        kwargs...)

Computes the Power Spectrum through the TwoFAST `xicalc` function of the 
[TwoFAST](https://github.com/hsgg/TwoFAST.jl) Julia package for a set of TPCFs. 
More precisely, it read the input file `input`, taking the first
column as the x-axis `ss` vector and the following columns as the y-axis ones, and computes 
the `L`-order PS multipole through the
following Fast Fourier Transform and the effective redshift approximation

```math
P_L(k) = \\frac{2 L + 1}{A^{'}} (-i)^L \\, \\phi(s_\\mathrm{eff}) \\int_0^\\infty 
    \\mathrm{d} s \\; s^2 \\, j_L(ks) \\, f_\\mathrm{in}(s) \\; ,
    \\quad \\; A^{'} = \\frac{1}{4\\,\\pi}
```

where ``f_\\mathrm{in}`` is the function samples by `ss` and each y-axis xis.

The `group::String=VALID_GROUPS[end]` argument allow you to specify the group of the input TPCF, 
if they were computed through GAPSE. The allowed values for this argument are:
`$(string(GaPSE.VALID_GROUPS .* " , "...))`

If you choose a group pay attention that the number of input TPCF must match the group number 
(16, 25, 20 and 20 respectively). The last group name (which is also the default value) is used in 
case the input xis do not belog to a specific group (and so no predefined number is expected).


## Optional arguments

- `pr::Bool=true` : want to print the automatic messages to the screen?
- `L::Int=0` : which multipole order should I use for this computation? IT MUST MATCH 
  THE MULTIPOLE ORDER OF THE INPUT TPCF!
- `epl::Bool=true` : do you want to extend the edges of the input vectors using two fitted
  power-laws (obtained from `EPLs`)
- `N_left::Int = 12` and `N_right::Int = 12` : number of points from left
  right edges to be used for the power law fitting in `EPLs`. They matters only
  if in the given input file ξ is not defined until the extremes of integration
  `int_s_min` and `int_s_max`.
- `int_s_min::Float64 = 1e-1` and `int_s_max::Float64 = 1e3`: extremes of integration; if `epl`
  is set to `false`, their values will be automatically set to `min(ss...)` and `max(ss...)`
  respectively. Their values do matter only if `epl=true`. 
- `p0_left=[-2.0, 1.0]` and `p0_right=[-2.0, 1.0]`: vectors with the initial values for the left/right 
  power-law fitting of `EPLs`; the power-law is in the form ``y = f(x) = b * x^s``, so the first vector 
  value is the initial value of ``s`` (and of course the second is the one of ``b``).
- `k0::Union{Nothing,Float64} = nothing` : starting point for the `xicalc` function; if `nothing`, 
  it will be set `k0 = 1.0 / max(ss...)`
- `right::Union{Float64,Nothing} = nothing` : do you want to cut the output elements with 
  `ks .> right`? if set to `nothing`, no cut will be done.
- `N::Int = 1024` : number of points to be used in Fourier transform 
- `cut_first_n::Int=0` and `cut_last_n::Int=0` : you can cout the first and/or last n elements
  of the input data, if they are highly irregular.

See also: [`TwoFAST_PS_multipole`](@ref), [`PS_multipole`](@ref)
"""
function TwoFAST_all_PS_multipole(input::String,
    group::String=VALID_GROUPS[end];
    L::Int=0, pr::Bool=true, kwargs...)

    check_group(group; valid_groups=VALID_GROUPS)
    check_fileisingroup(input, group)

    pr && begin
        print("\nI'm computing the PS_multipole from the file \"$input\"")
        if group == "GNC"
            println("for the Galaxy Number Counts.")
        elseif group == "LD"
            println("for the Luminosity Distance perturbations.")
        elseif group == "GNCxLD"
            println("for the cross correlations between " *
                    "Galaxy Number Counts and Luminosity Distance perturbations.")
        elseif group == "LDxGNC"
            println("for the cross correlations between " *
                    "Luminosity Distance perturbations and Galaxy Number Counts.")
        else
            println("(no specific group considered).")
        end
    end

    sps = group == "GNC" ? "GNC GR effects" :
        group == "LD" ? "LD GR effects" :
        group == "GNCxLD" ? "GNCxLD GR effects" :
        group == "LDxGNC" ? "LDxGNC GR effects" :
        "generic file"

    ks, VEC = begin
        table = readdlm(input; comments=true)
        xs = convert(Vector{Float64}, table[:, 1])
        all_YS = [convert(Vector{Float64}, col)
                for col in eachcol(table[:, 2:end])]
        res = pr ? begin
            @showprogress sps * ", L=$L: " [
                TwoFAST_PS_multipole(xs, ys; L=L, pr=false, kwargs...)
                for ys in all_YS]
        end : begin
            [TwoFAST_PS_multipole(xs, ys; L=L, pr=false, kwargs...)
            for ys in all_YS]
        end

        res[1][1], [res[i][2] for i in 1:length(res)]
    end

    return ks, VEC
end
