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
    InputPS(
        l_si::Float64
        l_b::Float64
        l_a::Float64
        left::Float64

        spline::GaPSE.MySpline
        
        r_si::Float64
        r_b::Float64
        r_a::Float64
        right::Float64)

Store the Input Power Spectrum.

## Arguments 

- `l_si, l_b, l_a :: Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the LEFT edge; when an input value `x < left` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `l_si` is the exponent, `l_b` the coefficient and `l_a` the 
  spurious adding constant). 

- `left::Float64` : the break between the left power-law (for `x <left`) and the 
  spline (for `x ≥ left`); its value is the `fit_min` of the used constructor.

- `spline::GaPSE.MySpline` : spline that interpolates between the real values of the 
  power spectrum inside the range `left ≤ x ≤ right`

- `right::Float64` : the break between the right power-law (for `x > left`) and the 
  spline (for `x ≤ right`); its value is the `fit_max` of the used constructor.

- `r_si, r_b, r_a :: Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the RIGHT edge; when an input value `x > right` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `r_si` is the exponent, `r_b` the coefficient and `r_a` the 
  spurious adding constant). 
  NOTE: for numerical issues, only the "pure" power-law ``y = f(x) = b + x^s`` can be used. 
  In other words, it always set `r_a = 0.0`.


## Constructors

- `InputPS(file::String; fit_left_min = 1e-6, fit_left_max = 3e-6, fit_right_min = 1e1, fit_right_max = 2e1)` : read the IPS from
  the given input `file`; it can contain comments (defined with a 
  starting `#` on each line), but the file structure must be space-separated in two colums
  (former for `k` values, latter for `P` ones).

  - `fit_left_min = 1e-6, fit_left_max = 3e-6` : the limits (min and max) where the PS
    must be fitted with a power law, for small wavenumbers. 

  - `fit_right_min = 1e1, fit_right_max = 2e1` : the limits (min and max) where the PS
    must be fitted with a power law, for high wavenumbers. 


- `InputPS(ks::AbstractVector{T1}, pks::AbstractVector{T2}; fit_left_min = 1e-6, fit_left_max = 3e-6,
  fit_right_min = 1e1, fit_right_max = 2e1)`

  - `ks::AbstractVector{T1}, pks::AbstractVector{T2}` : self-explanatory `ks` and `pks`
    array-like values.

  - `fit_left_min = 1e-6, fit_left_max = 3e-6` : the limits (min and max) where the PS
    must be fitted with a power law, for small wavenumbers. 

  - `fit_right_min = 1e1, fit_right_max = 2e1` : the limits (min and max) where the PS
    must be fitted with a power law, for high wavenumbers. 


  
All the power-law fitting (both "pure" and spurious) are made through the 
local function `power_law_from_data`.

See also: [`power_law_from_data`](@ref)
"""
struct InputPS
    l_si::Float64
    l_b::Float64
    l_a::Float64
    left::Float64

    spline::GaPSE.MySpline

    r_si::Float64
    r_b::Float64
    r_a::Float64
    right::Float64


    function InputPS(file::String; fit_left_min=1e-6, fit_left_max=3e-6,
        fit_right_min=1e1, fit_right_max=2e1)

        data = readdlm(file, comments=true)
        @assert size(data[:, 1]) == size(data[:, 2]) "ks and pks must have the same length!"

        ks = convert(Vector{Float64}, data[:, 1])
        pks = convert(Vector{Float64}, data[:, 2])

        l_si, l_b, l_a = power_law_from_data(
            ks, pks, [1.0, 1.0], fit_left_min, fit_left_max; con=false)

        r_si, r_b, r_a = power_law_from_data(
            ks, pks, [-3.0, 1.0], fit_right_min, fit_right_max; con=false)

        ind_left = findfirst(x -> x > fit_left_min, ks) - 1
        ind_right = findfirst(x -> x >= fit_right_max, ks)
        new_ks = vcat(ks[ind_left:ind_right])
        new_pks = vcat(pks[ind_left:ind_right])
        spline = GaPSE.MySpline(new_ks, new_pks; bc="error")


        new(l_si, l_b, l_a, fit_left_min, spline, r_si, r_b, r_a, fit_right_max)
    end

    function InputPS(ks::AbstractVector{T1}, pks::AbstractVector{T2};
        fit_left_min=1e-6, fit_left_max=3e-6,
        fit_right_min=1e1, fit_right_max=2e1) where {T1,T2}

        @assert size(ks) == size(pks) "ks and pks must have the same length!"

        l_si, l_b, l_a = power_law_from_data(
            ks, pks, [1.0, 1.0], fit_left_min, fit_left_max; con=false)

        r_si, r_b, r_a = power_law_from_data(
            ks, pks, [-3.0, 1.0], fit_right_min, fit_right_max; con=false)

        ind_left = findfirst(x -> x > fit_left_min, ks) - 1
        ind_right = findfirst(x -> x >= fit_right_max, ks)
        new_ks = vcat(ks[ind_left:ind_right])
        new_pks = vcat(pks[ind_left:ind_right])
        spline = GaPSE.MySpline(new_ks, new_pks; bc="error")

        new(l_si, l_b, l_a, fit_left_min, spline, r_si, r_b, r_a, fit_right_max)
    end
end


"""
    (f::InputPS)(x)

Return the value of the `f::InputPS` as follows:
```math
f(x)=
\\begin{cases}
a_\\mathrm{L} + b_\\mathrm{L} \\, x ^ {s_\\mathrm{L}} \\; ,
    \\quad x < \\mathrm{left}\\\\
\\mathrm{spline}(x) \\; , \\quad \\mathrm{left} \\leq x \\leq \\mathrm{right} \\\\
a_\\mathrm{R} + b_\\mathrm{R} \\, x ^ {s_\\mathrm{R}} \\; , 
\\quad x > \\mathrm{right}
\\end{cases}
```

where ``a_\\mathrm{L}``, ``b_\\mathrm{L}``, ``s_\\mathrm{L}``, ``\\mathrm{left}``,
``\\mathrm{spline}``, ``a_\\mathrm{R}``, ``b_\\mathrm{R}``, ``s_\\mathrm{R}`` and 
``\\mathrm{right}`` are all stored inside the `InputPS` considered.

See also: [`InputPS`](@ref)
"""
function (IPS::InputPS)(x)
    if x < IPS.left
        return power_law(x, IPS.l_si, IPS.l_b, IPS.l_a)
    elseif x > IPS.right
        return power_law(x, IPS.r_si, IPS.r_b, IPS.r_a)
    else
        return IPS.spline(x)
    end
end



##########################################################################################92



"""
    IntegralIPS(
        l_si::Float64
        l_b::Float64
        l_a::Float64
        left::Float64

        spline::GaPSE.MySpline

        r_si::Float64
        r_b::Float64
        r_a::Float64
        right::Float64
    )

Contains all the information useful in order to return the value of an integral
obtained from the Input Power Spectrum.

## Arguments 

- `l_si, l_b, l_a ::Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the LEFT edge; when an input value `x < left` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `l_si` is the exponent, `l_b` the coefficient and `l_a` the 
  spurious adding constant). 

- `left::Float64` : the break between the left power-law (for `x < left`) and the 
  spline (for `x ≥ left`); its value is the `fit_min` of the used constructor.

- `spline::GaPSE.MySpline` : spline that interpolates between the real values of the 
  integral calculated inside the range `left ≤ x ≤ right`

- `right::Float64` : the break between the right power-law (for `x > right`) and the 
  spline (for `x ≤ right`); its value is the `fit_max` of the used constructor.

- `r_si, r_b, r_a ::Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the RIGHT edge; when an input value `x > right` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `r_si` is the exponent, `r_b` the coefficient and `r_a` the 
  spurious adding constant). 
  NOTE: for numerical issues, only the "pure" power-law ``y = f(x) = b + x^s`` can be used. 
  In other words, it always set `r_a = 0.0`.

## Constructors

There are two type of integrals we are interested in, and so two constructors are
here provided:

- `IntegralIPS(ips, l, n; N::Int = 1024, kmin = 1e-4, kmax = 1e3, s0 = 1e-3,
    fit_left_min = 2.0, fit_left_max = 10.0, p0_left = nothing, con = false,
    fit_right_min = nothing, fit_right_max = nothing, p0_right = nothing)`
  This is the one used for the "classical" ``I_\\ell_n`` integrals:
  ```math
  I_\\ell^n(s) = \\int_0^\\infty \\frac{\\mathrm{d} q}{2 \\pi^2} q^2 \\, P(q) 
    \\, \\frac{j_\\ell(qs)}{(qs)^n}
  ```
  where, for a generic `Iab` name, ``\\ell`` is the FIRST number (`a`) and 
  ``n`` the second (`b`).
  The integral obtained with this constructor is calculated through `xicalc`, and
  expanded with power-laws at the edges.
    
  - `ips`: the function/spline that gives the Input Power Spectrum
    
  - `l` and `n`: self-explanatory degree of the integral, with the convenction
    above mentioned
    
  - `kmin = 1e-4, kmax = 1e3, s0 = 1e-3` : values to be passed to `xicalc` for the
    integration
    
  - `fit_left_min = 2.0, fit_left_max = 10.0` : the limits (min and max) where the integral ``I_\\ell^n``
    must be fitted with a power law, for small distances. This operation is necessary, because `xicalc`,
    in this context, gives wrong results for too small input distance `s`; nevertheless, all
    these ``I_\\ell^n`` integrals have fixed power-law trends for ``s \\rightarrow 0``, so this approach gives
    good results.
    
  - `p0_left = nothing` : vector with the initial values for the left power-law fitting; its length must
    be 2 (if you want to fit with a pure power-law ``y = f(x) = b * x^s``, so only `l_si` and `l_b` 
    are matter of concern) or 3 (if you want to fit with a spurious power-law ``y = f(x) = a + b * x^s``,
    so you are also interested in `l_a`), depending on the value of `con`; if `nothing`, it will be
    automatically set `p0 = [-1.0, 1.0, 0.0]` for `con==true` and
    `p0 = [-1.0, 1.0]` for `con==false`.
    
  - `con::Bool = false` : do you want that the fit of all the ``I_\\ell^n`` for the LEFT edge
    is not a simple power-law ``y = f(x) = b \\, x^s``, but also consider a constant ``a``,
    such that ``y = f(x) = a + b \\, x^s``?
    For the LEFT side, there is not a lot of difference empirically. 
    For the RIGHT side, there is not such an option due to numerical problems (it's like 
    is always set `con==false`).
    
  - `fit_right_min = nothing, fit_right_max = nothing` : the limits (min and max) where the integral ``I_\\ell^n``
    must be fitted with a power law, for high distances. 
    These ``I_\\ell^n`` integrals have fixed power-law trends for ``s \\rightarrow \\infty``, so this approach gives
    good results. If `nothing`, the last 15 points returned from `xicalc` are used for
    this fitting.
    NOTE: for numerical issues, only the "pure" power-law ``y = f(x) = b + x^s`` can be used. 
    
  - `p0_right = nothing` : vector with the initial values for the left power-law fitting; its length must
    be 2 (to fit with a pure power-law ``y = f(x) = b * x^s``, so only `r_si` and `r_b` 
    are matter of concern); if `nothing`, it will be
    automatically set `p0 = [-4.0, 1.0, 0.0]`.


- `IntegralIPS(ips, func::Function; N::Int = 1024, kmin = 1e-4, kmax = 1e3,
    fit_left_min = 0.1, fit_left_max = 1.0, p0_left = nothing, con = false,
    fit_right_min = nothing, fit_right_max = nothing, p0_right = nothing,
    kwargs...)`
  This is the one used for the "strange" ``\\tilde{I}`` integrals, such as:
  ```math
  \\tilde{I}^4_0 (s) = \\int \\frac{\\mathrm{d}q}{2\\pi^2} \\, q^2 \\, 
     P(q) \\,  \\frac{j_0(qs) - 1}{(qs)^4} \\;.
  ```
  The integral obtained with this constructor is calculated through the
  input function `func`, and expanded with power-laws at the edges.
  For `\\tilde{I}^4_0`, the function is `func_I04_tilde`.  

  - `ips`: the function/spline that gives the Input Power Spectrum

  - `func`: function that return the value of this specific integral in a given value

  - `kmin = 1e-4, kmax = 1e3, s0 = 1e-3` : values to be passed to `func` as extremes
    of integration

  - `fit_left_min = 0.1, fit_left_max = 1.0,` : the limits (min and max) where the integral ``\\tilde{I}``
    must be fitted with a power law, for small distances. This operation is necessary, because `xicalc`,
    in this context, gives wrong results for too small input distance `s`; nevertheless, all
    this ``\\tilde{I}`` integral have fixed power-law trends for ``s \\rightarrow 0``, so this approach gives
    good results.

  - `p0_left = nothing` : vector with the initial values for the left power-law fitting; its length must
    be 2 (if you want to fit with a pure power-law ``y = f(x) = b * x^s``, so only `l_si` and `l_b` 
    are matter of concern) or 3 (if you want to fit with a spurious power-law ``y = f(x) = a + b * x^s``,
    so you are also interested in `l_a`), depending on the value of `con`; if `nothing`, it will be
    automatically set `p0 = [-2.0, -1.0, 0.0]` for `con==true` and
    `p0 = [-2.0, -1.0]` for `con==false`.

  - `con::Bool = false` : do you want that the fit of all the ``I_\\ell^n`` for the LEFT edge
    is not a simple power-law ``y = f(x) = b \\, x^s``, but also consider a constant ``a``,
    such that ``y = f(x) = a + b \\, x^s``?
    For the LEFT side, there is not a lot of difference empirically. 
    For the RIGHT side, there is not such an option due to numerical problems (it's like 
    is always set `con==false`).

  - `fit_right_min = nothing, fit_right_max = nothing` : the limits (min and max) where the integral ``I_\\ell^n``
    must be fitted with a power law, for high distances. 
    These ``I_\\ell^n`` integrals have fixed power-law trends for ``s \\rightarrow \\infty``, so this approach gives
    good results. If `nothing`, the last 15 points returned from `xicalc` are used for
    this fitting.
    NOTE: for numerical issues, only the "pure" power-law ``y = f(x) = b + x^s`` can be used. 

  - `p0_right = nothing` : vector with the initial values for the left power-law fitting; its length must
    be 2 (to fit with a pure power-law ``y = f(x) = b * x^s``, so only `r_si` and `r_b` 
    are matter of concern); if `nothing`, it will be
    automatically set `p0 = [-4.0, -1.0]`.


All the power-law fitting (both "pure" and spurious) are made through the 
local function `power_law_from_data`.

See also: [`power_law_from_data`](@ref), [`power_law`](@ref),
[`func_I04_tilde`](@ref)
"""
struct IntegralIPS
    l_si::Float64
    l_b::Float64
    l_a::Float64
    left::Float64

    spline::GaPSE.MySpline

    r_si::Float64
    r_b::Float64
    r_a::Float64
    right::Float64

    function IntegralIPS(ips, l, n; N::Int=1024, kmin=1e-4, kmax=1e3, s0=1e-3,
        fit_left_min=2.0, fit_left_max=10.0, p0_left=nothing, con=false,
        fit_right_min=nothing, fit_right_max=nothing, p0_right=nothing)

        rs, xis = xicalc(ips, l, n; N=N, kmin=kmin, kmax=kmax, r0=s0)

        p_0_left = isnothing(p0_left) ? (con == true ? [-1.0, 1.0, 0.0] : [-1.0, 1.0]) : p0_left
        l_si, l_b, l_a = power_law_from_data(
            rs, xis, p_0_left, fit_left_min, fit_left_max; con=con)

        fit_right_MIN = isnothing(fit_right_min) ? rs[length(rs)-16] : fit_right_min
        fit_right_MAX = isnothing(fit_right_max) ? rs[length(rs)-1] : fit_right_max
        p_0_right = isnothing(p0_right) ? [-4.0, 1.0] : p0_right
        r_si, r_b, r_a = power_law_from_data(
            rs, xis, p_0_right, fit_right_MIN, fit_right_MAX; con=false)

        ind_left = findfirst(x -> x > fit_left_min, rs) - 1
        ind_right = findfirst(x -> x >= fit_right_MAX, rs)
        new_rs = vcat(rs[ind_left:ind_right])
        new_Is = vcat(xis[ind_left:ind_right])
        spline = GaPSE.MySpline(new_rs, new_Is; bc="error")

        #println("\nleft = $l_si , $l_b , $l_a, $fit_left_min")
        #println("right = $r_si , $r_b , $r_a, $fit_right_MAX\n")

        new(l_si, l_b, l_a, fit_left_min, spline, r_si, r_b, r_a, fit_right_MAX)
    end

    function IntegralIPS(ips, func::Function; N::Int=1024, kmin=1e-4, kmax=1e3,
        fit_left_min=0.1, fit_left_max=1.0, p0_left=nothing, con=false,
        fit_right_min=nothing, fit_right_max=nothing, p0_right=nothing,
        kwargs...)

        ss = 10 .^ range(log10(0.999 * fit_left_min), 4, length=N)
        Is = [func(ips, s, kmin, kmax; kwargs...) for s in ss]

        p_0_left = isnothing(p0_left) ? (con == true ? [-2.0, -1.0, 0.0] : [-2.0, -1.0]) : p0_left
        l_si, l_b, l_a = power_law_from_data(
            ss, Is, p_0_left, fit_left_min, fit_left_max; con=con)

        fit_right_MIN = isnothing(fit_right_min) ? ss[length(ss)-13] : fit_right_min
        fit_right_MAX = isnothing(fit_right_max) ? ss[length(ss)-1] : fit_right_max
        p_0_right = isnothing(p0_right) ? [-4.0, -1.0] : p0_right
        r_si, r_b, r_a = power_law_from_data(
            ss, Is, p_0_right, fit_right_MIN, fit_right_MAX; con=false)

        #println("\nLEFT = $l_si , $l_b , $l_a, $fit_left_min")
        #println("RIGHT = $r_si , $r_b , $r_a, $fit_right_MAX\n")

        spline = GaPSE.MySpline(ss, Is; bc="error")

        new(l_si, l_b, l_a, fit_left_min, spline, r_si, r_b, r_a, fit_right_MAX)
    end

    #=
    function IntegralIPS(xs::Vector{Float64}, ys::Vector{Float64}; N = 1024, kmin = 1e-4, kmax = 1e3,
        fit_left_min = 0.1, fit_left_max = 1.0, p0_left = nothing, con = false, 
        fit_right_min = nothing, fit_right_max = nothing, p0_right = nothing,
        kwargs...)

        ss = 10 .^ range(log10(0.999*fit_left_min), 4, length = 1024)
        Is = [func(ips, s, kmin, kmax; kwargs...) for s in ss]

        p_0_left = isnothing(p0_left) ? (con == true ? [-2.0, -1.0, 0.0] : [-2.0, -1.0]) : p0_left
        l_si, l_b, l_a = power_law_from_data(
            ss, Is, p_0_left, fit_left_min, fit_left_max; con = con)

        fit_right_MIN = isnothing(fit_right_min) ? ss[length(ss)-13] : fit_right_min
        fit_right_MAX = isnothing(fit_right_max) ? ss[length(ss)-1] : fit_right_max
        p_0_right = isnothing(p0_right) ? [-4.0, -1.0] : p0_right
        r_si, r_b, r_a = power_law_from_data(
            ss, Is, p_0_right, fit_right_MIN, fit_right_MAX; con = false)

        #println("\nLEFT = $l_si , $l_b , $l_a, $fit_left_min")
        #println("RIGHT = $r_si , $r_b , $r_a, $fit_right_MAX\n")

        spline = GaPSE.MySpline(ss, Is; bc="error")

        new(l_si, l_b, l_a, fit_left_min, spline, r_si, r_b, r_a, fit_right_MAX)
    end
    =#

end


"""
    (f::IntegralIPS)(x)

Return the value of the `f::IntegralIPS` as follows:
```math
f(x)=
\\begin{cases}
a_\\mathrm{L} + b_\\mathrm{L} \\, x ^ {s_\\mathrm{L}} \\; ,
    \\quad x < \\mathrm{left}\\\\
\\mathrm{spline}(x) \\; , \\quad \\mathrm{left} \\leq x \\leq \\mathrm{right} \\\\
a_\\mathrm{R} + b_\\mathrm{R} \\, x ^ {s_\\mathrm{R}} \\; , 
\\quad x > \\mathrm{right}
\\end{cases}
```

where ``a_\\mathrm{L}``, ``b_\\mathrm{L}``, ``s_\\mathrm{L}``, ``\\mathrm{left}``,
``\\mathrm{spline}``, ``a_\\mathrm{R}``, ``b_\\mathrm{R}``, ``s_\\mathrm{R}`` and 
``\\mathrm{right}`` are all stored inside the `IntegralIPS` considered.

See also: [`IntegralIPS`](@ref)
"""
function (Iln::IntegralIPS)(x)
    if x < Iln.left
        return power_law(x, Iln.l_si, Iln.l_b, Iln.l_a)
    elseif x > Iln.right
        #warning("i am going too right! ")
        return power_law(x, Iln.r_si, Iln.r_b, Iln.r_a)
    else
        return Iln.spline(x)
    end
end



##########################################################################################92



"""
    IPSTools(
        I00::IntegralIPS
        I20::IntegralIPS
        I40::IntegralIPS
        I02::IntegralIPS
        I22::IntegralIPS
        I31::IntegralIPS
        I13::IntegralIPS
        I11::IntegralIPS

        I04_tilde::IntegralIPS

        σ_0::Float64
        σ_1::Float64
        σ_2::Float64
        σ_3::Float64
        σ_4::Float64

        fit_min::Union{Float64,Nothing}
        fit_max::Union{Float64,Nothing}
        k_min::Float64
        k_max::Float64
        s_0::Float64
        )

Struct that contains all the useful functions and values obtained from the 
Input Power Spectrum.

## Arguments

- `I00, I20, I40, I02, I22, I31, I13, I11 ::IntegralIPS`: they return
  the value of the corresponding integral:

  ```math
  I_\\ell^n(s) = \\int_0^\\infty \\frac{\\mathrm{d} q}{2 \\pi^2} q^2 \\, P(q) 
    \\, \\frac{j_\\ell(qs)}{(qs)^n}
  ```
  where, for a generic `Iab` name, ``\\ell`` is the FIRST number (`a`) and 
  ``n`` the second (`b`).
  These integrals are performed through `xicalc`, with `kmin, kmax, s0 = 1e-5, 1e3, 1e-3`;
  at the edges they are fitted with power laws (for `s < fit_min` and 
  `s > max_s_returned_from_xi_calc`).

- `I04_tilde::IntegralIPS`: it returns the value of the integral:

  ```math
  \\tilde{I}^4_0 (s) = \\int \\frac{\\mathrm{d}q}{2\\pi^2} \\, q^2 \\, 
     P(q) \\,  \\frac{j_0(qs) - 1}{(qs)^4} \\;.
  ```
  This integral is calculated brute-force with `quadgk`, and fitted with power-laws
  at the edges (for `s < 0.1` and `s > 1e4`).

- `σ_0, σ_1, σ_2, σ_3, σ_4 :: Float64`: these are the results of the following integral:
  ```math
  \\sigma_i = \\int_{k_\\mathrm{min}}^{k_\\mathrm{max}} \\frac{\\mathrm{d} q}{2 \\pi^2} \\, q^{2-i} \\, P(q)
  ```

- `fit_min, fit_max :: Float64`: the limits (min and max) where the integral ``I_\\ell^n``
  must be fitted with a power law, for small distances. This operation is necessary, because `xicalc`,
  in this context, gives wrong results for too small input distance `s`; nevertheless, all
  these ``I_\\ell^n`` integrals have fixed power-law trends for ``s \\rightarrow 0``, so this approach gives
  good results.

- `k_min k_max::Float64` : because some of the ``\\sigma_i`` integrals from ``q = 0`` to
  ``q = +\\infty`` diverge, it is common practice to cut the integrals at the edges, so they
  are calculated from ``q = k_\\mathrm{min}`` to ``q = k_\\mathrm{max}``


## Constructors

    IPSTools(ips::InputPS; N::Int = 1024,
        fit_min::Float64 = 0.05, fit_max::Float64 = 0.5,
        k_min::Float64 = 1e-6, k_max::Float64 = 10.0
        con::Bool = false
    )

- `ips::InputPS` : the Input Power Spectrum to be used in all the calculations.

- `N::Int = 1024` : number of points to be used in the `xicalc` function

- `k_min::Float64 = 1e-6, k_max::Float64 = 10.0` : integrations extremes of 
  the ``\\sigma_i``s

- `con::Bool = false` : do you want that the fit of all the ``I_\\ell^n`` for the LEFT edge
  is not a simple power-law ``y = f(x) = b \\, x^s``, but also consider a constant ``a``,
  such that ``y = f(x) = a + b \\, x^s``?
  For the LEFT side, there is not a lot of difference empirically. 
  For the RIGHT side, there is not such an option due to numerical problems (it's like 
  is always set `con==false`).


See also: [`IntegralIPS`](@ref), [`InputPS`](@ref)
"""
struct IPSTools
    #=
    I00::GaPSE.MySpline
    I20::GaPSE.MySpline
    I40::GaPSE.MySpline
    I02::GaPSE.MySpline
    I22::GaPSE.MySpline
    I31::GaPSE.MySpline
    I13::GaPSE.MySpline
    I11::GaPSE.MySpline

    I04_tilde::GaPSE.MySpline
    =#

    I00::IntegralIPS
    I20::IntegralIPS
    I40::IntegralIPS
    I02::IntegralIPS
    I22::IntegralIPS
    I31::IntegralIPS
    I13::IntegralIPS
    I11::IntegralIPS

    I04_tilde::IntegralIPS

    σ_0::Float64
    σ_1::Float64
    σ_2::Float64
    σ_3::Float64
    σ_4::Float64

    fit_min::Float64
    fit_max::Float64
    k_min::Float64
    k_max::Float64

    function IPSTools(
        ips::InputPS;
        N::Int=1024,
        fit_min::Float64=0.05,
        fit_max::Float64=0.5,
        con::Bool=false,
        k_min::Float64=1e-6,
        k_max::Float64=10.0
    )
        #PK = GaPSE.MySpline(ips.ks, ips.pks; bc = "error")
        PK = ips

        #kmin, kmax = min(ips.ks...), max(ips.ks...)
        kmin, kmax, s0 = 1e-5, 1e3, 1e-3

        p0 = con ? [-1.0, 1.0, 0.0] : [-1.0, 1.0]

        I00 = IntegralIPS(PK, 0, 0; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)
        I20 = IntegralIPS(PK, 2, 0; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)
        I40 = IntegralIPS(PK, 4, 0; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)
        I02 = IntegralIPS(PK, 0, 2; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)
        I22 = IntegralIPS(PK, 2, 2; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)
        I31 = IntegralIPS(PK, 3, 1; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)
        I13 = IntegralIPS(PK, 1, 3; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)
        I11 = IntegralIPS(PK, 1, 1; N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)

        p0_tilde = con ? [-2.0, -1.0, 0.0] : [-2.0, -1.0]
        I04_tilde = IntegralIPS(PK, func_I04_tilde; N=N, kmin=kmin, kmax=kmax,
            fit_left_min=0.1, fit_left_max=1.0, p0_left=p0_tilde, con=con)


        #=
        I00 = GaPSE.MySpline(expanded_Iln(PK, 0, 0; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")
        I20 = GaPSE.MySpline(expanded_Iln(PK, 2, 0; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")
        I40 = GaPSE.MySpline(expanded_Iln(PK, 4, 0; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")
        I02 = GaPSE.MySpline(expanded_Iln(PK, 0, 2; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")
        I22 = GaPSE.MySpline(expanded_Iln(PK, 2, 2; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")
        I31 = GaPSE.MySpline(expanded_Iln(PK, 3, 1; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")
        I13 = GaPSE.MySpline(expanded_Iln(PK, 1, 3; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")
        I11 = GaPSE.MySpline(expanded_Iln(PK, 1, 1; lim = lim, N = N, kmin = kmin, kmax = kmax, s0 = s0,
                fit_left_min = fit_min, fit_left_max = fit_max, p0_left = p0, con = con)...; bc = "error")

        #ss = 10 .^ range(log10(s0), log10(s0) - log10(kmin) - log10(kmax), length = N)
        ss = 10 .^ range(log10(lim), 6, length = 1024)
        I04_tildes = expanded_I04_tilde(PK, ss; kmin = kmin, kmax = kmax)
        #I04_tildes = [func_I04_tilde(PK, s, kmin, kmax) for s in ss]
        I04_tilde = GaPSE.MySpline(ss, I04_tildes; bc = "error")
        =#

        σ_0 = quadgk(q -> PK(q) * q^2 / (2 * π^2), k_min, k_max)[1]
        σ_1 = quadgk(q -> PK(q) * q / (2 * π^2), k_min, k_max)[1]
        σ_2 = quadgk(q -> PK(q) / (2 * π^2), k_min, k_max)[1]
        σ_3 = quadgk(q -> PK(q) / (2 * π^2 * q), k_min, k_max)[1]
        σ_4 = quadgk(q -> PK(q) / (2 * π^2 * q^2), k_min, k_max)[1]

        new(I00, I20, I40, I02, I22, I31, I13, I11, I04_tilde, σ_0, σ_1, σ_2, σ_3, σ_4,
            fit_min, fit_max, k_min, k_max)
    end

    #=
    function IPSTools(ips::InputPS, iIs::String;
        k_min::Float64 = 1e-8,
        k_max::Float64 = 10.0
    )
        #PK = GaPSE.MySpline(ips.ks, ips.pks; bc = "error")
        Pk= ips

        tab_Is = readdlm(iIs, comments = true)
        ss = convert(Vector{Float64}, tab_Is[2:end, 1])

        #kmin, kmax = min(ips.ks...), max(ips.ks...)
        kmin, kmax, s0 = 1e-5, 1e3, 1e-3

        I00 = GaPSE.MySpline(ss, convert(Vector{Float64}, tab_Is[2:end, 2]); bc = "error")
        I20 = GaPSE.MySpline(ss, convert(Vector{Float64}, tab_Is[2:end, 3]); bc = "error")
        I40 = GaPSE.MySpline(ss, convert(Vector{Float64}, tab_Is[2:end, 4]); bc = "error")
        I02 = GaPSE.MySpline(ss, convert(Vector{Float64}, tab_Is[2:end, 5]) ./ ss .^ 2; bc = "error")
        I22 = GaPSE.MySpline(ss, convert(Vector{Float64}, tab_Is[2:end, 6]) ./ ss .^ 2; bc = "error")
        I31 = GaPSE.MySpline(ss, convert(Vector{Float64}, tab_Is[2:end, 7]) ./ ss; bc = "error")
        I11 = GaPSE.MySpline(ss, convert(Vector{Float64}, tab_Is[2:end, 8]) ./ ss; bc = "error")
        I13 = GaPSE.MySpline(xicalc(PK, 1, 3; N = 1024, kmin = kmin, kmax = kmax, r0 = s0)...; bc = "error")

        #ss = 10 .^ range(log10(s0), log10(s0) - log10(kmin) - log10(kmax), length = N)
        ss = 10 .^ range(log10(lim), 4, length = 1000)
        I04_tildes = expanded_I04_tilde(PK, ss; kmin = kmin, kmax = kmax)
        I04_tilde = GaPSE.MySpline(ss, I04_tildes; bc = "error")

        σ_0 = quadgk(q -> PK(q) * q^2 / (2 * π^2), k_min, k_max)[1]
        σ_1 = quadgk(q -> PK(q) * q / (2 * π^2), k_min, k_max)[1]
        σ_2 = quadgk(q -> PK(q) / (2 * π^2), k_min, k_max)[1]
        σ_3 = quadgk(q -> PK(q) / (2 * π^2 * q), k_min, k_max)[1]

        new(I00, I20, I40, I02, I22, I31, I13, I11, I04_tilde, σ_0, σ_1, σ_2, σ_3,
            nothing, nothing, k_min, k_max, s0)
    end
    =#

end



