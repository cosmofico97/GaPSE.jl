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
# along with  If not, see <http://www.gnu.org/licenses/>.
#



"""
    TF(
        left_value::Float64
        left::Float64

        spline::GaPSE.MySpline

        r_si::Float64
        r_b::Float64
        r_a::Float64
        right::Float64
        )

Contains all the information useful in order to return the Transfer Function value from:
- a spline inside the interval `left ≤ x ≤ right`
- the associated power law for `x > right` (with "right" coefficients `r_si`, `r_b` and `r_a`)
- the associated constant left value `left_value` for `x < left`



## Arguments 

- `left_value::Float64` : the constant value that must be returned in case `x < left`.

- `left::Float64` : the break between the left power-law (for `x <left`) and the 
  spline (for `x ≥ left`); its value is the `xs[begin]` one.

- `spline::GaPSE.MySpline` : spline that interpolates between the real values of the 
  integral calculated inside the range `left ≤ x ≤ right`

- `right::Float64` : the break between the right power-law (for `x ≥ left`) and the 
  spline (for `x ≤ right`); its value is the `xs[end]` one.

- `r_si, r_b, r_a :: Float64` : coefficient for the spurious power-law 
  ``y = f(x) = a + b \\, x^s`` for the RIGHT edge; when an input value `x > right` is
  given, the returned one is obtained from `power_law` with this coefficients (
  where, of course, `r_si` is the exponent, `r_b`` the coefficient and `r_a` the 
  spurious adding constant). 
  NOTE: for numerical issues, the "pure" power-law ``y = f(x) = b + x^s`` should be used. 

## Constructors

`TF(ks, Tks)` : from a set of `(ks, Tks)` pairs, take the mean of the first 10 as left
value and fit with `power_law_from_data` the last 15.


See also: [`power_law_from_data`](@ref)
"""
struct TF
    left_value::Float64
    left::Float64

    spline::GaPSE.MySpline

    r_si::Float64
    r_b::Float64
    r_a::Float64
    right::Float64


    function TF(ks, Tks)
        left_val = sum(Tks[1:10]) / 10
        r_si, r_b, r_a = power_law_from_data(
            ks, Tks, [-2.0, 1.0], ks[end-15], ks[end]; con=false)
        spline = GaPSE.MySpline(ks, Tks; bc="error")

        new(left_val, ks[2], spline, r_si, r_b, r_a, ks[end])
    end

end;


"""
    (f::TF)(x)

Return the value of the `f::TF` as follows:
```math
f(x)=
\\begin{cases}
y_{\\mathrm left} \\; ,
    \\quad \\quad x < \\mathrm{left}\\\\
\\mathrm{spline}(x) \\; , \\quad \\mathrm{left} \\leq x \\leq \\mathrm{right} \\\\
a_\\mathrm{R} + b_\\mathrm{R} \\, x ^ {s_\\mathrm{R}} \\; , 
\\quad x > \\mathrm{right}
\\end{cases}
```

where ``y_\\mathrm{left}``,``\\mathrm{left}``,
``\\mathrm{spline}``, ``a_\\mathrm{R}``, ``b_\\mathrm{R}``, ``s_\\mathrm{R}`` and 
``\\mathrm{right}`` are all stored inside the `TF` considered.

See also: [`TF`](@ref)
"""
function (tf::TF)(k)
    if k < tf.left
        return tf.left_value
    elseif k > tf.right
        return power_law(k, tf.r_si, tf.r_b, tf.r_a)
    else
        return tf.spline(k)
    end
end


##########################################################################################92


"""
    α_bias(k, tf::TF; bf=1.0, D=1.0, Ω_M0=0.29992)

Return the coefficient ``\\alpha_{\\rm bias}`` that relates the Non-Gaussian density fluctiations 
``\\delta_{\\rm NG}`` and the Non-Gaussian gravitational potential ``\\Phi_{\\rm NG}``
in Fourier space:

```math
\\delta_{\\rm NG}(k) = \\alpha(k, z) \\,  \\Phi_{\\rm NG}(k) \\; \\;  , \\quad
\\alpha(k, z) = \\frac{2}{3} \\frac{k^2 T_m(k) D(z)}{\\Omega_{\\mathrm{M}0}} \\left(\\frac{c}{H_0}\\right)^2 
\\; \\; ,  \\quad \\; \\alpha_{\\rm bias} = \\frac{b_{\\phi} f_{\\rm NL}}{\\alpha(k, z)} \\; .
```

- `bf=1.0` : value of the degenerate product ``b_{\\phi} f_{\\rm NL}``.
- `D = 1.0` : value of the linear growth factor ``D`` at present day; inside this function, 
  it is multiplied for a constant ``q = 0.779017`` in order to normalize ``D(z)`` to 1/(1+z) 
  in matter domination, i.e. such that ``q \\, D(z) \\, (1+z) = 1`` at ``z=20``  


See also: [`TF`](@ref)
"""
function α_bias(k, tf::TF; bf=1.0, D=1.0, Ω_M0=0.29992)
    return 1.5 * bf * Ω_M0 * (100 / 299792.458)^2 / (0.779017 * D * k^2 * tf(k))
end



"""
    IntegralIPSalpha(
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

Contains all the information useful in order to return the value of the integral
of the Input Power Spectrum weighted with the `α_bias` function. In other words,
return this expression:

```math
\\int_0^\\infty \\frac{\\mathrm{d} q}{2 \\pi^2} \\, q^2 \\,
    \\frac{j_\\ell(qs)}{(qs)^n} \\, P(q) \\, \\alpha_{\\mathrm{bias}} \\; ,
```

where ``P(q)`` is the Input Power Spectrum and

```math
\\delta_{\\rm NG}(k) = \\alpha(k, z) \\,  \\Phi_{\\rm NG}(k) \\; \\;  , \\quad
\\alpha(k, z) = \\frac{2}{3} \\frac{k^2 T_m(k) D(z)}{\\Omega_{\\mathrm{M}0}} \\left(\\frac{c}{H_0}\\right)^2 
\\; \\; ,  \\quad \\; \\alpha_{\\rm bias} = \\frac{b_{\\phi} f_{\\rm NL}}{\\alpha(k, z)} \\; .
```

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

    IntegralIPSalpha(tf::TF, cosmo::Cosmology, l, n=0; D=nothing, bf=1.0,
        N::Int=1024, kmin=1e-6, kmax=1e4, s0=1e-4,
        fit_left_min=nothing, fit_left_max=nothing, p0_left=nothing,
        fit_right_min=nothing, fit_right_max=nothing, p0_right=nothing)

The integral obtained with this constructor is calculated through `xicalc`, and
expanded with power-laws at the edges.

- `tf::TF`: the struct that contains all the data concerning the Transfer Function.

- `cosmo::Cosmology` : cosmology to be used in this computation

- `l` : degree of the spherical Bessel function to be used.

- `n=0` : degree of the exponent for the denominator. The interesting case is
  only the default value ``0``.

- `D = nothing` : value of the linear growth factor ``D`` to be used. If `nothing`,
  it will be internally set as ``D(z_{\\mathrm{eff}})``, where ``z_{\\mathrm{eff}}`` is
  the effective redshift for the input cosmology.

- `bf = 1.0` : value of the degenerate product ``b_{\\phi} f_{\\rm NL}``.

- `kmin = 1e-6, kmax = 1e4, s0 = 1e-4` : values to be passed to `xicalc` for the
  integration

- `fit_left_min = 2.0, fit_left_max = 10.0` : the limits (min and max) where the integral
  must be fitted with a power law, for small distances. This operation is necessary, because `xicalc`,
  in this context, gives wrong results for too small input distance `s`; nevertheless,
  this integral has fixed power-law trends for ``s \\rightarrow 0``, so this approach gives
  good results.

- `p0_left = nothing` : vector with the initial values for the left power-law fitting; its length must
  be 2 (if you want to fit with a pure power-law ``y = f(x) = b x^s``, so only `l_si` and `l_b` 
  are matter of concern) or 3 (if you want to fit with a spurious power-law ``y = f(x) = a + b x^s``,
  so you are also interested in `l_a`); if `nothing`, it will be
  automatically set `p0 = [-1.0, 1.0]`.

- `fit_right_min = nothing, fit_right_max = nothing` : the limits (min and max) where the integral
  must be fitted with a power law, for high distances. 
  This integral has fixed power-law trends for ``s \\rightarrow \\infty``, so this approach gives
  good results. If `nothing`, the last 15 points returned from `xicalc` are used for
  this fitting.
  NOTE: for numerical issues, only the "pure" power-law ``y = f(x) = b + x^s`` can be used. 

- `p0_right = nothing` : vector with the initial values for the left power-law fitting; its length must
  be 2 (to fit with a pure power-law ``y = f(x) = b x^s``, so only `r_si` and `r_b` 
  are matter of concern); if `nothing`, it will be
  automatically set `p0 = [-4.0, 1.0]`.

All the power-law fitting (both "pure" and spurious) are made through the 
local function `power_law_from_data`.

See also: [`power_law_from_data`](@ref), [`power_law`](@ref), 
[`Cosmology`](@ref), [`α_bias`](@ref)
"""
struct IntegralIPSalpha
    l_si::Float64
    l_b::Float64
    l_a::Float64
    left::Float64

    spline::GaPSE.MySpline

    r_si::Float64
    r_b::Float64
    r_a::Float64
    right::Float64

    function IntegralIPSalpha(tf::TF, cosmo::Cosmology, l, n=0;
        D=nothing, bf=1.0,
        N::Int=1024, kmin=1e-6, kmax=1e4, s0=1e-4,
        fit_left_min=nothing, fit_left_max=nothing, p0_left=nothing,
        fit_right_min=nothing, fit_right_max=nothing, p0_right=nothing)

        DD = isnothing(D) ? cosmo.D_of_s(cosmo.s_eff) : D
        Ω_M00 = cosmo.params.Ω_M0

        rs, xis = xicalc(k -> cosmo.IPS(k) * α_bias(k, tf; bf=bf, D=DD, Ω_M0=Ω_M00), l, n;
            N=N, kmin=kmin, kmax=kmax, r0=s0)

        fit_left_MIN = !isnothing(fit_left_min) ? fit_left_min : begin
            l ≈ 0.0 ? 5e-2 : l ≈ 2.0 ? 5e-1 : rs[2]
        end
        fit_left_MAX = !isnothing(fit_left_max) ? fit_left_max : begin
            l ≈ 0.0 ? 1e-1 : l ≈ 2.0 ? 1e0 : rs[16]
        end
        p_0_left = isnothing(p0_left) ? [-1.0, 1.0] : p0_left
        l_si, l_b, l_a = power_law_from_data(
            rs, xis, p_0_left, fit_left_MIN, fit_left_MAX; con=false)

        fit_right_MIN = isnothing(fit_right_min) ? rs[length(rs)-16] : fit_right_min
        fit_right_MAX = isnothing(fit_right_max) ? rs[length(rs)-1] : fit_right_max
        p_0_right = isnothing(p0_right) ? [-4.0, 1.0] : p0_right
        r_si, r_b, r_a = power_law_from_data(
            rs, xis, p_0_right, fit_right_MIN, fit_right_MAX; con=false)

        ind_left = findfirst(x -> x > fit_left_MIN, rs) - 1
        ind_right = findfirst(x -> x >= fit_right_MAX, rs)
        new_rs = vcat(rs[ind_left:ind_right])
        new_Js = vcat(xis[ind_left:ind_right])
        spline = GaPSE.MySpline(new_rs, new_Js; bc="error")

        #println("\nleft = $l_si , $l_b , $l_a, $fit_left_min")
        #println("right = $r_si , $r_b , $r_a, $fit_right_MAX\n")

        new(l_si, l_b, l_a, fit_left_MIN, spline, r_si, r_b, r_a, fit_right_MAX)
    end
end


"""
    (f::IntegralIPSalpha)(x)

Return the value of the `f::IntegralIPSalpha` as follows:
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
``\\mathrm{right}`` are all stored inside the `IntegralIPSalpha` considered.

See also: [`IntegralIPSalpha`](@ref)
"""
function (Jl::IntegralIPSalpha)(x)
    if x < Jl.left
        return power_law(x, Jl.l_si, Jl.l_b, Jl.l_a)
    elseif x > Jl.right
        #warning("i am going too right! ")
        return power_law(x, Jl.r_si, Jl.r_b, Jl.r_a)
    else
        return Jl.spline(x)
    end
end;



##########################################################################################92



"""
    CosmoParams(
            D::Float64 
            bf::Float64

            flm_0::Float64 
            flM_0::Float64 
            kmin_0::Float64 
            kmax_0::Float64  
            N_0::Int

            flm_2::Float64  
            flM_2::Float64 
            kmin_2::Float64 
            kmax_2::Float64  
            N_2::Int
        )


Struct that contains all the parameters and options that are 
matter of concerns for the `CosmoPNG` we are interested in.

## Arguments

- `D` : linear growth factor `D` to be used for the
  `α_bias` function.

- `bf` : value of the degenerate product ``b_{\\phi} f_{\\rm NL}``.

- `kmin_0`, `kmax_0`, s0_0` : values to be passed to `xicalc` for the
  integration made by `IntegralIPSalpha` of the term ``J_0``

- `flm_0`, `flM_0` : the limits (min and max) where the integral made by `IntegralIPSalpha` of the term ``J_0``
  must be fitted with a power law, for small distances. This operation is necessary, because `xicalc`,
  in this context, gives wrong results for too small input distance `s`; nevertheless,
  this integral has fixed power-law trends for ``s \\rightarrow 0``, so this approach gives
  good results.

- `kmin_2`, `kmax_2`, s0_2`, `flm_2`, `flM_2` : same as the previous terms, but for the integral ``J_2``


## Constructors

    function CosmoPNGParams(D; 
        bf = 1.0,
        flm_0 = 5e-2, flM_0 = 1e-1, s0_0 = 1e-4,
        kmin_0 = 1e-6, kmax_0 = 1e4, N_0::Int = 1024,
        flm_2 = 5e-1, flM_2 = 1e0, s0_2 = 1e-4,
        kmin_2 = 1e-6, kmax_2 = 1e4, N_2::Int = 1024,
        )
     
The associations are trivials.
The only thing to be put attention on is that `D` is a MANDATORY argument, while
all the other ones are keyword arguments with a default value.
You should use:

    pngparams = CosmoPNGParams(cosmo.D_of_s(cosmo.s_eff); ...)

where `cosmo::Cosmology` is the Cosmology you are interested in and 
`s_eff` is the effective comoving distance (stored on `cosmo`). 

See also: [`Cosmology`](@ref), [`CosmoPNG`](@ref), [`IntegralIPSalpha`](@ref),
[`α_bias`](@ref)
"""
struct CosmoPNGParams
    D::Float64
    bf::Float64

    flm_0::Float64
    flM_0::Float64
    s0_0::Float64
    kmin_0::Float64
    kmax_0::Float64
    N_0::Int

    flm_2::Float64
    flM_2::Float64
    s0_2::Float64
    kmin_2::Float64
    kmax_2::Float64
    N_2::Int

    function CosmoPNGParams(D;
        bf=1.0,
        flm_0=5e-2, flM_0=1e-1, s0_0=1e-4,
        kmin_0=1e-6, kmax_0=1e4, N_0::Int=1024,
        flm_2=5e-1, flM_2=1e0, s0_2=1e-4,
        kmin_2=1e-6, kmax_2=1e4, N_2::Int=1024
    )

        @assert D > 0.0 "D > 0.0 must hold!"
        @assert bf > 0.0 "bf > 0.0 must hold!"

        @assert N_0 > 10 "N_0 > 10 must hold!"
        @assert 0.0 < flm_0 < flM_0 "0.0 < flm_0 < flM_0 must hold!"
        @assert 0.0 < kmin_0 < kmax_0 "0.0 < kmin_0 < kmax_0 must hold!"
        @assert kmin_0 < s0_0 < kmax_0 "kmin_0 < s0_0 < kmax_0 must hold!"

        @assert N_2 > 10 "N_2 > 10 must hold!"
        @assert 0.0 < flm_2 < flM_2 "0.0 < flm_2 < flM_2 must hold!"
        @assert 0.0 < kmin_2 < kmax_2 "0.0 < kmin_2 < kmax_2 must hold!"
        @assert kmin_2 < s0_2 < kmax_2 "kmin_2 < s0_2 < kmax_2 must hold!"


        new(
            D, bf,
            flm_0, flM_0, s0_0, kmin_0, kmax_0, N_0,
            flm_2, flM_2, s0_2, kmin_2, kmax_2, N_2,
        )
    end
end

"""
    CosmoPNG(
        params::CosmoPNGParams
        tf::TF
        file_TF::String

        J0::IntegralIPSalpha
        J2::IntegralIPSalpha
        )

Struct that contains all the information that may be used for the 
Correlation Function computations of the Primordial Non-Gaussianities (PNG) signal.

## Arguments

- `params::CosmoPNGParams` : parameters to be used for this Cosmology. See the docstring
  of `CosmoParams` for more information on the possible inputs.

- `tf::TF` : transfer function to be used.

- `file_TF::String` : name of the file where the transfer function was read.

- `J0` and `J2::IntegralIPSalpha` : integrals with the following form:
  ```math
  J_\\ell = \\int_0^\\infty \\frac{\\mathrm{d} q}{2 \\pi^2} \\, q^2 \\,
  j_\\ell(qs) \\, P(q) \\, \\alpha_{\\mathrm{bias}} \\; ,
  ```

  where ``P(q)`` is the Input Power Spectrum and

  ```math
  \\delta_{\\rm NG}(k) = \\alpha(k, z) \\,  \\Phi_{\\rm NG}(k) \\; \\;  , \\quad
  \\alpha(k, z) = \\frac{2}{3} \\frac{k^2 T_m(k) D(z)}{\\Omega_{\\mathrm{M}0}} \\left(\\frac{c}{H_0}\\right)^2 
  \\; \\; ,  \\quad \\; \\alpha_{\\rm bias} = \\frac{b_{\\phi} f_{\\rm NL}}{\\alpha(k, z)} \\; .
  ```


## Constructor 

    CosmoPNG(
        pngparams::CosmoPNGParams,
        cosmo::Cosmology, file_TF::String;
        comments::Bool=true
    )

- `pngparams::CosmoParams` : parameters to be used for this Cosmology. See the docstring
  of `CosmoParams` for more information on the possible inputs.

- `cosmo::Cosmology` : cosmology to be considered, both in terms od Input Power Spectrum
  and of cosmological parameters.

- `file_TF::String ` : name of the file where the Transfer Function to be used is stored.

- `comments::Bool=true` : the `file_TF` file contains comments at the beginning?

See also: [`TF`](@ref), [`IntegralIPSalpha`](@ref), [`Cosmology`](@ref)
"""
struct CosmoPNG
    params::CosmoPNGParams

    tf::TF
    file_TF::String

    J0::IntegralIPSalpha
    J2::IntegralIPSalpha

    function CosmoPNG(
            pngparams::CosmoPNGParams,
            cosmo::Cosmology, file_TF::String;
            comments::Bool=true
        )

        table = readdlm(file_TF; comments=comments)
        ks_tf = convert(Vector{Float64}, table[:, 1])
        pks_tf = convert(Vector{Float64}, table[:, 2])

        #DD = isnothing(D) ? cosmo.D_of_s(cosmo.s_eff) : D
        tf = TF(ks_tf, pks_tf)

        J0 = IntegralIPSalpha(tf, cosmo, 0, 0;
            D=pngparams.D, bf=pngparams.bf,
            kmin=pngparams.kmin_0, kmax=pngparams.kmax_0,
            s0=pngparams.s0_0,
            fit_left_min=pngparams.flm_0, fit_left_max=pngparams.flM_0,
            N=pngparams.N_0)
        J2 = IntegralIPSalpha(tf, cosmo, 2, 0;
            D=pngparams.D, bf=pngparams.bf,
            kmin=pngparams.kmin_2, kmax=pngparams.kmax_2,
            s0=pngparams.s0_2,
            fit_left_min=pngparams.flm_2, fit_left_max=pngparams.flM_2,
            N=pngparams.N_2)

        new(pngparams, tf, file_TF, J0, J2)
    end
end


##########################################################################################92


err_png = "the keyword 'sp' must indicate which galaxy species you want to consider; " *
          "this means that you can chose only between type (and corresponding 'sp' values) 1 or 2; sp="

function ξ_S_L0(P::Point, cosmo::Cosmology, cosmopng::CosmoPNG; b=nothing, sp::Int64=1)
    b = !isnothing(b) ? b : sp == 1 ? cosmo.params.b1 : sp == 2 ? cosmo.params.b2 : throw(AssertionError(err_ppg * "$sp is not valid!"))
    s = P.comdist

    Peff = Point(cosmo.s_eff, cosmo)
    D, f = Peff.D, Peff.f

    2.0 * (b + f / 3.0) * D^2 * cosmopng.J0(s)
end

function ξ_S_L0(s1, cosmo::Cosmology, cosmopng::CosmoPNG; kwargs...)
    P1 = Point(s1, cosmo)
    return ξ_S_L0(P1, cosmo, cosmopng; kwargs...)
end

"""
    ξ_S_L0(P::Point, cosmo::Cosmology, cosmopng::CosmoPNG; b=nothing, sp::Int64=1)
    ξ_S_L0(s1, cosmo::Cosmology, cosmopng::CosmoPNG; kwargs...)

Return the value of the Two-Point Correlation Function (TPCF) monopole of the signal (S)
of the local Primordial Non-Gaussianities (PNG) (for the given `cosmo::Cosmology` and 
`cosmopng::CosmoPNG`).
In the first method, you should pass the `Point` where to evaluate that function,
while in the second (that internally recalls the first) you must provide the 
comoving distance `s`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of such TPCF monopole is the following:
```math
\\xi^{\\mathrm{S}}_0(s) = 2 \\left( b + \\frac{1}{3}f(z_{\\mathrm{eff}})\\right)
\\,  D^2(z_{\\mathrm{eff}}) \\, J_0(s)
```

where: 
- ``b`` is the galaxy bias (stored in `cosmo`)
- ``z`` is the redshift associated to the comoving distance ``s`` in this cosmology
- ``s_{\\mathrm{eff}}`` is the effective comoving distance stored in `cosmo` (and ``z_{\\mathrm{eff}}``
  its associated effective redshift in that cosmology)
- ``D`` the linear growth factor and ``f`` the linear growth rate (whose splines are stored in `cosmo`)
- ``J_\\ell`` (stored in `cosmopng`) is defined as
  ```math
  J_\\ell(s) = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^2 \\, P(q) \\, j_\\ell(qs) \\, \\alpha_{\\mathrm{bias}}(q,z)
  ```
  with ``P(q)`` as the matter Power Spectrum at ``z=0`` (stored in `cosmo`), ``j_\\ell`` as spherical
  Bessel function of order ``\\ell`` and
  ```math
  \\alpha_{\\rm bias}(q,z) = \\frac{b_{\\phi} f_{\\rm NL}}{\\alpha(q, z)} \\quad ,  \\quad 
  \\alpha(q, z) = \\frac{2}{3} \\frac{q^2 T_m(q) D(z)}{\\Omega_{\\mathrm{M}0}} \\left(\\frac{c}{H_0}\\right)^2
  ```
  with ``b_{\\phi}f_{\\rm NL}`` is stored in `cosmopng`. Check the documentation of `α_bias` and `CosmoPNG`
  for more information.

## Keyword arguments

- `b = nothing` : galaxy bias value to be considered in the computation; if set to `nothing`, its value
  is inferred from the other keyword argument `sp`;

- `sp::Int64=1` : galaxy type to be considered between the two species stored in `cosmo`: it can be either `1` or `2`,
  and the corresponding set of galaxy, magnification and evolutionary biases stored in `cosmo` are considered;
  in case `b` is set to a value, this parameter is ignored.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`CosmoPNG`](@ref), [`α_bias`](@ref), 
[`ξ_S_L2`](@ref), [`ξ_S`](@ref), 
[`integrand_ξ_S_multipole`](@ref), [`ξ_S_multipole`](@ref) 
[`map_ξ_S_multipole`](@ref), [`print_map_ξ_S_multipole`](@ref)
"""
ξ_S_L0

function ξ_S_L2(P::Point, cosmo::Cosmology, cosmopng::CosmoPNG; b=nothing, sp::Int64=1)
    s = P.comdist

    Peff = Point(cosmo.s_eff, cosmo)
    D, f = Peff.D, Peff.f

    -4.0 / 3.0 * f * D^2 * cosmopng.J2(s)
end

function ξ_S_L2(s1, cosmo::Cosmology, cosmopng::CosmoPNG; kwargs...)
    P1 = Point(s1, cosmo)
    return ξ_S_L2(P1, cosmo, cosmopng; kwargs...)
end


"""
    ξ_S_L2(P::Point, cosmo::Cosmology, cosmopng::CosmoPNG; b=nothing, sp::Int64=1)
    ξ_S_L2(s1, cosmo::Cosmology, cosmopng::CosmoPNG; kwargs...)

Return the value of the Two-Point Correlation Function (TPCF) quadrupole of the signal (S)
of the local Primordial Non-Gaussianities (PNG) (for the given `cosmo::Cosmology` and 
`cosmopng::CosmoPNG`).
In the first method, you should pass the `Point` where to evaluate that function,
while in the second (that internally recalls the first) you must provide the 
comoving distance `s`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of such TPCF monopole is the following:
```math
\\xi^{\\mathrm{S}}_2(s) = - \\frac{4}{3} \\, f(z_{\\mathrm{eff}})
\\,  D^2(z_{\\mathrm{eff}}) \\, J_2(s)
```

where: 
- ``b`` is the galaxy bias (stored in `cosmo`)
- ``z`` is the redshift associated to the comoving distance ``s`` in this cosmology
- ``s_{\\mathrm{eff}}`` is the effective comoving distance stored in `cosmo` (and ``z_{\\mathrm{eff}}``
  its associated effective redshift in that cosmology)
- ``D`` the linear growth factor and ``f`` the linear growth rate (whose splines are stored in `cosmo`)
- ``J_\\ell`` (stored in `cosmopng`) is defined as
  ```math
  J_\\ell(s) = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^2 \\, P(q) \\, j_\\ell(qs) \\, \\alpha_{\\mathrm{bias}}(q,z)
  ```
  with ``P(q)`` as the matter Power Spectrum at ``z=0`` (stored in `cosmo`), ``j_\\ell`` as spherical
  Bessel function of order ``\\ell`` and
  ```math
  \\alpha_{\\rm bias}(q,z) = \\frac{b_{\\phi} f_{\\rm NL}}{\\alpha(q, z)} \\quad ,  \\quad 
  \\alpha(q, z) = \\frac{2}{3} \\frac{q^2 T_m(q) D(z)}{\\Omega_{\\mathrm{M}0}} \\left(\\frac{c}{H_0}\\right)^2
  ```
  with ``b_{\\phi}f_{\\rm NL}`` is stored in `cosmopng`. Check the documentation of `α_bias` and `CosmoPNG`
  for more information.

## Keyword arguments

- `b = nothing` : galaxy bias value to be considered in the computation; if set to `nothing`, its value
  is inferred from the other keyword argument `sp`;

- `sp::Int64=1` : galaxy type to be considered between the two species stored in `cosmo`: it can be either `1` or `2`,
  and the corresponding set of galaxy, magnification and evolutionary biases stored in `cosmo` are considered;
  in case `b` is set to a value, this parameter is ignored.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`CosmoPNG`](@ref), [`α_bias`](@ref), 
[`ξ_S_L0`](@ref), [`ξ_S`](@ref), 
[`integrand_ξ_S_multipole`](@ref), [`ξ_S_multipole`](@ref) 
[`map_ξ_S_multipole`](@ref), [`print_map_ξ_S_multipole`](@ref)
"""
ξ_S_L2


"""
    ξ_S(s, μ, cosmo::Cosmology, cosmopng::CosmoPNG; b=nothing, sp::Int64=1)

Return the value of the Two-Point Correlation Function (TPCF) of the signal (S)
of the local Primordial Non-Gaussianities (PNG) in the given comoving distance `s` and cosine
value for the Legendre polynomials `μ` (for the given `cosmo::Cosmology` and 
`cosmopng::CosmoPNG`).
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of such TPCF monopole is the following:
```math
\\begin{split}
\\xi^{\\mathrm{S}}(s&,\\mu) = \\xi^{\\mathrm{S}}_0(s) + 
    \\xi^{\\mathrm{S}}_2(s) \\mathcal{L}_2(\\mu) \\\\
&\\xi^{\\mathrm{S}}_0(s) = 2 \\left( b + \\frac{1}{3}f(z_{\\mathrm{eff}})\\right)
\\,  D^2(z_{\\mathrm{eff}}) \\, J_0(s) \\\\
&\\xi^{\\mathrm{S}}_2(s) = - \\frac{4}{3} \\, f(z_{\\mathrm{eff}})
\\,  D^2(z_{\\mathrm{eff}}) \\, J_2(s)
\\end{split}
```


where: 
- ``b`` is the galaxy bias (stored in `cosmo`)
- ``z`` is the redshift associated to the comoving distance ``s`` in this cosmology
- ``s_{\\mathrm{eff}}`` is the effective comoving distance stored in `cosmo` (and ``z_{\\mathrm{eff}}``
  its associated effective redshift in that cosmology)
- ``D`` the linear growth factor and ``f`` the linear growth rate (whose splines are stored in `cosmo`)
- ``\\mathcal{L}_\\ell`` the Legendre polynomial of order ``\\ell``
- ``J_\\ell`` (stored in `cosmopng`) is defined as
  ```math
  J_\\ell(s) = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^2 \\, P(q) \\, j_\\ell(qs) \\, \\alpha_{\\mathrm{bias}}(q,z)
  ```
  with ``P(q)`` as the matter Power Spectrum at ``z=0`` (stored in `cosmo`), ``j_\\ell`` as spherical
  Bessel function of order ``\\ell`` and
  ```math
  \\alpha_{\\rm bias}(q,z) = \\frac{b_{\\phi} f_{\\rm NL}}{\\alpha(q, z)} \\quad ,  \\quad 
  \\alpha(q, z) = \\frac{2}{3} \\frac{q^2 T_m(q) D(z)}{\\Omega_{\\mathrm{M}0}} \\left(\\frac{c}{H_0}\\right)^2
  ```
  with ``b_{\\phi}f_{\\rm NL}`` is stored in `cosmopng`. Check the documentation of `α_bias` and `CosmoPNG`
  for more information.

## Keyword arguments

- `b = nothing` : galaxy bias value to be considered in the computation; if set to `nothing`, its value
  is inferred from the other keyword argument `sp`;

- `sp::Int64=1` : galaxy type to be considered between the two species stored in `cosmo`: it can be either `1` or `2`,
  and the corresponding set of galaxy, magnification and evolutionary biases stored in `cosmo` are considered;
  in case `b` is set to a value, this parameter is ignored.


See also: [`Point`](@ref), [`Cosmology`](@ref), [`CosmoPNG`](@ref), [`α_bias`](@ref), 
[`ξ_S_L0`](@ref), [`ξ_S_L2`](@ref), 
[`integrand_ξ_S_multipole`](@ref), [`ξ_S_multipole`](@ref) 
[`map_ξ_S_multipole`](@ref), [`print_map_ξ_S_multipole`](@ref)
"""
function ξ_S(s1, y, cosmo::Cosmology, cosmopng::CosmoPNG; kwargs...)
    ξ_S_L0(s1, cosmo, cosmopng; kwargs...) + ξ_S_L2(s1, cosmo, cosmopng; kwargs...) * Pl(y, 2)
end



##########################################################################################92


"""
    integrand_ξ_S_multipole(s, μ, cosmo::Cosmology, cosmopng::CosmoPNG;
        L::Int=0, use_windows::Bool=true, b=nothing, sp::Int64=1 )::Float64

Return the integrand on ``\\mu = \\hat{\\mathbf{s}}_1 \\cdot \\hat{\\mathbf{s}}`` 
of the Two-Point Correlation Function (TPCF) concerning the signal (S) of the 
local Primordial Non-Gaussianities (PNG), i.e. the following function ``f(s, \\mu)``:

```math
     f_L(s, \\mu) = \\xi^{\\mathrm{S}} \\left(s, \\mu\\right) 
          \\, \\mathcal{L}_L(\\mu) \\, \\times 
    \\begin{cases} 
        \\frac{1}{\\mathcal{N}}\\mathcal{F}(s, \\mu) \\quad \\mathrm{use\\_windows == true} \\\\
        1 \\quad\\quad \\mathrm{use\\_windows == false}
    \\end{cases}
```

where:
- ``\\xi^{\\mathrm{S}}`` is the TPCF of the PNG signal, computed from `ξ_S`.
- ``\\mathcal{L}_L(\\mu)`` is the Legendre polynomial of order ``L``
- ``\\mathcal{F}(s, \\mu)`` is the integrated window function stored in `cosmo::Cosmology` (check the documentation of `WindowFIntegrated`)
- ``\\mathcal{N}`` is the integrated window function norm (check the documentation of `WindowFIntegrated`)

## Inputs

- `s`: the comoving distance  where must be evaluated the integral

- `μ`: the cosine between `s1` and `s` where must be evaluated the integral

- `cosmo::Cosmology`: cosmology to be used in this computation

- `cosmopng::CosmoPNG`: struct that contains all the information that may be used for the TPCF 
  computations of the PNG signal.

## Keyword arguments 

- `L::Int = 0`: order of the Legendre polynomial to be used

- `use_windows::Bool = false`: tells if the integrand must consider ``\\mathcal{F}``
  or not.

- `b = nothing` : galaxy bias value to be considered in the computation; if set to `nothing`, its value
  is inferred from the other keyword argument `sp`;

- `sp::Int64=1` : galaxy type to be considered between the two species stored in `cosmo`: it can be either `1` or `2`,
  and the corresponding set of galaxy, magnification and evolutionary biases stored in `cosmo` are considered;
  in case `b` is set to a value, this parameter is ignored.

See also:[`ξ_S`](@ref), [`ξ_S_multipole`](@ref), 
[`map_ξ_S_multipole`](@ref), [`print_map_ξ_S_multipole`](@ref)
[`WindowFIntegrated`](@ref), [`Cosmology`](@ref), [`CosmoPNG`](@ref), 
"""
function integrand_ξ_S_multipole(s, μ, cosmo::Cosmology, cosmopng::CosmoPNG;
    L::Int=0, use_windows::Bool=true, kwargs...)

    res = if use_windows == true
        ξ_S(s, μ, cosmo, cosmopng; kwargs...) .* (
            spline_integrF(s, μ, cosmo.windowFint) / cosmo.WFI_norm * Pl(μ, L)
        )
    else
        ξ_S(s, μ, cosmo, cosmopng; kwargs...) .* Pl(μ, L)
    end

    return (2.0 * L + 1.0) / 2.0 * res
end


"""
    ξ_S_multipole(
        s, cosmo::Cosmology, cosmopng::CosmoPNG;;
        L::Int = 0, use_windows::Bool = true,
        atol_quad::Float64 = 0.0,
        rtol_quad::Float64 = 1e-2
        enhancer::Float64 = 1e6,
        b=nothing, sp::Int64=1 ) ::Float64


Evaluate the multipole of order `L` of the Two-Point Correlation Function (TPCF) 
concerning the signal (S) of the local Primordial Non-Gaussianities (PNG), 
i.e. the following function ``\\xi^{\\mathrm{S}} (s)``:

```math
     \\xi^{\\mathrm{S}} (s) = \\frac{2 L + 1}{2} \\int_{-1}^{+1} \\mathrm{d}\\mu \\; 
    \\xi^{\\mathrm{S}} \\left(s, \\mu\\right) 
          \\, \\mathcal{L}_L(\\mu) \\, \\times 
    \\begin{cases} 
        \\frac{1}{\\mathcal{N}}\\mathcal{F}(s, \\mu) \\quad \\mathrm{use\\_windows == true} \\\\
        1 \\quad\\quad \\mathrm{use\\_windows == false}
    \\end{cases}
```

where:
- ``\\xi^{\\mathrm{S}}(s,\\mu)`` is the TPCF of the PNG signal with the angular dependence, computed from `ξ_S`.
- ``\\mathcal{L}_L(\\mu)`` is the Legendre polynomial of order ``L``
- ``\\mathcal{F}(s, \\mu)`` is the integrated window function stored in `cosmo::Cosmology` (check the documentation of `WindowFIntegrated`)
- ``\\mathcal{N}`` is the integrated window function norm (check the documentation of `WindowFIntegrated`)

The integration over ``\\mu`` is preformed through the Julia function `quadgk` 
from the [`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl) Julia package, that uses an adaptive 
Gauss-Kronrod quadrature.

## Inputs

- `s`: the comoving distance  where must be evaluated the integral

- `cosmo::Cosmology`: cosmology to be used in this computation

- `cosmopng::CosmoPNG`: struct that contains all the information that may be used for the TPCF 
  computations of the PNG signal.

## Keyword arguments 

- `L::Int = 0`: order of the Legendre polynomial to be used

- `use_windows::Bool = false`: tells if the integrand must consider ``\\mathcal{F}``
  or not.

- `atol_quad::Float64 = 0.0` and `rtol_quad::Float64 = 1e-2`: absolute and relative tolerance
  to be passed to the function `quadgk`; it's recommended not to set `rtol_quad < 1e-2` 
  because the time for evaluation increase quickly.

- `enhancer::Float64 = 1e6`: just a float number used in order to deal better with small numbers; 
  the returned value is NOT modified by this value, because after a multiplication
  the internal result is divided by `enhancer`.

- `b = nothing` : galaxy bias value to be considered in the computation; if set to `nothing`, its value
  is inferred from the other keyword argument `sp`;

- `sp::Int64=1` : galaxy type to be considered between the two species stored in `cosmo`: it can be either `1` or `2`,
  and the corresponding set of galaxy, magnification and evolutionary biases stored in `cosmo` are considered;
  in case `b` is set to a value, this parameter is ignored.

See also: [`ξ_S`](@ref), [`integrand_ξ_S_multipole`](@ref), 
[`map_ξ_S_multipole`](@ref), [`print_map_ξ_S_multipole`](@ref)
[`WindowFIntegrated`](@ref), [`Cosmology`](@ref), [`CosmoPNG`](@ref), 
"""
function ξ_S_multipole(
    s, cosmo::Cosmology, cosmopng::CosmoPNG;
    atol_quad::Float64=0.0,
    rtol_quad::Float64=1e-2,
    enhancer::Float64=1e6, kwargs...)

    orig_f(μ) = enhancer * integrand_ξ_S_multipole(s, μ, cosmo, cosmopng; kwargs...)

    int = quadgk(μ -> orig_f(μ), -1.0, 1.0; atol=atol_quad, rtol=rtol_quad)[1]

    return int / enhancer
end



##########################################################################################92


"""
    map_ξ_S_multipole(
        cosmo::Cosmology, cosmopng::CosmoPNG,
        ss = nothing;
        L::Int = 0, use_windows::Bool = true,
        atol_quad::Float64 = 0.0,
        rtol_quad::Float64 = 1e-2,
        enhancer::Float64 = 1e6,
        pr::Bool = true,
        N_log::Int = 1000,
        kwargs...) ::Tuple{Vector{Float64}, Vector{Float64}}


Evaluate the multipole of order `L` of the Two-Point Correlation Function (TPCF) 
concerning the signal (S) of the local Primordial Non-Gaussianities (PNG),
for all the comoving distance values stored inside `ss`.
If `ss = nothing`, it is set `ss = 10 .^ range(0, log10(2 * cosmo.s_max), length=N_log)`.

The function evaluated is then the following ``\\xi^{\\mathrm{S}} (s)``:

```math
     \\xi^{\\mathrm{S}} (s) = \\frac{2 L + 1}{2} \\int_{-1}^{+1} \\mathrm{d}\\mu \\; 
    \\xi^{\\mathrm{S}} \\left(s, \\mu\\right) 
          \\, \\mathcal{L}_L(\\mu) \\, \\times 
    \\begin{cases} 
        \\frac{1}{\\mathcal{N}}\\mathcal{F}(s, \\mu) \\quad \\mathrm{use\\_windows == true} \\\\
        1 \\quad\\quad \\mathrm{use\\_windows == false}
    \\end{cases}
```

where:
- ``\\xi^{\\mathrm{S}}(s,\\mu)`` is the TPCF of the PNG signal
  with the angular dependence, computed from `ξ_S`.
- ``\\mathcal{L}_L(\\mu)`` is the Legendre polynomial of order ``L``
- ``\\mathcal{F}(s, \\mu)`` is the integrated window function stored in `cosmo::Cosmology` (check the documentation of `WindowFIntegrated`)
- ``\\mathcal{N}`` is the integrated window function norm (check the documentation of `WindowFIntegrated`)

The integration over ``\\mu`` is preformed through the Julia function `quadgk` 
from the [`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl) Julia package, that uses an adaptive 
Gauss-Kronrod quadrature.

## Inputs

- `cosmo::Cosmology`: cosmology to be used in this computation

- `cosmopng::CosmoPNG`: struct that contains all the information that may be used for the TPCF 
  computations of the PNG signal.

- `ss` : vector/range of `s` values where the function must be evaluated; if `ss = nothing`, 
  it is set `ss = 10 .^ range(0, log10(2 * cosmo.s_max), length=N_log)`. This is why it is returned 
  also the vector of the "input" values.

## Keyword arguments 

This function recall internally `ξ_S_multipole`, so the kwargs of the latter are valid also for the former; 
we report them for comfortness:

- `L::Int = 0`: order of the Legendre polynomial to be used

- `use_windows::Bool = false`: tells if the integrand must consider ``\\mathcal{F}``
  or not.

- `atol_quad::Float64 = 0.0` and `rtol_quad::Float64 = 1e-2`: absolute and relative tolerance
  to be passed to the function `quadgk`; it's recommended not to set `rtol_quad < 1e-2` 
  because the time for evaluation increase quickly.

- `enhancer::Float64 = 1e6`: just a float number used in order to deal better with small numbers; 
  the returned value is NOT modified by this value, because after a multiplication
  the internal result is divided by `enhancer`.

- `N_log::Int = 1000` : number of points to be used in the default logaritmically-spaced 
  range for `ss`, i.e. `range(0, log10(2 * cosmo.s_max), length=N_log)`; it is ignored if `ss ≠ nothing` 

- `pr::Bool = true` : do you want the progress bar showed on screen, in order to 
  check the time needed for the computation? (`true` recommended)

- `b = nothing` : galaxy bias value to be considered in the computation; if set to `nothing`, its value
  is inferred from the other keyword argument `sp`;

- `sp::Int64=1` : galaxy type to be considered between the two species stored in `cosmo`: it can be either `1` or `2`,
  and the corresponding set of galaxy, magnification and evolutionary biases stored in `cosmo` are considered;
  in case `b` is set to a value, this parameter is ignored.

# Returns

A `Tuple{Vector{Float64}, Vector{Float64}}`, which has as first element the `ss` vector
and as second one the corresponding ξ value evaluated.

See also: [`ξ_S`](@ref), [`integrand_ξ_S_multipole`](@ref), 
[`ξ_S_multipole`](@ref), [`print_map_ξ_S_multipole`](@ref)
[`WindowFIntegrated`](@ref), [`Cosmology`](@ref), [`CosmoPNG`](@ref), 
"""
function map_ξ_S_multipole(
    cosmo::Cosmology, cosmopng::CosmoPNG,
    ss=nothing;
    pr::Bool=true,
    N_log::Int=1000,
    L::Int=0,
    kwargs...)

    t1 = time()
    v_ss = isnothing(ss) ? 10 .^ range(0, log10(2 * cosmo.s_max), length=N_log) : ss
    xis = pr ? begin
        @showprogress "ξ_S, L=$L: " [
            ξ_S_multipole(s, cosmo, cosmopng; L=L, kwargs...) for s in v_ss
        ]
    end : [
        ξ_S_multipole(s, cosmo, cosmopng; L=L, kwargs...) for s in v_ss
    ]

    t2 = time()
    pr && println("\ntime needed for map_ξ_S_multipole " *
                "[in s] = $(@sprintf("%.5f", t2-t1)) ")
    return (v_ss, xis)
end



##########################################################################################92


"""
    print_map_ξ_S_multipole(
        cosmo::Cosmology, cosmopng::CosmoPNG, 
        out::String, ss = nothing;
        L::Int = 0, use_windows::Bool = true,
        atol_quad::Float64 = 0.0,
        rtol_quad::Float64 = 1e-2,
        enhancer::Float64 = 1e6,
        pr::Bool = true,
        N_log::Int = 1000,
        kwargs...)


Evaluate the multipole of order `L` of the Two-Point Correlation Function (TPCF) 
concerning the signal (S) of the local Primordial Non-Gaussianities (PNG),
for all the comoving distance values stored inside `ss`, 
and print the results (with all the options used) in a file named `out`.
If `ss = nothing`, it is set `ss = 10 .^ range(0, log10(2 * cosmo.s_max), length=N_log)`.

The function evaluated is then the following ``\\xi^{\\mathrm{S}} (s)``:

```math
    \\xi^{\\mathrm{S}} (s) = \\frac{2 L + 1}{2} \\int_{-1}^{+1} \\mathrm{d}\\mu \\; 
    \\xi^{\\mathrm{S}} \\left(s, \\mu\\right) 
          \\, \\mathcal{L}_L(\\mu) \\, \\times 
    \\begin{cases} 
        \\frac{1}{\\mathcal{N}}\\mathcal{F}(s, \\mu) \\quad \\mathrm{use\\_windows == true} \\\\
        1 \\quad\\quad \\mathrm{use\\_windows == false}
    \\end{cases}
```

where:
- ``\\xi^{\\mathrm{S}}(s,\\mu)`` is the TPCF of the PNG signal
  with the angular dependence, computed from `ξ_S`.
- ``\\mathcal{L}_L(\\mu)`` is the Legendre polynomial of order ``L``
- ``\\mathcal{F}(s, \\mu)`` is the integrated window function stored in `cosmo::Cosmology` (check the documentation of `WindowFIntegrated`)
- ``\\mathcal{N}`` is the integrated window function norm (check the documentation of `WindowFIntegrated`)

The integration over ``\\mu`` is preformed through the Julia function `quadgk` 
from the [`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl) Julia package, that uses an adaptive 
Gauss-Kronrod quadrature.

## Inputs

- `cosmo::Cosmology`: cosmology to be used in this computation

- `cosmopng::CosmoPNG`: struct that contains all the information that may be used for the TPCF 
  computations of the PNG signal.

- `out::String` : name of the file where the results must be stored.

- `ss` : vector/range of `s` values where the function must be evaluated; if `ss = nothing`, 
  it is set `ss = 10 .^ range(0, log10(2 * cosmo.s_max), length=N_log)`. This is why it is returned 
  also the vector of the "input" values.

## Keyword arguments 

This function recall internally `map_ξ_S_multipole`, so the kwargs of the latter are valid also for the former; 
we report them for comfortness:

- `L::Int = 0`: order of the Legendre polynomial to be used

- `use_windows::Bool = false`: tells if the integrand must consider ``\\mathcal{F}``
  or not.

- `atol_quad::Float64 = 0.0` and `rtol_quad::Float64 = 1e-2`: absolute and relative tolerance
  to be passed to the function `quadgk`; it's recommended not to set `rtol_quad < 1e-2` 
  because the time for evaluation increase quickly.

- `enhancer::Float64 = 1e6`: just a float number used in order to deal better with small numbers; 
  the returned value is NOT modified by this value, because after a multiplication
  the internal result is divided by `enhancer`.

- `N_log::Int = 1000` : number of points to be used in the default logaritmically-spaced 
  range for `ss`, i.e. `range(0, log10(2 * cosmo.s_max), length=N_log)`; it is ignored if `ss ≠ nothing` 

- `pr::Bool = true` : do you want the progress bar showed on screen, in order to 
  check the time needed for the computation? (`true` recommended)

- `b = nothing` : galaxy bias value to be considered in the computation; if set to `nothing`, its value
  is inferred from the other keyword argument `sp`;

- `sp::Int64=1` : galaxy type to be considered between the two species stored in `cosmo`: it can be either `1` or `2`,
  and the corresponding set of galaxy, magnification and evolutionary biases stored in `cosmo` are considered;
  in case `b` is set to a value, this parameter is ignored.

See also: [`ξ_S`](@ref), [`integrand_ξ_S_multipole`](@ref), 
[`ξ_S_multipole`](@ref), [`map_ξ_S_multipole`](@ref)
[`WindowFIntegrated`](@ref), [`Cosmology`](@ref), [`CosmoPNG`](@ref)
"""
function print_map_ξ_S_multipole(
    cosmo::Cosmology, cosmopng::CosmoPNG,
    out::String,
    v_ss=nothing;
    L::Int=0,
    kwargs...)

    check_parent_directory(out)
    check_namefile(out)

    t1 = time()
    vec = map_ξ_S_multipole(cosmo, cosmopng, v_ss; L=L, kwargs...)
    t2 = time()

    isfile(out) && run(`rm $out`)
    open(out, "w") do io
        println(io, BRAND)

        println(io, "#\n# This is an integration map on mu of the ξ L=$L multipole of S,\n" *
                    "# which is the difference between the Power Spectrum with PNG and with f_NL = 0.")
        println(io, "# Transfer function read from the file: $(cosmopng.file_TF)")
        print(io, "# Parameters used for the considered CosmoPNG: ")
        print(io, "\n")
        println(io, "# \t D = $(cosmopng.params.D) \t bf = $(cosmopng.params.bf)")
        println(io, "# \t flm_0 = $(cosmopng.params.flm_0) \t flM_0 = $(cosmopng.params.flM_0) \t N_0 = $(cosmopng.params.N_0)")
        println(io, "# \t kmin_0 = $(cosmopng.params.kmin_0) \t kmax_0 = $(cosmopng.params.kmax_0) \t s0_0 = $(cosmopng.params.s0_0)")
        println(io, "# \t flm_2 = $(cosmopng.params.flm_2) \t flM_2 = $(cosmopng.params.flM_2) \t N_2 = $(cosmopng.params.N_2)")
        println(io, "# \t kmin_2 = $(cosmopng.params.kmin_2) \t kmax_2 = $(cosmopng.params.kmax_2) \t s0_2 = $(cosmopng.params.s0_2)")
        println(io, "#")

        parameters_used(io, cosmo; logo=false)
        println(io, "# computational time needed (in s) : $(@sprintf("%.4f", t2-t1))")
        print(io, "# kwards passed: ")

        println(io, "\n# \t\tL = $L")
        if !isempty(kwargs)
            for key in keys(kwargs)
                println(io, "# \t\t$(key) = $(kwargs[key])")
            end
        end

        println(io, "# ")
        println(io, "# s [Mpc/h_0] \t \t xi")
        for (s, xi) in zip(vec[1], vec[2])
            println(io, "$s \t $xi")
        end
    end
end


