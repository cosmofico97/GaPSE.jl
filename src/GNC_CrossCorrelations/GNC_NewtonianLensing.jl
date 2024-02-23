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


function integrand_ξ_GNC_Newtonian_Lensing(
    IP::Point, P1::Point, P2::Point, y, cosmo::Cosmology; 
    Δχ_min::Float64=1e-1, b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 
    𝑓_evo1=nothing, 𝑓_evo2=nothing, s_lim=nothing,
    obs::Union{Bool,Symbol}=:noobsvel)

    s1, D_s1, f_s1 = P1.comdist, P1.D, P1.f
    s2 = P2.comdist
    χ2, D2, a2 = IP.comdist, IP.D, IP.a

    Ω_M0 = cosmo.params.Ω_M0
    b_s1 = isnothing(b1) ? cosmo.params.b1 : b1
    s_b_s2 = isnothing(s_b2) ? cosmo.params.s_b2 : s_b2

    Δχ2_square = s1^2 + χ2^2 - 2 * s1 * χ2 * y
    Δχ2 = Δχ2_square > 0 ? √(Δχ2_square) : 0

    common = D_s1 * ℋ0^2 * Ω_M0 * D2 * (χ2 - s2) * (5 * s_b_s2 - 2) / (a2 * s2)

    if Δχ2 ≥ Δχ_min
        new_J00 = 1 / 5 * (f_s1 * χ2 * (3 * y^2 - 1) - 3 * y * s1 * f_s1 - 5 * y * s1 * b_s1)
        new_J02 = 1 / (14 * Δχ2^2) * (
            7 * s1 * b_s1 * (-2 * χ2^2 * y + χ2 * s1 * (y^2 + 3) - 2 * y * s1^2) +
            f_s1 * (
            4 * χ2^3 * (3 * y^2 - 1) - 2 * χ2^2 * y * s1 * (3 * y^2 + 8)
            +
            χ2 * s1^2 * (9 * y^2 + 11) - 6 * y * s1^3
            )
        )
        new_J04 = 3 / (70 * Δχ2^4) * f_s1 * (
            χ2^5 * (6 * y^2 - 2) + 6 * χ2^4 * y * s1 * (y^2 - 3)
            -
            χ2^3 * s1^2 * (y^4 + 12 * y^2 - 21)
            +
            2 * χ2^2 * y * s1^3 * (y^2 + 3) - 12 * χ2 * s1^4
            +
            4 * y * s1^5
        )

        I00 = cosmo.tools.I00(Δχ2)
        I20 = cosmo.tools.I20(Δχ2)
        I40 = cosmo.tools.I40(Δχ2)

        return common * (new_J00 * I00 + new_J02 * I20 + new_J04 * I40)
    else
        #return - 2.0 / 3.0 * Δχ2 * f(Δχ2) * D(Δχ2) * cosmo.tools.σ_2^2.0
        #return - 2.0 / 3.0 * common * χ2 * f_s1 * cosmo.tools.σ_2^2
        return -common * s1 * (f_s1 + 5 * b_s1) * cosmo.tools.σ_0 / 5.0
    end
end


function integrand_ξ_GNC_Newtonian_Lensing(
    χ2::Float64, s1::Float64, s2::Float64,
    y, cosmo::Cosmology;
    kwargs...)

    P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
    IP = Point(χ2, cosmo)
    return integrand_ξ_GNC_Newtonian_Lensing(IP, P1, P2, y, cosmo; kwargs...)
end



"""
    integrand_ξ_GNC_Newtonian_Lensing(
        IP::Point, P1::Point, P2::Point, y, cosmo::Cosmology;
        Δχ_min::Float64=1e-1, b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 
        𝑓_evo1=nothing, 𝑓_evo2=nothing, s_lim=nothing,
        obs::Union{Bool,Symbol}=:noobsvel
        ) ::Float64

    integrand_ξ_GNC_Newtonian_Lensing(
        χ2::Float64, s1::Float64, s2::Float64,
        y, cosmo::Cosmology;
        kwargs... )::Float64

Return the integrand of the Two-Point Correlation Function (TPCF) 
given by the cross correlation between the Newtonian and the Lensing effects 
arising from the Galaxy Number Counts (GNC).

In the first method, you should pass the two extreme `Point`s (`P1` and `P2`) and the 
intermediate integrand `Point` (`IP`) where to 
evaluate the function. In the second method (that internally recalls the first),
you must provide the three corresponding comoving distances `s1`, `s2`, `χ2`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of this integrand is the following:

```math
\\begin{equation}
    f^{\\delta\\kappa} (\\chi_2, s_1, s_2, y) =
    D_1 J^{\\delta \\kappa}_{\\alpha}
    \\left[ 
        J^{\\delta \\kappa}_{00} I_0^0 ( \\Delta\\chi_2 ) + 
        J^{\\delta \\kappa}_{02} I_2^0 ( \\Delta \\chi_2 ) + 
        J^{\\delta \\kappa}_{04} I_4^0 ( \\Delta \\chi_2 ) 
    \\right] \\, ,
\\end{equation}
```

with

```math
\\begin{split}
    J^{\\delta \\kappa}_{\\alpha} &=
    \\frac{
        \\mathcal{H}_0 ^2 \\Omega_{\\mathrm{M}0} D (\\chi_2)
    }{
        a(\\chi_2 ) s_2
    } 
    (\\chi_2 - s_2 ) (5s_{\\mathrm{b}, 2} - 2) 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    J^{\\delta \\kappa}_{00} &=
        \\frac{1}{5}
        \\left[
            (3 y^2 - 1) \\chi_2 f_1 - y s_1(3 f_1 + 5 b_1) 
        \\right] 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    J^{\\delta \\kappa}_{02} &=
        \\frac{1}{14 \\Delta\\chi_2^2} 
        \\left\\{
            4 f_1 (3 y^2 - 1) \\chi_2^3 - 
            2 y 
            \\left[
                (3 y^2 + 8) f_1 + 7 b_1
            \\right] s_1 \\chi_2^2 +
            \\right. \\nonumber \\\\
            &\\left.\\qquad \\qquad\\qquad
            \\left[
                (9 y^2 + 11) f_1 - 7 (y^2 + 3) b_1
            \\right] s_1^2 \\chi_2 -
            2 y \\left[7 b_1 + 3 f_1 \\right] s_1^3
        \\right\\} 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    J^{\\delta \\kappa}_{04} &=
    \\frac{f_1}{70 \\Delta\\chi_2^4 }
    \\left\\{
        (6 y^2 - 2) \\chi_2^5 +
        6 y (y^2 - 3) s_1 \\chi_2^4 -
        (y^4 + 12 y^2 - 21) s_1^2 \\chi_2^3 +
        \\right.\\nonumber \\\\
        &\\left.\\qquad\\qquad\\qquad
        2 y (y^2 + 3) s_1^3 \\chi_2^2 -
        12 \\chi_2 s_1^4 + 
        4 y s_1 ^5
    \\right\\} 
    \\, ,
\\end{split}
```

where:

- ``s_1`` and ``s_2`` are comoving distances;

- ``D_1 = D(s_1)``, ... is the linear growth factor (evaluated in ``s_1``);

- ``a_1 = a(s_1)``, ... is the scale factor (evaluated in ``s_1``);

- ``f_1 = f(s_1)``, ... is the linear growth rate (evaluated in ``s_1``);

- ``\\mathcal{H}_1 = \\mathcal{H}(s_1)``, ... is the comoving 
  Hubble distances (evaluated in ``s_1``);

- ``y = \\cos{\\theta} = \\hat{\\mathbf{s}}_1 \\cdot \\hat{\\mathbf{s}}_2``;

- ``\\mathcal{R}_1 = \\mathcal{R}(s_1)``, ... is 
  computed by `func_ℛ_GNC` in `cosmo::Cosmology` (and evaluated in ``s_1`` );
  the definition of ``\\mathcal{R}(s)`` is the following:
  ```math
  \\mathcal{R}(s) = 5 s_{\\mathrm{b}}(s) + \\frac{2 - 5 s_{\\mathrm{b}}(s)}{\\mathcal{H}(s) \\, s} +  
  \\frac{\\dot{\\mathcal{H}}(s)}{\\mathcal{H}(s)^2} - \\mathit{f}_{\\mathrm{evo}} \\quad ;
  ```

- ``b_1``, ``s_{\\mathrm{b}, 1}``, ``\\mathit{f}_{\\mathrm{evo}, 1}`` 
  (and ``b_2``, ``s_{\\mathrm{b}, 2}``, ``\\mathit{f}_{\\mathrm{evo}, 2}``) : 
  galaxy bias, magnification bias (i.e. the slope of the luminosity function at the luminosity threshold), 
  and evolution bias for the first (second) effect;

- ``\\Omega_{\\mathrm{M}0} = \\Omega_{\\mathrm{cdm}} + \\Omega_{\\mathrm{b}}`` is the sum of 
  cold-dark-matter and barionic density parameters (again, stored in `cosmo`);

- ``I_\\ell^n`` and ``\\sigma_i`` are defined as
  ```math
  I_\\ell^n(s) = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^2 \\, P(q) \\, \\frac{j_\\ell(qs)}{(qs)^n} \\quad , 
  \\quad \\sigma_i = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^{2-i} \\, P(q)
  ```
  with ``P(q)`` as the matter Power Spectrum at ``z=0`` (stored in `cosmo`) 
  and ``j_\\ell`` as spherical Bessel function of order ``\\ell``;

- ``\\tilde{I}_0^4`` is defined as
  ```math
  \\tilde{I}_0^4 = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^2 \\, P(q) \\, \\frac{j_0(qs) - 1}{(qs)^4}
  ``` 
  with ``P(q)`` as the matter Power Spectrum at ``z=0`` (stored in `cosmo`) 
  and ``j_\\ell`` as spherical Bessel function of order ``\\ell``;

- ``\\mathcal{H}_0``, ``f_0`` and so on are evaluated at the observer position (i.e. at present day);

- ``\\Delta\\chi_1 := \\sqrt{\\chi_1^2 + s_2^2-2\\,\\chi_1\\,s_2\\,y}`` and 
  ``\\Delta\\chi_2 := \\sqrt{s_1^2 + \\chi_2^2-2\\,s_1\\,\\chi_2\\,y}``;

- ``s=\\sqrt{s_1^2 + s_2^2 - 2 \\, s_1 \\, s_2 \\, y}`` and 
  ``\\Delta\\chi := \\sqrt{\\chi_1^2 + \\chi_2^2-2\\,\\chi_1\\,\\chi_2\\,y}``.

In this TPCF there are no observer terms. The `obs` keyword is inserted only for compatibility with 
the other GNC TPCFs.

This function is used inside `ξ_GNC_Newton_Lensing` with the trapz() from the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package.


## Inputs

-  `IP::Point`, `P1::Point`, `P2::Point` or `χ2`,`s1`,`s2`: `Point`/comoving 
  distances where the TPCF has to be calculated; they contain all the 
  data of interest needed for this calculus (comoving distance, growth factor and so on).
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer

- `cosmo::Cosmology`: cosmology to be used in this computation; it contains all the splines
  used for the conversion `s` -> `Point`, and all the cosmological parameters ``b``, ...

## Keyword Arguments

- `b1=nothing`, `s_b1=nothing`, `𝑓_evo1=nothing` and `b2=nothing`, `s_b2=nothing`, `𝑓_evo2=nothing`:
  galaxy, magnification and evolutionary biases respectively for the first and the second effect 
  computed in this TPCF:
  - if not set (i.e. if you leave the default value `nothing`) the values stored in the input `cosmo`
    will be considered;
  - if you set one or more values, they will override the `cosmo` ones in this computation;
  - the two sets of values should be different only if you are interested in studing two galaxy species;
  - only the required parameters for the chosen TPCF will be used, depending on its analytical expression;
    all the others will have no effect, we still inserted them for pragmatical code compatibilities. 

- `s_lim=nothing` : parameter used in order to avoid the divergence of the ``\\mathcal{R}`` and 
  ``\\mathfrak{R}`` denominators: when ``0 \\leq s \\leq s_\\mathrm{lim}`` the returned values are
  ```math
  \\forall \\, s \\in [ 0, s_\\mathrm{lim} ] \\; : \\quad 
      \\mathfrak{R}(s) = 1 - \\frac{1}{\\mathcal{H}_0 \\, s_\\mathrm{lim}} \\; , \\quad
      \\mathcal{R}(s) = 5 s_{\\mathrm{b}} + 
          \\frac{2 - 5 s_{\\mathrm{b}}}{\\mathcal{H}_0 \\, s_\\mathrm{lim}} +  
          \\frac{\\dot{\\mathcal{H}}}{\\mathcal{H}_0^2} - \\mathit{f}_{\\mathrm{evo}} \\; .
  ```
  If `nothing`, the default value stored in `cosmo` will be considered.

- `obs::Union{Bool,Symbol} = :noobsvel` : do you want to consider the observer terms in the computation of the 
  chosen GNC TPCF effect?
  - `:yes` or `true` -> all the observer effects will be considered
  - `:no` or `false` -> no observer term will be taken into account
  - `:noobsvel` -> the observer terms related to the observer velocity (that you can find in the CF concerning Doppler)
    will be neglected, the other ones will be taken into account

- `Δχ_min::Float64 = 1e-4` : when ``\\Delta\\chi_2 = \\sqrt{s_1^2 + \\chi_2^2 - 2 \\, s_1 \\chi_2 y} \\to 0^{+}``,
  some ``I_\\ell^n`` term diverges, but the overall parenthesis has a known limit:

  ```math
    \\lim_{\\Delta \\chi_2 \\to 0^{+}} 
    \\left(
    J_{00}^{\\delta \\kappa} \\, I^0_0(\\Delta \\chi_2 ) + 
    J_{02}^{\\delta \\kappa} \\, I^0_2(\\Delta \\chi_2 ) + 
    J_{04}^{\\delta \\kappa} \\, I^4_0(\\Delta \\chi_2 ) 
    \\right) = 
    - \\frac{1}{5} \\, s_1 \\left(f_1 + 5 b_1\\right) \\, \\sigma_0
  ```

  So, when it happens that ``\\Delta \\chi_2 < \\Delta\\chi_\\mathrm{min}``, the function considers this limit
  as the result of the parenthesis instead of calculating it in the normal way; it prevents
  computational divergences.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_GNC_multipole`](@ref), 
[`map_ξ_GNC_multipole`](@ref), [`print_map_ξ_GNC_multipole`](@ref),
[`ξ_GNC_Newtonian_Lensing`](@ref)
"""
integrand_ξ_GNC_Newtonian_Lensing



##########################################################################################92



function ξ_GNC_Newtonian_Lensing(s1, s2, y, cosmo::Cosmology;
    en::Float64=1e6, N_χs::Int=100, suit_sampling::Bool=true,
    kwargs...)

    STARTING = 0.0
    frac_begin, frac_middle, FRAC_s = 0.6, 0.6, 0.10

    χ2s = if suit_sampling == false
        s2 .* range(STARTING, 1.0, length=N_χs)
    else
        if s2 < (1.0 - FRAC_s) * s1
            s2 .* range(STARTING, 1.0, length=N_χs)

        elseif (1.0 - FRAC_s) * s1 ≤ s2 ≤ (1.0 + FRAC_s) * s1
            GaPSE.sample_subdivision_begin(STARTING, (1.0 - FRAC_s) * s1, s2;
            frac_begin=frac_begin, N=N_χs, ass=false)

        elseif (1.0 + FRAC_s) * s1 < s2
            GaPSE.sample_subdivision_middle(
            STARTING, (1.0 - FRAC_s) * s1, (1.0 + FRAC_s) * s1, s2;
            frac_middle=frac_middle, N=N_χs, ass=false
            )
        else
            throw(AssertionError("how the hell did you arrived here?"))
        end
    end

    P1, P2 = GaPSE.Point(s1, cosmo), GaPSE.Point(s2, cosmo)
    IPs = [GaPSE.Point(x, cosmo) for x in χ2s]

    int_ξs = [
        en * GaPSE.integrand_ξ_GNC_Newtonian_Lensing(IP, P1, P2, y, cosmo; kwargs...)
        for IP in IPs
    ]

    return trapz(χ2s, int_ξs) / en
end


"""
    ξ_GNC_Newtonian_Lensing(s1, s2, y, cosmo::Cosmology;
        en::Float64=1e6, N_χs::Int=100, Δχ_min::Float64=1e-1,
        obs::Union{Bool,Symbol}=:noobsvel, 
        b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 
        𝑓_evo1=nothing, 𝑓_evo2=nothing, s_lim=nothing,
        suit_sampling::Bool = true )::Float64

Return the Two-Point Correlation Function (TPCF) given by the cross correlation between the 
Newtonian and the Lensing effects arising from the Galaxy Number Counts (GNC).

You must provide the two comoving distances `s1` and `s2` where to 
evaluate the function.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of this TPCF is the following:

```math
\\begin{equation}
    \\xi^{\\delta \\kappa} ( s_1 , s_2, y ) =
    D_1  \\int_0^{s_2}\\mathrm{d} \\chi_2
    J^{\\delta \\kappa}_{\\alpha}
    \\left[ 
        J^{\\delta \\kappa}_{00} I_0^0 ( \\Delta\\chi_2 ) + 
        J^{\\delta \\kappa}_{02} I_2^0 ( \\Delta \\chi_2 ) + 
        J^{\\delta \\kappa}_{04} I_4^0 ( \\Delta \\chi_2 ) 
    \\right] \\, ,
\\end{equation}
```

with

```math
\\begin{split}
    J^{\\delta \\kappa}_{\\alpha} &=
    \\frac{
        \\mathcal{H}_0 ^2 \\Omega_{\\mathrm{M}0} D (\\chi_2)
    }{
        a(\\chi_2 ) s_2
    } 
    (\\chi_2 - s_2 ) (5s_{\\mathrm{b}, 2} - 2) 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    J^{\\delta \\kappa}_{00} &=
        \\frac{1}{5}
        \\left[
            (3 y^2 - 1) \\chi_2 f_1 - y s_1(3 f_1 + 5 b_1) 
        \\right] 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    J^{\\delta \\kappa}_{02} &=
        \\frac{1}{14 \\Delta\\chi_2^2} 
        \\left\\{
            4 f_1 (3 y^2 - 1) \\chi_2^3 - 
            2 y 
            \\left[
                (3 y^2 + 8) f_1 + 7 b_1
            \\right] s_1 \\chi_2^2 +
            \\right. \\nonumber \\\\
            &\\left.\\qquad \\qquad\\qquad
            \\left[
                (9 y^2 + 11) f_1 - 7 (y^2 + 3) b_1
            \\right] s_1^2 \\chi_2 -
            2 y \\left[7 b_1 + 3 f_1 \\right] s_1^3
        \\right\\} 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    J^{\\delta \\kappa}_{04} &=
    \\frac{f_1}{70 \\Delta\\chi_2^4 }
    \\left\\{
        (6 y^2 - 2) \\chi_2^5 +
        6 y (y^2 - 3) s_1 \\chi_2^4 -
        (y^4 + 12 y^2 - 21) s_1^2 \\chi_2^3 +
        \\right.\\nonumber \\\\
        &\\left.\\qquad\\qquad\\qquad
        2 y (y^2 + 3) s_1^3 \\chi_2^2 -
        12 \\chi_2 s_1^4 + 
        4 y s_1 ^5
    \\right\\} 
    \\, ,
\\end{split}
```

where:

- ``s_1`` and ``s_2`` are comoving distances;

- ``D_1 = D(s_1)``, ... is the linear growth factor (evaluated in ``s_1``);

- ``a_1 = a(s_1)``, ... is the scale factor (evaluated in ``s_1``);

- ``f_1 = f(s_1)``, ... is the linear growth rate (evaluated in ``s_1``);

- ``\\mathcal{H}_1 = \\mathcal{H}(s_1)``, ... is the comoving 
  Hubble distances (evaluated in ``s_1``);

- ``y = \\cos{\\theta} = \\hat{\\mathbf{s}}_1 \\cdot \\hat{\\mathbf{s}}_2``;

- ``\\mathcal{R}_1 = \\mathcal{R}(s_1)``, ... is 
  computed by `func_ℛ_GNC` in `cosmo::Cosmology` (and evaluated in ``s_1`` );
  the definition of ``\\mathcal{R}(s)`` is the following:
  ```math
  \\mathcal{R}(s) = 5 s_{\\mathrm{b}}(s) + \\frac{2 - 5 s_{\\mathrm{b}}(s)}{\\mathcal{H}(s) \\, s} +  
  \\frac{\\dot{\\mathcal{H}}(s)}{\\mathcal{H}(s)^2} - \\mathit{f}_{\\mathrm{evo}} \\quad ;
  ```

- ``b_1``, ``s_{\\mathrm{b}, 1}``, ``\\mathit{f}_{\\mathrm{evo}, 1}`` 
  (and ``b_2``, ``s_{\\mathrm{b}, 2}``, ``\\mathit{f}_{\\mathrm{evo}, 2}``) : 
  galaxy bias, magnification bias (i.e. the slope of the luminosity function at the luminosity threshold), 
  and evolution bias for the first (second) effect;

- ``\\Omega_{\\mathrm{M}0} = \\Omega_{\\mathrm{cdm}} + \\Omega_{\\mathrm{b}}`` is the sum of 
  cold-dark-matter and barionic density parameters (again, stored in `cosmo`);

- ``I_\\ell^n`` and ``\\sigma_i`` are defined as
  ```math
  I_\\ell^n(s) = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^2 \\, P(q) \\, \\frac{j_\\ell(qs)}{(qs)^n} \\quad , 
  \\quad \\sigma_i = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^{2-i} \\, P(q)
  ```
  with ``P(q)`` as the matter Power Spectrum at ``z=0`` (stored in `cosmo`) 
  and ``j_\\ell`` as spherical Bessel function of order ``\\ell``;

- ``\\tilde{I}_0^4`` is defined as
  ```math
  \\tilde{I}_0^4 = \\int_0^{+\\infty} \\frac{\\mathrm{d}q}{2\\pi^2} 
  \\, q^2 \\, P(q) \\, \\frac{j_0(qs) - 1}{(qs)^4}
  ``` 
  with ``P(q)`` as the matter Power Spectrum at ``z=0`` (stored in `cosmo`) 
  and ``j_\\ell`` as spherical Bessel function of order ``\\ell``;

- ``\\mathcal{H}_0``, ``f_0`` and so on are evaluated at the observer position (i.e. at present day);

- ``\\Delta\\chi_1 := \\sqrt{\\chi_1^2 + s_2^2-2\\,\\chi_1\\,s_2\\,y}`` and 
  ``\\Delta\\chi_2 := \\sqrt{s_1^2 + \\chi_2^2-2\\,s_1\\,\\chi_2\\,y}``;

- ``s=\\sqrt{s_1^2 + s_2^2 - 2 \\, s_1 \\, s_2 \\, y}`` and 
  ``\\Delta\\chi := \\sqrt{\\chi_1^2 + \\chi_2^2-2\\,\\chi_1\\,\\chi_2\\,y}``.

In this TPCF there are no observer terms. The `obs` keyword is inserted only for compatibility with 
the other GNC TPCFs.

This function is computed integrating `integrand_ξ_GNC_Newtonian_Lensing` with trapz() from the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package.

## Inputs

- `s1` and `s2`: comoving distances where the TPCF has to be calculated.
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer

- `cosmo::Cosmology`: cosmology to be used in this computation; it contains all the splines
  used for the conversion `s` -> `Point`, and all the cosmological parameters ``b``, ...

## Keyword Arguments

- `b1=nothing`, `s_b1=nothing`, `𝑓_evo1=nothing` and `b2=nothing`, `s_b2=nothing`, `𝑓_evo2=nothing`:
  galaxy, magnification and evolutionary biases respectively for the first and the second effect 
  computed in this TPCF:
  - if not set (i.e. if you leave the default value `nothing`) the values stored in the input `cosmo`
    will be considered;
  - if you set one or more values, they will override the `cosmo` ones in this computation;
  - the two sets of values should be different only if you are interested in studing two galaxy species;
  - only the required parameters for the chosen TPCF will be used, depending on its analytical expression;
    all the others will have no effect, we still inserted them for pragmatical code compatibilities. 

- `s_lim=nothing` : parameter used in order to avoid the divergence of the ``\\mathcal{R}`` and 
  ``\\mathfrak{R}`` denominators: when ``0 \\leq s \\leq s_\\mathrm{lim}`` the returned values are
  ```math
  \\forall \\, s \\in [ 0, s_\\mathrm{lim} ] \\; : \\quad 
      \\mathfrak{R}(s) = 1 - \\frac{1}{\\mathcal{H}_0 \\, s_\\mathrm{lim}} \\; , \\quad
      \\mathcal{R}(s) = 5 s_{\\mathrm{b}} + 
          \\frac{2 - 5 s_{\\mathrm{b}}}{\\mathcal{H}_0 \\, s_\\mathrm{lim}} +  
          \\frac{\\dot{\\mathcal{H}}}{\\mathcal{H}_0^2} - \\mathit{f}_{\\mathrm{evo}} \\; .
  ```
  If `nothing`, the default value stored in `cosmo` will be considered.

- `en::Float64 = 1e6`: just a float number used in order to deal better 
  with small numbers;

- `Δχ_min::Float64 = 1e-4` : when ``\\Delta\\chi_2 = \\sqrt{s_1^2 + \\chi_2^2 - 2 \\, s_1 \\chi_2 y} \\to 0^{+}``,
  some ``I_\\ell^n`` term diverges, but the overall parenthesis has a known limit:

  ```math
    \\lim_{\\Delta \\chi_2 \\to 0^{+}} 
    \\left(
    J_{00}^{\\delta \\kappa} \\, I^0_0(\\Delta \\chi_2 ) + 
    J_{02}^{\\delta \\kappa} \\, I^0_2(\\Delta \\chi_2 ) + 
    J_{04}^{\\delta \\kappa} \\, I^4_0(\\Delta \\chi_2 ) 
    \\right) = 
    - \\frac{1}{5} \\, s_1 \\left(f_1 + 5 b_1\\right) \\, \\sigma_0
  ```

  So, when it happens that ``\\Delta \\chi_2 < \\Delta\\chi_\\mathrm{min}``, the function considers this limit
  as the result of the parenthesis instead of calculating it in the normal way; it prevents
  computational divergences.

- `N_χs::Int = 100`: number of points to be used for sampling the integral
  along the range `(0, s2)` (for `χ2`); it has been checked that
  with `N_χs ≥ 100` the result is stable.

- `suit_sampling::Bool = true` : this bool keyword can be found in all the TPCFs which have at least one `χ` integral;
  it is conceived to enable a sampling of the `χ` integral(s) suited for the given TPCF; however, it actually have an
  effect only in the TPCFs that have such a sampling implemented in the code.
  Currently, only `ξ_GNC_Newtonian_Lensing` (and its simmetryc TPCF) has it.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_GNC_multipole`](@ref), 
[`map_ξ_GNC_multipole`](@ref), [`print_map_ξ_GNC_multipole`](@ref),
[`integrand_ξ_GNC_Newtonian_Lensing`](@ref)
"""
ξ_GNC_Newtonian_Lensing


##########################################################################################92

##########################################################################################92

##########################################################################################92




"""
    ξ_GNC_Lensing_Newtonian(s1, s2, y, cosmo::Cosmology; 
        en::Float64=1e6, N_χs::Int=100, Δχ_min::Float64=1e-1,
        obs::Union{Bool,Symbol}=:noobsvel, 
        b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 
        𝑓_evo1=nothing, 𝑓_evo2=nothing, s_lim=nothing,
        suit_sampling::Bool = true )::Float64

Return the Two-Point Correlation Function (TPCF) given by the cross correlation between the 
Lensing and the Newtonian effects arising from the Galaxy Number Counts (GNC).

It's computed through the symmetric function `ξ_GNC_Newtonian_Lensing`; check its documentation for
more details about the analytical expression and the keyword arguments.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.


## Inputs

- `s1` and `s2`: comoving distances where the TPCF has to be calculated;
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer

- `cosmo::Cosmology`: cosmology to be used in this computation; it contains all the splines
  used for the conversion `s` -> `Point`, and all the cosmological parameters ``b``, ...

## Keyword Arguments

- `kwargs...` : Keyword arguments to be passed to the symmetric TPCF

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_GNC_multipole`](@ref), 
[`map_ξ_GNC_multipole`](@ref), [`print_map_ξ_GNC_multipole`](@ref),
[`ξ_GNC_Newtonian_Lensing`](@ref)
"""
function ξ_GNC_Lensing_Newtonian(s1, s2, y, cosmo::Cosmology; 
    b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing,
    𝑓_evo1=nothing, 𝑓_evo2=nothing, s_lim=nothing,
    kwargs...)
    

    b1 = isnothing(b1) ? cosmo.params.b1 : b1
    b2 = isnothing(b2) ? cosmo.params.b2 : b2
    s_b1 = isnothing(s_b1) ? cosmo.params.s_b1 : s_b1
    s_b2 = isnothing(s_b2) ? cosmo.params.s_b2 : s_b2
    𝑓_evo1 = isnothing(𝑓_evo1) ? cosmo.params.𝑓_evo1 : 𝑓_evo1
    𝑓_evo2 = isnothing(𝑓_evo2) ? cosmo.params.𝑓_evo2 : 𝑓_evo2

    ξ_GNC_Newtonian_Lensing(s2, s1, y, cosmo; b1=b2, b2=b1, s_b1=s_b2, s_b2=s_b1,
        𝑓_evo1=𝑓_evo2, 𝑓_evo2=𝑓_evo1, s_lim=s_lim, kwargs...)
end
