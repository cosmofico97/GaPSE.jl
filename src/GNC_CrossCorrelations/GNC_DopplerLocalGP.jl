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



function ξ_GNC_Doppler_LocalGP(P1::Point, P2::Point, y, cosmo::Cosmology; 
    b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 𝑓_evo1=nothing, 𝑓_evo2=nothing,
    s_lim=nothing, obs::Union{Bool,Symbol}=:noobsvel)
    
    s1, D1, f1, ℋ1 = P1.comdist, P1.D, P1.f, P1.ℋ
    s2, D2, f2, a2, ℋ2 = P2.comdist, P2.D, P2.f, P2.a, P2.ℋ

    Ω_M0 = cosmo.params.Ω_M0
    s_b1 = isnothing(s_b1) ? cosmo.params.s_b1 : s_b1
    s_b2 = isnothing(s_b2) ? cosmo.params.s_b2 : s_b2
    𝑓_evo1 = isnothing(𝑓_evo1) ? cosmo.params.𝑓_evo1 : 𝑓_evo1
    𝑓_evo2 = isnothing(𝑓_evo2) ? cosmo.params.𝑓_evo2 : 𝑓_evo2

    s_lim = isnothing(s_lim) ? cosmo.params.s_lim : s_lim
    ℛ1 = func_ℛ_GNC(s1, P1.ℋ, P1.ℋ_p; s_b=s_b1, 𝑓_evo=𝑓_evo1, s_lim=s_lim)
    ℛ2 = func_ℛ_GNC(s2, P2.ℋ, P2.ℋ_p; s_b=s_b2, 𝑓_evo=𝑓_evo2, s_lim=s_lim)

    Δs = s(s1, s2, y)

    common = Δs^2 * f1 * ℋ1 * ℛ1 * (s1 - y * s2) / a2
    parenth = 2 * f2 * a2 * ℋ2^2 * (𝑓_evo2 - 3) + 3 * ℋ0^2 * Ω_M0 * (f2 + ℛ2 + 5 * s_b2 - 2)

    I00 = cosmo.tools.I00(Δs)
    I20 = cosmo.tools.I20(Δs)
    I40 = cosmo.tools.I40(Δs)
    I02 = cosmo.tools.I02(Δs)

    if obs == false || obs == :no
        return D1 * D2 * common * parenth * (
                    -1 / 90 * I00 - 1 / 63 * I20
                    -
                    1 / 210 * I40 - 1 / 6 * I02
                )
    elseif obs == true || obs == :yes
        #### New observer terms #########

        I13_s1 = cosmo.tools.I13(s1)
        I13_s2 = cosmo.tools.I13(s2)

        J31_a = ℋ0 * s1^3 * f1 * ℋ1 * ℛ1 * (ℋ0 * s2 * ℛ2 * (3 * Ω_M0 - 2 * f0) + 2 * f0 * (2 - 5 * s_b2)) / (2 * s2)
        J31_b = -y * f0 * ℋ0 * s2^3 * (ℛ1 - 5 * s_b1 + 2) / (2 * a2) * parenth

        obs_terms = D1 * J31_a * I13_s1 + D2 * J31_b * I13_s2

        #################################

        return D1 * D2 * common * parenth * (
                    -1 / 90 * I00 - 1 / 63 * I20
                    -
                    1 / 210 * I40 - 1 / 6 * I02
                ) + obs_terms

    elseif obs == :noobsvel
        #### New observer terms #########

        I13_s1 = cosmo.tools.I13(s1)

        J31_a = ℋ0 * s1^3 * f1 * ℋ1 * ℛ1 * (ℋ0 * s2 * ℛ2 * (3 * Ω_M0 - 2 * f0) + 2 * f0 * (2 - 5 * s_b2)) / (2 * s2)

        obs_terms = D1 * J31_a * I13_s1

        #################################

        return D1 * D2 * common * parenth * (
                    -1 / 90 * I00 - 1 / 63 * I20
                    -
                    1 / 210 * I40 - 1 / 6 * I02
                ) + obs_terms

    else
        throw(AssertionError(":$obs is not a valid Symbol for \"obs\"; they are: \n\t" *
                            "$(":".*string.(VALID_OBS_VALUES) .* vcat([" , " for i in 1:length(VALID_OBS_VALUES)-1], " .")... )"
        ))
    end
end




function ξ_GNC_Doppler_LocalGP(s1, s2, y, cosmo::Cosmology; kwargs...)
    P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
    return ξ_GNC_Doppler_LocalGP(P1, P2, y, cosmo; kwargs...)
end


"""
    ξ_GNC_Doppler_LocalGP(
        P1::Point, P2::Point, y, cosmo::Cosmology; 
        b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 
        𝑓_evo1=nothing, 𝑓_evo2=nothing, s_lim=nothing,
        obs::Union{Bool,Symbol}=:noobsvel
        ) ::Float64

    ξ_GNC_Doppler_LocalGP(
        s1, s2, y, cosmo::Cosmology; 
        kwargs...) ::Float64

Returns the Two-Point Correlation Function (TPCF) given by the cross correlation between the 
Doppler and the Local Gravitational Potential (GP) effects arising from the Galaxy Number Counts (GNC).

In the first method, you should pass the two `Point` (`P1` and `P2`) where to 
evaluate the function, while in the second method (that internally recalls the first) 
you must provide the two corresponding comoving distances `s1` and `s2`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of this TPCF is the following:

```math
\\begin{split}
    \\xi^{v_{\\parallel} \\phi} ( s_1 , s_2, y ) &= 
    D_1 D_2 J^{v_{\\parallel} \\phi}_{\\alpha}
    \\left[ 
        \\frac{1}{90} I_0^0 (s) +
        \\frac{1}{63} I_2^0 (s) + 
        \\frac{1}{210} I_4^0 (s) +
        \\frac{1}{6} I_0^2 (s) 
    \\right]
    \\nonumber \\\\
    & +
    D_1 J^{v_{\\parallel} \\phi_0 }_{31} I^3_1 (s_1) +
    D_2 J^{v_{\\parallel, 0} \\phi }_{31} I^3_1 (s_2) \\, ,
\\end{split}
```

with

```math
\\begin{split}
    J^{v_{\\parallel} \\phi}_{\\alpha} &=
    \\frac{f_1 \\mathcal{H}_1 \\mathcal{R}_1 s^2}{a_2} (y s_2 - s_1) 
    \\times\\\\
    &\\qquad\\qquad\\qquad
    \\left[ 
        2 f_2 a_2 \\mathcal{H}_2^2 (\\mathit{f}_{\\mathrm{evo}, 2} - 3) + 
        3 \\mathcal{H}_0^2 \\Omega_{\\mathrm{M}0} (f_2 + \\mathcal{R}_2 + 5 s_{\\mathrm{b}, 2} - 2)
    \\right]
    \\, , \\nonumber \\\\
    %%%%%%%%%%%%%%%%%%%%
    J^{v_{\\parallel} \\phi_0 }_{31} &=
    \\frac{ f_1 \\mathcal{H}_1 \\mathcal{R}_1}{2 s_2} \\mathcal{H}_0 s_1^3 
    \\left[ 
        3 \\mathcal{H}_0 \\Omega_{\\mathrm{M}0} s_2 \\mathcal{R}_2 -
        2 f_0 \\left( \\mathcal{H}_0 s_2 \\mathcal{R}_2 + 5 s_{\\mathrm{b}, 2} - 2 \\right)
    \\right]
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%
    J^{v_{\\parallel, 0} \\,  \\phi }_{31} &=
    -\\frac{y f_0 \\mathcal{H}_0 s_2^3}{2 a_2} (\\mathcal{R}_1 - 5 s_{\\mathrm{b}, 1} + 2) 
    \\times \\\\
    &\\qquad\\qquad\\qquad
    \\left[ 
        2 a_2 f_2 \\mathcal{H}_2^2 (\\mathit{f}_{\\mathrm{evo}, 2} - 3) +
        3 \\mathcal{H}_0^2 \\Omega_{\\mathrm{M}0} (f_2 + \\mathcal{R}_2 + 5 s_{\\mathrm{b}, 2} - 2)
    \\right]
    \\nonumber \\, .
\\end{split}
```

where:

- ``s_1`` and ``s_2`` are comoving distances;

- ``D_1 = D(s_1)``, ... is the linear growth factor (evaluated in ``s_1``);

- ``a_1 = a(s_1)``, ... is the scale factor (evaluated in ``s_1``);

- ``f_1 = f(s_1)``, ... is the linear growth rate (evaluated in ``s_1``);

- ``\\mathcal{H}_1 = \\mathcal{H}(s_1)``, ... is the comoving 
  Hubble parameter (evaluated in ``s_1``, ...);

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

In this TPCF, the terms proportional to ``D(s_1)`` and ``D(s_2)`` are observer terms 
(while the term proportional to ``D(s_1) \\, D(s_2)`` is not), but only the ``\\propto D(s_2)`` one depends
on the observer velocity. Consequently, if you set:
- `obs = :yes` or `obs = true` all of them will be computed
- `obs = :noobsvel` then the  ``\\propto D(s_2)`` term will be neglected
- `obs = :no` or `obs = false` only the first one ``\\propto D(s_1) \\, D(s_2)`` will be taken into account.

## Inputs

- `P1::Point` and `P2::Point`, or `s1` and `s2`: `Point`/comoving distances where the 
  TPCF has to be calculated; they contain all the 
  data of interest needed for this calculus (comoving distance, growth factor and so on);
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer;

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
  - `:yes` or `true` -> all the observer effects will be considered;
  - `:no` or `false` -> no observer term will be taken into account;
  - `:noobsvel` -> the observer terms related to the observer velocity (that you can find in the CF concerning Doppler)
    will be neglected, the other ones will be taken into account.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_GNC_multipole`](@ref), 
[`map_ξ_GNC_multipole`](@ref), [`print_map_ξ_GNC_multipole`](@ref)
"""
ξ_GNC_Doppler_LocalGP




##########################################################################################92

##########################################################################################92

##########################################################################################92




"""
    ξ_GNC_LocalGP_Doppler(s1, s2, y, cosmo::Cosmology; 
        b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 
        𝑓_evo1=nothing, 𝑓_evo2=nothing,
        s_lim=nothing, obs::Union{Bool,Symbol}=:noobsvel ) ::Float64 

Returns the Two-Point Correlation Function (TPCF) given by the cross correlation between the 
Local Gravitational Potential (GP) and the Doppler effects arising from the Galaxy Number Counts (GNC).

It's computed through the symmetric function `ξ_GNC_Doppler_LocalGP`; check its documentation for
more details about the analytical expression and the keyword arguments.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.


## Inputs

- `s1` and `s2`: comoving distances where the TPCF has to be calculated;
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer;

- `cosmo::Cosmology`: cosmology to be used in this computation; it contains all the splines
  used for the conversion `s` -> `Point`, and all the cosmological parameters ``b``, ...

## Keyword Arguments

- `kwargs...` : Keyword arguments to be passed to the symmetric TPCF.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_GNC_multipole`](@ref), 
[`map_ξ_GNC_multipole`](@ref), [`print_map_ξ_GNC_multipole`](@ref),
[`ξ_GNC_Doppler_LocalGP`](@ref)
"""
function ξ_GNC_LocalGP_Doppler(s1, s2, y, cosmo::Cosmology; 
    b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 𝑓_evo1=nothing, 𝑓_evo2=nothing,
    s_lim=nothing, kwargs...)

    b1 = isnothing(b1) ? cosmo.params.b1 : b1
    b2 = isnothing(b2) ? cosmo.params.b2 : b2
    s_b1 = isnothing(s_b1) ? cosmo.params.s_b1 : s_b1
    s_b2 = isnothing(s_b2) ? cosmo.params.s_b2 : s_b2
    𝑓_evo1 = isnothing(𝑓_evo1) ? cosmo.params.𝑓_evo1 : 𝑓_evo1
    𝑓_evo2 = isnothing(𝑓_evo2) ? cosmo.params.𝑓_evo2 : 𝑓_evo2

    ξ_GNC_Doppler_LocalGP(s2, s1, y, cosmo; b1=b2, b2=b1, s_b1=s_b2, s_b2=s_b1,
        𝑓_evo1=𝑓_evo2, 𝑓_evo2=𝑓_evo1, s_lim=s_lim, kwargs...)
end

