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


function integrand_ξ_LD_Lensing_Doppler(
    IP::Point, P1::Point, P2::Point,
    y, cosmo::Cosmology)

    s1 = P1.comdist
    s2, D_s2, f_s2, ℋ_s2, ℛ_s2 = P2.comdist, P2.D, P2.f, P2.ℋ, P2.ℛ_LD
    χ1, D1, a1 = IP.comdist, IP.D, IP.a
    Ω_M0 = cosmo.params.Ω_M0


    Δχ1_square = χ1^2 + s2^2 - 2 * χ1 * s2 * y
    Δχ1 = Δχ1_square > 0.0 ? √(Δχ1_square) : 0.0

    common = ℋ0^2 * Ω_M0 * D1 * (χ1 - s1) / (s1 * a1)
    factor = D_s2 * f_s2 * ℋ_s2 * ℛ_s2

    new_J00 = 1.0 / 15.0 * (χ1^2 * y + χ1 * (4 * y^2 - 3) * s2 - 2 * y * s2^2)
    new_J02 = 1.0 / (42 * Δχ1^2) * (
        4 * χ1^4 * y + 4 * χ1^3 * (2 * y^2 - 3) * s2
        + χ1^2 * y * (11 - 23 * y^2) * s2^2
        + χ1 * (23 * y^2 - 3) * s2^3 - 8 * y * s2^4)
    new_J04 = 1.0 / (70 * Δχ1^2) * (
        2 * χ1^4 * y + 2 * χ1^3 * (2 * y^2 - 3) * s2
        -
        χ1^2 * y * (y^2 + 5) * s2^2
        +
        χ1 * (y^2 + 9) * s2^3 - 4 * y * s2^4)
    new_J20 = y * Δχ1^2

    I00 = cosmo.tools.I00(Δχ1)
    I20 = cosmo.tools.I20(Δχ1)
    I40 = cosmo.tools.I40(Δχ1)
    I02 = cosmo.tools.I02(Δχ1)

    #println("J00 = $new_J00, \t I00(Δχ1) = $(I00)")
    #println("J02 = $new_J02, \t I20(Δχ1) = $(I20)")
    #println("J31 = $new_J31, \t I13(Δχ1) = $(I13)")
    #println("J22 = $new_J22, \t I22(Δχ1) = $(I22)")

    parenth = (
        new_J00 * I00 + new_J02 * I20 +
        new_J04 * I40 + new_J20 * I02
    )

    first = common * factor * parenth

    #new_J31 = -3 * χ1^2 * y * f0 * ℋ0
    #I13 = cosmo.tools.I13(χ1)
    #second = new_J31 * I13 * common

    return first
end


function integrand_ξ_LD_Lensing_Doppler(
    χ1::Float64, s1::Float64, s2::Float64,
    y, cosmo::Cosmology)

    P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
    IP = Point(χ1, cosmo)
    return integrand_ξ_LD_Lensing_Doppler(IP, P1, P2, y, cosmo)
end




"""
    integrand_ξ_LD_Lensing_Doppler(
        IP::Point, P1::Point, P2::Point,
        y, cosmo::Cosmology ) ::Float64

    integrand_ξ_LD_Lensing_Doppler(
        χ1::Float64, s1::Float64, s2::Float64,
        y, cosmo::Cosmology ) ::Float64


Return the  integrand of the Two-Point Correlation Function (TPCF) given by 
the cross correlation between the 
Lensing and the Doppler effects arising from the Luminosity Distance (LD) perturbations.

In the first method, you should pass the two extreme `Point`s (`P1` and `P2`) and the 
intermediate integrand `Point` (`IP`) where to 
evaluate the function. In the second method (that internally recalls the first),
you must provide the three corresponding comoving distances `s1`, `s2`, `χ1`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of this integrand is the following:

```math
\\begin{split}
    f^{\\kappa v_{\\parallel}} (\\chi_1, s_1, s_2, y) = 
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{\\alpha} \\times \\\\
    &\\left[
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{00} I_0^0(\\Delta\\chi_1) + 
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{02} I_2^0(\\Delta\\chi_1) + 
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{04} I_4^0(\\Delta\\chi_1) + 
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{20} I_0^2(\\Delta\\chi_1)
    \\right] \\, ,
\\end{split}
```

with

```math
\\begin{split}
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{\\alpha} &= 
    \\mathcal{H}_0^2 \\Omega_{\\mathrm{M}0} D_2 f_2 \\mathcal{H}_2 \\mathfrak{R}_2 
    \\frac{D(\\chi_1) (\\chi_1 - s_1)}{a(\\chi_1) s_1}
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{00} & = 
    \\frac{1}{15}
    \\left[
        \\chi_1^2 y + \\chi_1(4 y^2 - 3) s_2 - 2 y s_2^2
    \\right]
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{02} & = 
    \\frac{1}{42 \\Delta\\chi_1^2}
    \\left[
        4 \\chi_1^4 y + 4 \\chi_1^3 (2 y^2 - 3) s_2 +
        \\chi_1^2 y (11 - 23 y^2) s_2^2 +
        \\right.\\\\
        &\\left.\\qquad\\qquad\\qquad
        \\chi_1 (23 y^2 - 3) s_2^3 - 8 y s_2^4
    \\right] \\nonumber
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{04} & = 
    \\frac{1}{70 \\Delta\\chi_1^2} 
    \\left[
        2\\chi_1^4 y + 2 \\chi_1^3 (2 y^2 - 3) s_2 -
        \\chi_1^2 y (y^2 + 5) s_2^2 + 
        \\right.\\\\
        &\\left.\\qquad\\qquad\\qquad
        \\chi_1(y^2 + 9) s_2^3 - 4 y s_2^4
    \\right] \\nonumber
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{20} & = y \\Delta\\chi_1^2 \\, ,
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

- ``\\mathfrak{R}_1 = \\mathfrak{R}(s_1)``, ... is 
  computed by `func_ℛ_LD` in `cosmo::Cosmology` (and evaluated in ``s_1`` );
  the definition of ``\\mathcal{R}(s)`` is the following:
  ```math
  \\mathfrak{R}(s) = 1 - \\frac{1}{\\mathcal{H}(s) s} ;
  ```

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

This function is used inside `ξ_LD_Lensing_Doppler` with trapz() from the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package.

## Inputs

- `IP::Point`, `P1::Point` and `P2::Point`, or `χ2`,`s1`,`s2`: `Point`/comoving distances where the 
  TPCF has to be calculated; they contain all the 
  data of interest needed for this calculus (comoving distance, growth factor and so on).
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer

- `cosmo::Cosmology`: cosmology to be used in this computation; it contains all the splines
  used for the conversion `s` -> `Point`, and all the cosmological parameters ``\\Omega_{\\mathrm{M}0}``, ...


See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_LD_multipole`](@ref), 
[`map_ξ_LD_multipole`](@ref), [`print_map_ξ_LD_multipole`](@ref)
"""
integrand_ξ_LD_Lensing_Doppler


##########################################################################################92


"""
    ξ_LD_Lensing_Doppler(
        s1, s2, y, cosmo::Cosmology;
        en::Float64 = 1e6, N_χs::Int = 100 ) ::Float64

Return the Two-Point Correlation Function (TPCF) given by the cross correlation between the 
Lensing and the Doppler effects arising from the Luminosity Distance (LD) perturbations.

You must provide the two comoving distances `s1` and `s2` where to 
evaluate the function.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of this integrand is the following:

```math
\\begin{split}
    \\xi^{\\kappa v_{\\parallel}} (s_1, s_2, y) = 
    \\int_0^{s_1} &\\mathrm{d}\\chi_1 \\; 
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{\\alpha} \\times \\\\
    &\\left[
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{00} I_0^0(\\Delta\\chi_1) + 
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{02} I_2^0(\\Delta\\chi_1) + 
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{04} I_4^0(\\Delta\\chi_1) + 
        \\mathcal{J}^{\\kappa v_{\\parallel}}_{20} I_0^2(\\Delta\\chi_1)
    \\right] \\, ,
\\end{split}
```

with

```math
\\begin{split}
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{\\alpha} &= 
    \\mathcal{H}_0^2 \\Omega_{\\mathrm{M}0} D_2 f_2 \\mathcal{H}_2 \\mathfrak{R}_2 
    \\frac{D(\\chi_1) (\\chi_1 - s_1)}{a(\\chi_1) s_1}
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{00} & = 
    \\frac{1}{15}
    \\left[
        \\chi_1^2 y + \\chi_1(4 y^2 - 3) s_2 - 2 y s_2^2
    \\right]
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{02} & = 
    \\frac{1}{42 \\Delta\\chi_1^2}
    \\left[
        4 \\chi_1^4 y + 4 \\chi_1^3 (2 y^2 - 3) s_2 +
        \\chi_1^2 y (11 - 23 y^2) s_2^2 +
        \\right.\\\\
        &\\left.\\qquad\\qquad\\qquad
        \\chi_1 (23 y^2 - 3) s_2^3 - 8 y s_2^4
    \\right] \\nonumber
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{04} & = 
    \\frac{1}{70 \\Delta\\chi_1^2} 
    \\left[
        2\\chi_1^4 y + 2 \\chi_1^3 (2 y^2 - 3) s_2 -
        \\chi_1^2 y (y^2 + 5) s_2^2 + 
        \\right.\\\\
        &\\left.\\qquad\\qquad\\qquad
        \\chi_1(y^2 + 9) s_2^3 - 4 y s_2^4
    \\right] \\nonumber
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%%%%%%%
    \\mathcal{J}^{\\kappa v_{\\parallel}}_{20} & = y \\Delta\\chi_1^2 \\, ,
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

- ``\\mathfrak{R}_1 = \\mathfrak{R}(s_1)``, ... is 
  computed by `func_ℛ_LD` in `cosmo::Cosmology` (and evaluated in ``s_1`` );
  the definition of ``\\mathcal{R}(s)`` is the following:
  ```math
  \\mathfrak{R}(s) = 1 - \\frac{1}{\\mathcal{H}(s) s} ;
  ```

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

This function is computed integrating `integrand_ξ_LD_Lensing_Doppler` with trapz() from the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package.

## Inputs

- `P1::Point` and `P2::Point`, or `s1` and `s2`: `Point`/comoving distances where the 
  TPCF has to be calculated; they contain all the 
  data of interest needed for this calculus (comoving distance, growth factor and so on).
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer

- `cosmo::Cosmology`: cosmology to be used in this computation; it contains all the splines
  used for the conversion `s` -> `Point`, and all the cosmological parameters ``\\Omega_{\\mathrm{M}0}``, ...


## Keyword Arguments

- `en::Float64 = 1e6`: just a float number used in order to deal better 
  with small numbers;

- `N_χs::Int = 100`: number of points to be used for sampling the integral
  along the range `(0, s1)` (for `χ1`); it has been checked that
  with `N_χs ≥ 50` the result is stable.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_LD_multipole`](@ref), 
[`map_ξ_LD_multipole`](@ref), [`print_map_ξ_LD_multipole`](@ref)
"""
function ξ_LD_Lensing_Doppler(s1, s2, y, cosmo::Cosmology;
    en::Float64=1e6, N_χs::Int=100)

    adim_χs = range(1e-6, 1.0, N_χs)
    χ1s = adim_χs .* s1

    P1, P2 = GaPSE.Point(s1, cosmo), GaPSE.Point(s2, cosmo)
    IPs = [GaPSE.Point(x, cosmo) for x in χ1s]

    int_ξs = [
        en * GaPSE.integrand_ξ_LD_Lensing_Doppler(IP, P1, P2, y, cosmo)
        for IP in IPs
    ]

    res = trapz(χ1s, int_ξs)
    #println("res = $res")
    return res / en
end





##########################################################################################92

##########################################################################################92

##########################################################################################92


"""
    ξ_LD_Doppler_Lensing(
        s1, s2, y, cosmo::Cosmology; 
        kwargs... ) ::Float64

Return the Two-Point Correlation Function (TPCF) given by the cross correlation between the 
Doppler and the Lensing effects arising from the 
Luminosity Distance (LD) perturbations.

It's computed through the symmetric function `ξ_LD_Lensing_Doppler`; check its documentation for
more details about the analytical expression and the keyword arguments.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.


## Inputs

- `s1` and `s2`: comoving distances where the TPCF has to be calculated;
  
- `y`: the cosine of the angle between the two points `P1` and `P2` wrt the observer

- `cosmo::Cosmology`: cosmology to be used in this computation; it contains all the splines
  used for the conversion `s` -> `Point`, and all the cosmological parameters ``b``, ...

## Keyword Arguments

- `kwargs...` : Keyword arguments to be passed to the symmetric TPCF

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_LD_multipole`](@ref), 
[`map_ξ_LD_multipole`](@ref), [`print_map_ξ_LD_multipole`](@ref),
[`ξ_LD_Lensing_Doppler`](@ref)
"""
function ξ_LD_Doppler_Lensing(s1, s2, y, cosmo::Cosmology; kwargs...)
    ξ_LD_Lensing_Doppler(s2, s1, y, cosmo; kwargs...)
end

