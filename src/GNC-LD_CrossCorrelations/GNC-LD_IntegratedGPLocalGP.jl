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
     integrand_ξ_GNCxLD_IntegratedGP_LocalGP(
          IP::Point, P1::Point, P2::Point,
          y, cosmo::Cosmology) :: Float64

Return the integrand of the LocalGP-IntegratedGP cross-correlation function 
``\\xi^{\\phi\\int\\phi} (s_1, s_2, \\cos{\\theta})``, i.e. the function 
``f(s_1, s_2, y, \\chi_1, \\chi_2)`` defined as follows:  

```math
f(s_1, s_2, y, \\chi_1, \\chi_2) = 
     \\frac{9 \\mathcal{H}_0^4 \\Omega_{M0}^2 D(s_1) (\\mathcal{R}(s_1) +1)}{2 a(s_1)} 
     \\frac{D(\\chi_2) \\Delta\\chi_2^4}{ a(\\chi_2)}
     \\left(
          \\mathcal{H}(\\chi_2)( f(\\chi_2) - 1) \\mathcal{R}(s_2) - \\frac{1}{s_2}
     \\right) \\tilde{I}^4_0(\\Delta\\chi_2)
```
where ``\\mathcal{H} = a H``, 
``\\Delta\\chi_2 = \\sqrt{s_1^2 + \\chi_2^2 - 2 s_1 \\chi_2 \\cos{\\theta}}``, 
``y = \\cos{\\theta} = \\hat{\\mathbf{s}}_1 \\cdot \\hat{\\mathbf{s}}_2``).

## Inputs

- `IP::Point`: `Point` inside the integration limits, placed 
  at comoving distance `χ1`.

- `P1::Point` and `P2::Point`: extreme `Point` of the integration, placed 
  at comoving distance `s2` and `s1` respectively.

- `y`: the cosine of the angle between the two points `P1` and `P2`

- `cosmo::Cosmology`: cosmology to be used in this computation


See also: [`ξ_GNCxLD_IntegratedGP_LocalGP`](@ref), [`int_on_mu_LocalGP_IntegratedGP`](@ref)
[`integral_on_mu`](@ref), [`ξ_GNC_multipole`](@ref)
"""
function integrand_ξ_GNCxLD_IntegratedGP_LocalGP(
     IP::Point, P1::Point, P2::Point,
     y, cosmo::Cosmology)

     s1, ℛ_s1 = P1.comdist, P1.ℛ_GNC
     s2, D_s2, a_s2, ℜ_s2 = P2.comdist, P2.D, P2.a, P2.ℛ_LD
     χ1, D1, a1, f1, ℋ1 = IP.comdist, IP.D, IP.a, IP.f, IP.ℋ
     s_b_s1 = cosmo.params.s_b
     Ω_M0 = cosmo.params.Ω_M0

     Δχ1_square = s2^2 + χ1^2 - 2 * s2 * χ1 * y
     Δχ1 = Δχ1_square > 0 ? √(Δχ1_square) : 0

     factor = 9 / 4 * D_s2 * Δχ1^4 * ℋ0^4 * Ω_M0^2 * D1 * (1 + ℜ_s2) / (s1 * a1 * a_s2)
     parenth = (s1 * ℋ1 * ℛ_s1 * (f1 - 1) - 5 * s_b_s1 + 2) 

     I04_tilde = cosmo.tools.I04_tilde(Δχ1)

     return factor * parenth * I04_tilde
end


function integrand_ξ_GNCxLD_IntegratedGP_LocalGP(
     χ1::Float64, s1::Float64, s2::Float64,
     y, cosmo::Cosmology;
     kwargs...)

     P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
     IP = Point(χ1, cosmo)
     return integrand_ξ_GNCxLD_IntegratedGP_LocalGP(IP, P1, P2, y, cosmo; kwargs...)
end


"""
     ξ_GNCxLD_IntegratedGP_LocalGP(s1, s2, y, cosmo::Cosmology;
          en::Float64 = 1e6, N_χs::Integer = 100):: Float64

Return the LocalGP-IntegratedGP cross-correlation function 
``\\xi^{v_{\\parallel}\\int \\phi} (s_1, s_2, \\cos{\\theta})`` concerning the perturbed
luminosity distance, defined as follows:
    
```math
\\xi^{v_{\\parallel}\\int \\phi} (s_1, s_2, \\cos{\\theta}) = 
     \\frac{9 \\mathcal{H}_0^4 \\Omega_{M0}^2 D(s_1) (\\mathcal{R}(s_1) +1)}{2 a(s_1)} 
     \\int_0^{s_2} \\mathrm{d}\\chi_2 \\frac{D(\\chi_2) \\Delta\\chi_2^4}{ a(\\chi_2)}
     \\left(
          \\mathcal{H}(\\chi_2)( f(\\chi_2) - 1) \\mathcal{R}(s_2) - \\frac{1}{s_2}
     \\right) \\tilde{I}^4_0(\\Delta\\chi_2)
```
where ``\\mathcal{H} = a H``, 
``\\Delta\\chi_2 = \\sqrt{s_1^2 + \\chi_2^2 - 2 s_1 \\chi_2 \\cos{\\theta}}``, 
``y = \\cos{\\theta} = \\hat{\\mathbf{s}}_1 \\cdot \\hat{\\mathbf{s}}_2``).

The computation is made applying [`trapz`](@ref) (see the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package) to
the integrand function `integrand_ξ_GNCxLD_IntegratedGP_LocalGP`.


## Inputs

- `s2` and `s1`: comovign distances where the function must be evaluated

- `y`: the cosine of the angle between the two points `P1` and `P2`

- `cosmo::Cosmology`: cosmology to be used in this computation


## Optional arguments 

- `en::Float64 = 1e6`: just a float number used in order to deal better 
  with small numbers;

- `N_χs::Integer = 100`: number of points to be used for sampling the integral
  along the ranges `(0, s2)` (for `χ1`) and `(0, s2)` (for `χ1`); it has been checked that
  with `N_χs ≥ 50` the result is stable.


See also: [`integrand_ξ_GNCxLD_IntegratedGP_LocalGP`](@ref), [`int_on_mu_LocalGP_IntegratedGP`](@ref)
[`integral_on_mu`](@ref), [`ξ_GNC_multipole`](@ref)
"""
function ξ_GNCxLD_IntegratedGP_LocalGP(s1, s2, y, cosmo::Cosmology;
     en::Float64 = 1e6, N_χs::Integer = 100)

     #=
     f(χ1) = en * integrand_ξ_GNCxLD_IntegratedGP_LocalGP(χ1, s2, s1, y, cosmo)

     return quadgk(f, 1e-6, s1; rtol=1e-3)[1] / en
     =#

     adim_χs = range(1e-6, 1, N_χs)
     χ1s = adim_χs .* s1

     P1, P2 = GaPSE.Point(s1, cosmo), GaPSE.Point(s2, cosmo)
     IPs = [GaPSE.Point(x, cosmo) for x in χ1s]

     int_ξs = [
          en * GaPSE.integrand_ξ_GNCxLD_IntegratedGP_LocalGP(IP, P1, P2, y, cosmo)
          for IP in IPs
     ]

     res = trapz(χ1s, int_ξs)
     #println("res = $res")
     return res / en
end





