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


@doc raw"""
     ξ_GNC_Doppler_LocalGP(P1::Point, P2::Point, y, cosmo::Cosmology) :: Float64

Return the Doppler-LocalGP cross-correlation function concerning the perturbed
luminosity distance, defined as follows:

```math
\\xi^{v_{\\parallel}\\phi} (s_1, s_2, \\cos{\\theta}) = 
     \\frac{3}{2 a(s_2)} \\mathcal{H}(s_1) f(s_1) D(s_1)
     \\mathcal{R}(s_1) \\mathcal{H}_0^2 \\Omega_{M0} D(s_2)
     (1 + \\mathcal{R}(s_2)) (s_2\\cos{\\theta} - s_1) s^2 I^3_1(s)
```
where ``\\mathcal{H} = a H``,
``y = \\cos{\\theta} = \\hat{\\mathbf{s}}_1 \\cdot \\hat{\\mathbf{s}}_2`` and :

```math
I^n_l(s) = \\int_0^\\infty \\frac{\\mathrm{d}q}{2\\pi^2} q^2 \\, P(q) \\, \\frac{j_l(qs)}{(q s)^n}
```

## Inputs

- `P1::Point` and `P2::Point`: `Point` where the CF has to be calculated; they contain all the 
  data of interest needed for this calculus (comoving distance, growth factor and so on)
     
- `y`: the cosine of the angle between the two points `P1` and `P2`

- `cosmo::Cosmology`: cosmology to be used in this computation


See also: [`Point`](@ref), [`Cosmology`](@ref)
"""
function ξ_GNC_Doppler_LocalGP(P1::Point, P2::Point, y, cosmo::Cosmology; obs::Bool = true)
     s1, D1, f1, ℋ1, ℛ1 = P1.comdist, P1.D, P1.f, P1.ℋ, P1.ℛ_GNC
     s2, D2, f2, a2, ℋ2, ℛ2 = P2.comdist, P2.D, P2.f, P2.a, P2.ℋ, P2.ℛ_GNC
     𝑓_evo2 = cosmo.params.𝑓_evo
     s_b1, s_b2 = cosmo.params.s_b, cosmo.params.s_b
     Ω_M0 = cosmo.params.Ω_M0

     Δs = s(s1, s2, y)

     common = Δs^2 * f1 * ℋ1 * ℛ1 * (s1 - y * s2) / a2
     parenth = 2 * f2 * a2 * ℋ2^2 * (𝑓_evo2 - 3) + 3 * ℋ0^2 * Ω_M0 * (f2 + ℛ2 + 5 * s_b2 - 2)

     I00 = cosmo.tools.I00(Δs)
     I20 = cosmo.tools.I20(Δs)
     I40 = cosmo.tools.I40(Δs)
     I02 = cosmo.tools.I02(Δs)

     if obs == false
          return D1 * D2 * common * parenth * (
               -1 / 90 * I00 - 1 / 63 * I20 
               - 1 / 210 * I40 - 1 / 6 * I02
               )
     else
          #### New observer terms #########

          P0 = Point(0.0, cosmo)
          ℋ0, f0 = P0.ℋ, P0.f

          I13_s1 = cosmo.tools.I13(s1)
          I13_s2 = cosmo.tools.I13(s2)

          J31_a = ℋ0 * s1^3 * f1 * ℋ1 * ℛ1 * (ℋ0 * s2 * ℛ2 * (3 * Ω_M0 - 2 * f0) + 2 * f0 * (2 - 5 * s_b2)) / (2 * s2)
          J31_b = - y * f0 * ℋ0 * s2^3 * (ℛ1 - 5 * s_b1 + 2) / (2 * a2) * parenth 

          obs_terms = D1 * J31_a * I13_s1 + D2 * J31_b * I13_s2 

          #################################

          return D1 * D2 * common * parenth * (
               -1 / 90 * I00 - 1 / 63 * I20 
               - 1 / 210 * I40 - 1 / 6 * I02
               ) + obs_terms
     end
end


function ξ_GNC_Doppler_LocalGP(s1, s2, y, cosmo::Cosmology; obs::Bool = true)
     P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
     return ξ_GNC_Doppler_LocalGP(P1, P2, y, cosmo; obs = obs)
end




##########################################################################################92

##########################################################################################92

##########################################################################################92



function ξ_GNC_LocalGP_Doppler(s1, s2, y, cosmo::Cosmology; kwargs...)
     ξ_GNC_Doppler_LocalGP(s2, s1, y, cosmo; kwargs...)
end

