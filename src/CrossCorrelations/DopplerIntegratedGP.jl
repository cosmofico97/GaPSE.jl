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
     integrand_ξ_dopplerintegratedgp(
          IP::Point, P1::Point, P2::Point,
          y, cosmo::Cosmology) :: Float64

Return the integrand of the lensing auto-correlation function 
``\xi^{v_{\parallel}\kappa} (s_1, s_2, \cos{\theta})``, i.e. the function 
``f(s_1, s_2, y, \chi_1, \chi_2)`` defined as follows:  

```math
f(s_1, s_2, y, \chi_1, \chi_2) = 
     3 \mathcal{H}(s_1) f(s_1) D(s_1) \mathcal{H_0}^2 \Omega_{M0} 
     \mathcal{R}(s_1) J_{31} I^3_1(\chi)
```
where ``\mathcal{H} = a H``, 
``\chi = \sqrt{s_1^2 + \chi_2^2 - 2 s_1 \chi_2 \cos{\theta}}``, 
``y = \cos{\theta} = \hat{\mathbf{s}}_1 \dot \hat{\mathbf{s}}_2``) 
and:

```math
J_{31} = 
     \frac{D(\chi_2) (s_1 - \chi_2 \cos{\theta})}{a(\chi_2)} \chi^2 
     \left(
          - \frac{1}{s_2} + \mathcal{R}(s_2) \mathcal{H}(\chi_2) (f(\chi_2) - 1)
     \right)
```

## Inputs

- `IP::Point`: `Point` inside the integration limits, placed 
  at comoving distance `χ1`.

- `P1::Point` and `P2::Point`: extreme `Point` of the integration, placed 
  at comoving distance `s1` and `s2` respectively.

- `y`: the cosine of the angle between the two points `P1` and `P2`

- `cosmo::Cosmology`: cosmology to be used in this computation


See also: [`ξ_dopplerintegratedgp`](@ref), [`int_on_mu_dopplerintegratedgp`](@ref)
[`integral_on_mu`](@ref), [`ξ_multipole`](@ref)
"""
function integrand_ξ_dopplerintegratedgp(
     IP::Point, P1::Point, P2::Point,
     y, cosmo::Cosmology)

     s1, ℛ_s1 = P1.comdist, P1.ℛ
     s2, ℛ_s2 = P2.comdist, P2.ℛ
     χ2, D2, a2, f2, ℋ2 = IP.comdist, IP.D, IP.a, IP.f, IP.ℋ
     Ω_M0 = cosmo.params.Ω_M0

     Δχ2 = √(s1^2 + χ2^2 - 2 * s1 * χ2 * y)
     #println(" Δχ2 =  $Δχ2")

     common = ℋ0^2 * Ω_M0 * D2 / (s2 * a2)
     factor = Δχ2^2 * (χ2 * y - s1) * (s2 * (f2 - 1) * ℋ2 * ℛ_s2 + 1)

     I00 = cosmo.tools.I00(Δχ2)
     I20 = cosmo.tools.I20(Δχ2)
     I40 = cosmo.tools.I40(Δχ2)
     I02 = cosmo.tools.I02(Δχ2)

     first = common * factor * (1 / 15 * I00 + 2 / 21 * I20 + 1 / 35 * I40 + I02)

     new_J31 = -3 * χ2^3 * y * f0 * ℋ0 * (ℛ_s1 + 1) * (s2 * (f2 - 1) * ℋ2 * ℛ_s2 + 1)
     I13 = cosmo.tools.I13(χ2)

     second = common * new_J31 * I13

     #println("J00 = $new_J00, \t I00(Δχ) = $(I00)")
     #println("J02 = $new_J02, \t I20(Δχ) = $(I20)")
     #println("J31 = $new_J31, \t I13(Δχ) = $(I13)")
     #println("J22 = $new_J22, \t I22(Δχ) = $(I22)")

     return second
end


function integrand_ξ_dopplerintegratedgp(
     χ1::Float64, s1::Float64, s2::Float64,
     y, cosmo::Cosmology;
     kwargs...)

     P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
     IP = Point(χ1, cosmo)
     return integrand_ξ_dopplerintegratedgp(IP, P1, P2, y, cosmo; kwargs...)
end


@doc raw"""
     ξ_dopplerintegratedgp(s1, s2, y, cosmo::Cosmology;
          en::Float64 = 1e6, N_χs::Integer = 100):: Float64

Return the lensing auto-correlation function 
``\xi^{v_{\parallel}\int \phi} (s_1, s_2, \cos{\theta})``, defined as follows:
    
```math
\xi^{v_{\parallel}\int \phi} (s_1, s_2, \cos{\theta}) = 
     3 \mathcal{H}(s_1) f(s_1) D(s_1) \mathcal{H_0}^2 \Omega_{M0} \mathcal{R}(s_1) 
     \int_0^{s_2} \mathrm{d}\chi_2 \,  J_{31} \,  I^3_1(\chi)
```

where ``\mathcal{H} = a H``, 
``\chi = \sqrt{s_1^2 + \chi_2^2 - 2 s_1 \chi_2 \cos{\theta}}``, 
``y = \cos{\theta} = \hat{\mathbf{s}}_1 \dot \hat{\mathbf{s}}_2``) 
and:

```math
J_{31} = 
     \frac{D(\chi_2) (s_1 - \chi_2 \cos{\theta})}{a(\chi_2)} \chi^2 
     \left(
          - \frac{1}{s_2} + \mathcal{R}(s_2) \mathcal{H}(\chi_2) (f(\chi_2) - 1)
     \right)
```

The computation is made applying [`trapz`](@ref) (see the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package) to
the integrand function `integrand_ξ_dopplerintegratedgp`.


## Inputs

- `s1` and `s2`: comovign distances where the function must be evaluated

- `y`: the cosine of the angle between the two points `P1` and `P2`

- `cosmo::Cosmology`: cosmology to be used in this computation


## Optional arguments 

- `en::Float64 = 1e6`: just a float number used in order to deal better 
  with small numbers;

- `N_χs::Integer = 100`: number of points to be used for sampling the integral
  along the ranges `(0, s1)` (for `χ1`) and `(0, s1)` (for `χ2`); it has been checked that
  with `N_χs ≥ 50` the result is stable.


See also: [`integrand_ξ_dopplerintegratedgp`](@ref), [`int_on_mu_dopplerintegratedgp`](@ref)
[`integral_on_mu`](@ref), [`ξ_multipole`](@ref)
"""
function ξ_dopplerintegratedgp(s1, s2, y, cosmo::Cosmology;
     en::Float64 = 1e6, N_χs::Integer = 100)

     adim_χs = range(1e-6, 1.0, N_χs)
     χ2s = adim_χs .* s2

     P1, P2 = GaPSE.Point(s1, cosmo), GaPSE.Point(s2, cosmo)
     IPs = [GaPSE.Point(x, cosmo) for x in χ2s]

     int_ξs = [
          en * GaPSE.integrand_ξ_dopplerintegratedgp(IP, P1, P2, y, cosmo)
          for IP in IPs
     ]

     res = trapz(χ2s, int_ξs)
     #println("res = $res")
     return res / en
end



##########################################################################################92



@doc raw"""
     int_on_mu_dopplerintegratedgp(s1, s, μ, cosmo::Cosmology;
          L::Integer = 0, 
          use_windows::Bool = true, 
          en::Float64 = 1e6,
          N_χs::Integer = 100) :: Float64

Return the integrand on ``\mu = \hat{\mathbf{s}}_1 \dot \hat{\mathbf{s}}`` 
of the lensing auto-correlation function, i.e.
the following function ``f(s_1, s, \mu)``:

```math
     f(s_1, s, \mu) = \xi^{\kappa\kappa} (s_1, s_2, \cos{\theta}) 
          \, \mathcal{L}_L(\mu) \,  \phi(s_2) \, F\left(\frac{s}{s_1}, \mu \right)
```
where ``y =  \cos{\theta} = \hat{\mathbf{s}}_1 \dot \hat{\mathbf{s}}_2`` and
``s = \sqrt{s_1^2 + s_2^2 - 2 \, s_1 \, s_2 \, y}``.

In case `use_windows` is set to `false`, the window functions ``\phi`` and ``F``
are removed, i.e is returned the following function ``f^{'}(s_1, s, \mu)``:

```math
     f^{'}(s_1, s, \mu) = \xi^{\kappa\kappa} (s_1, s_2, \cos{\theta}) 
          \, \mathcal{L}_L(\mu) 
```

The function ``\xi^{\kappa\kappa}(s_1, s_2, \cos{\theta})`` is calculated
from `ξ_lensing`; note that these is an internal conversion of coordiate sistems
from `(s1, s, μ)` to `(s1, s2, y)` thorugh the functions `y` and `s2`

## Inputs

- `s1`: the comoving distance where must be evaluated the integral

- `s`: the comoving distance from `s1` where must be evaluated the integral

- `μ`: the cosine between `s1` and `s` where must be evaluated the integral

- `cosmo::Cosmology`: cosmology to be used in this computation


## Optional arguments 

- `L::Integer = 0`: order of the Legendre polynomial to be used

- `en::Float64 = 1e6`: just a float number used in order to deal better 
  with small numbers;

- `use_windows::Bool = false`: tells if the integrand must consider the two
   window function ``\phi`` and ``F``

- `N_χs::Integer = 100`: number of points to be used for sampling the integral
  along the ranges `(0, s1)` (for `χ1`) and `(0, s1)` (for `χ2`); it has been checked that
  with `N_χs ≥ 50` the result is stable.

See also: [`integrand_ξ_dopplerintegratedgp`](@ref), [`ξ_dopplerintegratedgp`](@ref),
[`integral_on_mu`](@ref), [`map_integral_on_mu`](@ref),
[`spline_F`](@ref), [`ϕ`](@ref), [`Cosmology`](@ref), 
[`y`](@ref), [`s2`](@ref)
"""
function int_on_mu_dopplerintegratedgp(s1, s, μ, cosmo::Cosmology;
     L::Integer = 0,
     use_windows::Bool = true,
     en::Float64 = 1e6,
     N_χs::Integer = 100)

     s2_value = s2(s1, s, μ)
     y_value = y(s1, s, μ)
     res = if use_windows == true
          ϕ_s2 = ϕ(s2_value; s_min = cosmo.s_min, s_max = cosmo.s_max)
          (ϕ_s2 > 0.0) || (return 0.0)
          #println("s1 = $s1 \t s2 = $(s2(s1, s, μ)) \t  y=$(y(s1, s, μ))")
          int = ξ_dopplerintegratedgp(s1, s2_value, y_value, cosmo;
               en = en, N_χs = N_χs)
          #println("int = $int")
          int .* (ϕ_s2 * spline_F(s / s1, μ, cosmo.windowF) * Pl(μ, L))
     else
          #println("s1 = $s1 \t s2 = $(s2(s1, s, μ)) \t  y=$(y(s1, s, μ))")
          int = ξ_dopplerintegratedgp(s1, s2_value, y_value, cosmo;
               en = en, N_χs = N_χs)
          #println("int = $int")
          #println( "Pl(μ, L) = $(Pl(μ, L))")
          int .* Pl(μ, L)
     end

     #println("res = $res")
     return res
end

