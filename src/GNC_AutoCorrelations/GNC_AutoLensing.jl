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



function integrand_ξ_GNC_Lensing(
    IP1::Point, IP2::Point,
    P1::Point, P2::Point,
    y, cosmo::Cosmology; Δχ_min::Float64=1e-1, 
    b1=nothing, b2=nothing, s_b1=nothing, s_b2=nothing, 𝑓_evo1=nothing, 𝑓_evo2=nothing,
    s_lim=nothing, obs::Union{Bool,Symbol}=:noobsvel)

    
    s1 = P1.comdist
    s2 = P2.comdist
    χ1, D1, a1 = IP1.comdist, IP1.D, IP1.a
    χ2, D2, a2 = IP2.comdist, IP2.D, IP2.a

    Ω_M0 = cosmo.params.Ω_M0
    s_b_s1 = isnothing(s_b1) ? cosmo.params.s_b1 : s_b1
    s_b_s2 = isnothing(s_b2) ? cosmo.params.s_b2 : s_b2

    Δχ_square = χ1^2 + χ2^2 - 2 * χ1 * χ2 * y
    Δχ = Δχ_square > 0 ? √(Δχ_square) : 0

    denomin = s1 * s2 * a1 * a2
    factor = ℋ0^4 * Ω_M0^2 * D1 * (s1 - χ1) * D2 * (s2 - χ2) * (5 * s_b_s1 - 2) * (5 * s_b_s2 - 2)

    first_res = if Δχ > Δχ_min
        χ1χ2 = χ1 * χ2

        new_J00 = -3 / 4 * χ1χ2^2 / Δχ^4 * (y^2 - 1) * (8 * y * (χ1^2 + χ2^2) - χ1χ2 * (9 * y^2 + 7))
        new_J02 = -3 / 2 * χ1χ2^2 / Δχ^4 * (y^2 - 1) * (4 * y * (χ1^2 + χ2^2) - χ1χ2 * (3 * y^2 + 5))
        new_J31 = 9 * y * Δχ^2
        new_J22 = 9 / 4 * χ1χ2 / Δχ^4 * (
            2 * (χ1^4 + χ2^4) * (7 * y^2 - 3)
            - 16 * y * χ1χ2 * (y^2 + 1) * (χ1^2 + χ2^2)
            + χ1χ2^2 * (11y^4 + 14y^2 + 23)
        )

        I00 = cosmo.tools.I00(Δχ)
        I20 = cosmo.tools.I20(Δχ)
        I13 = cosmo.tools.I13(Δχ)
        I22 = cosmo.tools.I22(Δχ)

        (
            new_J00 * I00 + new_J02 * I20 +
            new_J31 * I13 + new_J22 * I22
        )

    else

        #3 / 5 * (5 * cosmo.tools.σ_2 + 6 * cosmo.tools.σ_0 * χ2^2)
        3 * cosmo.tools.σ_2 + 6 / 5 * χ1^2 * cosmo.tools.σ_0
    end

    return factor / denomin * first_res
end

function integrand_ξ_GNC_Lensing(
    χ1::Float64, χ2::Float64,
    s1::Float64, s2::Float64,
    y, cosmo::Cosmology;
    kwargs...)

    P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
    IP1, IP2 = Point(χ1, cosmo), Point(χ2, cosmo)
    return integrand_ξ_GNC_Lensing(IP1, IP2, P1, P2, y, cosmo; kwargs...)
end


"""
    integrand_ξ_GNC_Lensing(
        IP1::Point, IP2::Point,
        P1::Point, P2::Point,
        y, cosmo::Cosmology;
        Δχ_min::Float64=1e-1, b1=nothing, b2=nothing, 
        s_b1=nothing, s_b2=nothing, 𝑓_evo1=nothing, 𝑓_evo2=nothing,
        s_lim=nothing, obs::Union{Bool,Symbol}=:noobsvel
        ) ::Float64

    integrand_ξ_GNC_Lensing(
        χ1::Float64, χ2::Float64,
        s1::Float64, s2::Float64,
        y, cosmo::Cosmology;
        kwargs... )::Float64

Return the integrand of the Two-Point Correlation Function (TPCF) of the 
Lensing auto-correlation effect arising from the Galaxy Number Counts (GNC).

In the first method, you should pass the two extreme `Point`s (`P1` and `P2`) and the two 
intermediate integrand `Point`s (`IP1` and `IP2`) where to 
evaluate the function. In the second method (that internally recalls the first),
you must provide the four corresponding comoving distances `s1`, `s2`, `χ1`, `χ2`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of this integrand is the following:

```math
\\begin{equation}
    f^{\\kappa\\kappa} (\\chi_1, \\chi_2, s_1, s_2, y) = 
    J^{\\kappa\\kappa}_{\\alpha}
    \\left[
        J^{\\kappa\\kappa}_{00} I_0^0(\\Delta\\chi) + 
        J^{\\kappa\\kappa}_{02} I_2^0(\\Delta\\chi) +
        J^{\\kappa\\kappa}_{31} I_1^3(\\Delta\\chi) +
        J^{\\kappa\\kappa}_{22} I_2^2(\\Delta\\chi)
    \\right]  \\, , 
\\end{equation}
```

with

```math
\\begin{split}
    J^{\\kappa\\kappa}_{\\alpha} & = 
    \\frac{
        \\mathcal{H}_0^4 \\Omega_{\\mathrm{M}0}^2 D(\\chi_1) D(\\chi_2) 
    }{
        s_1 s_2 a(\\chi_1) a(\\chi_2)}
    (\\chi_1 - s_1)(\\chi_2 - s_2)
    (5 s_{\\mathrm{b}, 1} - 2)(5 s_{\\mathrm{b}, 2} - 2) 
    \\, , \\\\
    %%%%&%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{00} & = 
    -\\frac{ 3 \\chi_1^2 \\chi_2^2}{4 \\Delta\\chi^4} (y^2 - 1)
    \\left[
        8 y (\\chi_1^2 + \\chi_2^2) - 9\\chi_1\\chi_2y^2 - 
        7\\chi_1\\chi_2
    \\right] 
    \\, , \\\\
    %%%%&%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{02} & = 
    -\\frac{ 3 \\chi_1^2 \\chi_2^2}{2 \\Delta\\chi^4}(y^2 - 1)
    \\left[
        4 y (\\chi_1^2 + \\chi_2^2) - 3 \\chi_1 \\chi_2 y^2 -
        5 \\chi_1 \\chi_2
    \\right] 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{31} & = 9 y \\Delta\\chi^2 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{22} & = 
    \\frac{9 \\chi_1 \\chi_2}{4 \\Delta\\chi^4}
    \\left[
        2(\\chi_1^4 + \\chi_2^4)(7 y^2 - 3) - 
        16 y \\chi_1 \\chi_2 (\\chi_1^2 + \\chi_2^2)(y^2 + 1) + 
        \\right.\\\\
        &\\left.\\qquad\\qquad\\qquad
        \\chi_1^2 \\chi_2^2 (11y^4 + 14y^2 + 23) 
    \\right] 
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

This function is used inside `ξ_GNC_Lensing` with trapz() from the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package.


## Inputs

-  `IP1::Point`, `IP2::Point`, `P1::Point`, `P2::Point` or `χ1`, `χ2`, `s1`, `s2`: `Point`/comoving 
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

- `Δχ_min::Float64 = 1e-4` : when 
  ``\\Delta\\chi = \\sqrt{\\chi_1^2 + \\chi_2^2 - 2 \\, \\chi_1 \\chi_2 y} \\to 0^{+}``,
  some ``I_\\ell^n`` term diverges, but the overall parenthesis has a known limit:

  ```math
      \\lim_{\\Delta\\chi \\to 0^{+}} \\left(
          J_{00}^{\\kappa\\kappa} \\, I^0_0(\\Delta\\chi) + 
          J_{02}^{\\kappa\\kappa} \\, I^0_2(\\Delta\\chi) + 
          J_{31}^{\\kappa\\kappa} \\, I^3_1(\\Delta\\chi) + 
          J_{22}^{\\kappa\\kappa} \\, I^2_2(\\Delta\\chi)
      \\right) = 
          3 \\, \\sigma_2 + \\frac{6}{5} \\, \\chi_2^2 \\, \\sigma_0
  ```

  So, when it happens that ``\\Delta\\chi < \\Delta\\chi_\\mathrm{min}``, the function considers this limit
  as the result of the parenthesis instead of calculating it in the normal way; it prevents
  computational divergences.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_GNC_multipole`](@ref), 
[`map_ξ_GNC_multipole`](@ref), [`print_map_ξ_GNC_multipole`](@ref),
[`ξ_GNC_Lensing`](@ref)
"""
integrand_ξ_GNC_Lensing



##########################################################################################92


function lr(a,b,n,i)
    return a + (i-1.0)/(n-1.0)*(b-a)
end

function ξ_GNC_Lensing(P1::Point, P2::Point, y, cosmo::Cosmology;
    en::Float64=1e6, N_χs_2::Int=100, suit_sampling::Bool=true, 
    backend=CPU(), kwargs...)

    if backend==false
        χ1s = P1.comdist .* range(1e-6, 1, length=N_χs_2)
        #χ2s = P2.comdist .* range(1e-5, 1, length = N_χs_2 + 7)
        χ2s = P2.comdist .* range(1e-6, 1, length=N_χs_2)

        IP1s = [GaPSE.Point(x, cosmo) for x in χ1s]
        IP2s = [GaPSE.Point(x, cosmo) for x in χ2s]

        int_ξ_Lensings = [
          en * GaPSE.integrand_ξ_GNC_Lensing(IP1, IP2, P1, P2, y, cosmo; kwargs...)
          for IP1 in IP1s, IP2 in IP2s
        ]

        res = trapz((χ1s, χ2s), int_ξ_Lensings)
        return res / en

    else
        #= 
        # with this everything works as expected
        χ1s = P1.comdist .* range(1e-6, 1, length=N_χs_2)
        #χ2s = P2.comdist .* range(1e-5, 1, length = N_χs_2 + 7)
        χ2s = P2.comdist .* range(1e-6, 1, length=N_χs_2)

        IP1s = [GaPSE.Point(x, cosmo) for x in χ1s]
        IP2s = [GaPSE.Point(x, cosmo) for x in χ2s]

        int_ξ_Lensings = [
            en * GaPSE.integrand_ξ_GNC_Lensing(IP1, IP2, P1, P2, y, cosmo; kwargs...)
            for IP1 in IP1s, IP2 in IP2s
        ]

        res = trapz((χ1s, χ2s), int_ξ_Lensings)
        return res / en
        =#
        
        χ1s = P1.comdist .* range(1e-6, 1, length=N_χs_2)
        #χ2s = P2.comdist .* range(1e-5, 1, length = N_χs_2 + 7)
        χ2s = P2.comdist .* range(1e-6, 1, length=N_χs_2)

        #IP1s = [GaPSE.Point(x, cosmo) for x in χ1s]
        #IP2s = [GaPSE.Point(x, cosmo) for x in χ2s]

        int_ξ_Lensings = KernelAbstractions.zeros(backend, Float64, N_χs_2, N_χs_2)
        #tmp = KernelAbstractions.zeros(length(IP1s))

        #i = 1
        #for j in 1:length(IP2s)
        #  int_ξ_Lensing[i,j] = @oneapi items=30 en * GaPSE.integrand_ξ_GNC_Lensing_oneapi(IP1s, IP2s[j], P1, P2, y, cosmo; kwargs...)
        #end
        #  global i+=1

        #function vadd(tmp, IP1s, IP2s;kwargs...)
        #   i = get_global_id()
        #   #@inbounds c[i] = a[i] + b[i]
        #   tmp[i] = en * GaPSE.integrand_ξ_GNC_Lensing(IP1s[i], IP2s, P1, P2, y, cosmo;kwargs...)
        #   return 
        #end

        #a = oneArray(rand(10));

        #b = oneArray(rand(10));

        #c = similar(a);
        #for j in 1:length(IP2s)
        # @oneapi items=10 vadd(tmp, IP1s, IP2s[j])
        # [int_ξ_Lensings[i, j] =  tmp[i] for i in 1:legnth(IP1s)]
        #end
        #@kernel function mykernel!(int_ξ_Lensings, integrand_ξ_GNC_Lensing, IP1s, IP2s, P1, P2, y, cosmo, kwargs...)
        @kernel function mykernel!(int_ξ_Lensings, integrand_ξ_GNC_Lensing, P1, P2, y, cosmo, kwargs...) 
          i, j = @index(Global, NTuple)
          IP1 = GaPSE.Point(P1.comdist * lr(1e-6, 1, N_χs_2, i), cosmo)
          IP2 = GaPSE.Point(P2.comdist * lr(1e-6, 1, N_χs_2, j), cosmo)
          #IP1 = GaPSE.Point(χ1s[i], cosmo)
          #IP2 = GaPSE.Point(χ2s[j], cosmo) 
          int_ξ_Lensings[i,j] = en * integrand_ξ_GNC_Lensing(IP1, IP2, P1, P2, y, cosmo; kwargs...)
        end
        #Array(a) .+ Array(b) == Array(c)

        kernel! = mykernel!(backend) 
        #kernel!(int_ξ_Lensings, GaPSE.integrand_ξ_GNC_Lensing, IP1s, IP2s, P1, P2, y, cosmo, kwargs...; ndrange=size(int_ξ_Lensings))
        kernel!(int_ξ_Lensings, GaPSE.integrand_ξ_GNC_Lensing, P1, P2, y, cosmo, kwargs...; ndrange=size(int_ξ_Lensings))
        KernelAbstractions.synchronize(backend)

        res = trapz((χ1s, χ2s), reshape(int_ξ_Lensings,N_χs_2, N_χs_2))
        #println("res = $res")
        return res / en
        
    end
end


function ξ_GNC_Lensing(s1, s2, y, cosmo::Cosmology; kwargs...)
    P1, P2 = Point(s1, cosmo), Point(s2, cosmo)
    return ξ_GNC_Lensing(P1, P2, y, cosmo; kwargs...)
end


"""
    ξ_GNC_Lensing(P1::Point, P2::Point, y, cosmo::Cosmology;
        en::Float64 = 1e6, Δχ_min::Float64 = 1e-1,
        N_χs_2::Int = 100,
        s_b1=nothing, s_b2=nothing, 𝑓_evo1=nothing, 𝑓_evo2=nothing,
        s_lim=nothing, obs::Union{Bool,Symbol}=:noobsvel,
        suit_sampling::Bool=true ) ::Float64

    ξ_GNC_Lensing(s1, s2, y, cosmo::Cosmology; 
        kwargs...) ::Float64

Return the Two-Point Correlation Function (TPCF) of the Lensing auto-correlation effect
arising from the Galaxy Number Counts (GNC).

In the first method, you should pass the two `Point` (`P1` and `P2`) where to 
evaluate the function, while in the second method (that internally recalls the first) 
you must provide the two corresponding comoving distances `s1` and `s2`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

The analytical expression of this TPCF is the following:

```math
\\begin{split}
    \\xi^{\\kappa\\kappa} (s_1, s_2, y) = 
    \\int_0^{s_1} \\mathrm{d}\\chi_1 \\int_0^{s_2} \\mathrm{d}\\chi_2\\;  
    J^{\\kappa\\kappa}_{\\alpha}
    &\\left[
        J^{\\kappa\\kappa}_{00} I_0^0(\\Delta\\chi) + 
        J^{\\kappa\\kappa}_{02} I_2^0(\\Delta\\chi) +
        \\right.\\\\
        &\\left.
        J^{\\kappa\\kappa}_{31} I_1^3(\\Delta\\chi) +
        J^{\\kappa\\kappa}_{22} I_2^2(\\Delta\\chi)
    \\right]  \\, , 
\\end{split}
```

with

```math
\\begin{split}
    J^{\\kappa\\kappa}_{\\alpha} & = 
    \\frac{
        \\mathcal{H}_0^4 \\Omega_{\\mathrm{M}0}^2 D(\\chi_1) D(\\chi_2) 
    }{
        s_1 s_2 a(\\chi_1) a(\\chi_2)}
    (\\chi_1 - s_1)(\\chi_2 - s_2)
    (5 s_{\\mathrm{b}, 1} - 2)(5 s_{\\mathrm{b}, 2} - 2) 
    \\, , \\\\
    %%%%&%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{00} & = 
    -\\frac{ 3 \\chi_1^2 \\chi_2^2}{4 \\Delta\\chi^4} (y^2 - 1)
    \\left[
        8 y (\\chi_1^2 + \\chi_2^2) - 9\\chi_1\\chi_2y^2 - 
        7\\chi_1\\chi_2
    \\right] 
    \\, , \\\\
    %%%%&%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{02} & = 
    -\\frac{ 3 \\chi_1^2 \\chi_2^2}{2 \\Delta\\chi^4}(y^2 - 1)
    \\left[
        4 y (\\chi_1^2 + \\chi_2^2) - 3 \\chi_1 \\chi_2 y^2 -
        5 \\chi_1 \\chi_2
    \\right] 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{31} & = 9 y \\Delta\\chi^2 
    \\, , \\\\
    %%%%%%%%%%%%%%%%%%
    J^{\\kappa\\kappa}_{22} & = 
    \\frac{9 \\chi_1 \\chi_2}{4 \\Delta\\chi^4}
    \\left[
        2(\\chi_1^4 + \\chi_2^4)(7 y^2 - 3) - 
        16 y \\chi_1 \\chi_2 (\\chi_1^2 + \\chi_2^2)(y^2 + 1) + 
        \\right.\\\\
        &\\left.\\qquad\\qquad\\qquad
        \\chi_1^2 \\chi_2^2 (11y^4 + 14y^2 + 23) 
    \\right] 
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

- ``\\mathcal{H}_0``, ``f_0`` and so on are evaluated at the observer position (i.e. at present day);

- ``\\Delta\\chi_1 := \\sqrt{\\chi_1^2 + s_2^2-2\\,\\chi_1\\,s_2\\,y}`` and 
  ``\\Delta\\chi_2 := \\sqrt{s_1^2 + \\chi_2^2-2\\,s_1\\,\\chi_2\\,y}``;

- ``s=\\sqrt{s_1^2 + s_2^2 - 2 \\, s_1 \\, s_2 \\, y}`` and 
  ``\\Delta\\chi := \\sqrt{\\chi_1^2 + \\chi_2^2-2\\,\\chi_1\\,\\chi_2\\,y}``.

In this TPCF there are no observer terms. The `obs` keyword is inserted only for compatibility with 
the other GNC TPCFs.

This function is computed integrating `integrand_ξ_GNC_Lensing` with trapz() from the 
[Trapz](https://github.com/francescoalemanno/Trapz.jl) Julia package.


## Inputs

- `P1::Point` and `P2::Point`, or `s1` and `s2`: `Point`/comoving distances where the 
  TPCF has to be calculated; they contain all the 
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

- `en::Float64 = 1e6`: just a float number used in order to deal better 
  with small numbers;

- `Δχ_min::Float64 = 1e-4` : when 
  ``\\Delta\\chi = \\sqrt{\\chi_1^2 + \\chi_2^2 - 2 \\, \\chi_1 \\chi_2 y} \\to 0^{+}``,
  some ``I_\\ell^n`` term diverges, but the overall parenthesis has a known limit:

  ```math
      \\lim_{\\Delta\\chi \\to 0^{+}} \\left(
          J_{00}^{\\kappa\\kappa} \\, I^0_0(\\Delta\\chi) + 
          J_{02}^{\\kappa\\kappa} \\, I^0_2(\\Delta\\chi) + 
          J_{31}^{\\kappa\\kappa} \\, I^3_1(\\Delta\\chi) + 
          J_{22}^{\\kappa\\kappa} \\, I^2_2(\\Delta\\chi)
      \\right) = 
          3 \\, \\sigma_2 + \\frac{6}{5} \\, \\chi_2^2 \\, \\sigma_0
  ```

  So, when it happens that ``\\Delta\\chi < \\Delta\\chi_\\mathrm{min}``, the function considers this limit
  as the result of the parenthesis instead of calculating it in the normal way; it prevents
  computational divergences.

- `N_χs_2::Int = 100`: number of points to be used for sampling the integral
  along the ranges `(0, s1)` (for `χ1`) and `(0, s2)` (for `χ2`); it has been checked that
  with `N_χs_2 ≥ 50` the result is stable.

- `suit_sampling::Bool = true` : this bool keyword can be found in all the TPCFs which have at least one `χ` integral;
  it is conceived to enable a sampling of the `χ` integral(s) suited for the given TPCF; however, it actually have an
  effect only in the TPCFs that have such a sampling implemented in the code.
  Currently, only `ξ_GNC_Newtonian_Lensing` (and its simmetryc TPCF) has it.

See also: [`Point`](@ref), [`Cosmology`](@ref), [`ξ_GNC_multipole`](@ref), 
[`map_ξ_GNC_multipole`](@ref), [`print_map_ξ_GNC_multipole`](@ref),
[`integrand_ξ_GNC_Lensing`](@ref)
"""
ξ_GNC_Lensing
