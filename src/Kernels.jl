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
    lr(a, b, n, i) ::Float64

Return the `i`th-number in the linear range `[a,b]` subdivided in `n` pieces:

```math
\\mathrm{lr}(a,b,n,i) = a + \\frac{i-1}{n-1}\\,(b-a)
```

It means that, with `n>2`, `1 ≤ i ≤ n` and:

- `i=1` => `lr(a, b, n, 1) = a`;
- `1<i<n` => `a < lr(a, b, n, i) < b`;
- `i=n` => `lr(a, b, n, 1) = b`. 

It's used inside [`kernel_1d_P1!`](@ref), [`kernel_1d_P2!`](@ref) and [`kernel_2d!`](@ref)
"""
function lr(a, b, n, i)
    @assert (a<b) && (1≤i≤n) "Not valid inputs: a, b = $a, $b \t i, n = $i, $n"
    return a + (i - 1.0) / (n - 1.0) * (b - a)
end



##########################################################################################92



"""
    @kernel function kernel_1d_P1!(results_vector, integrand, P1, P2, y, cosmo, N_χs, kwargs...) ::Nothing

One-dimensional (`1d`) kernel used for the parallelisation of the 1d integrals over χ1 (`P1`).
The inputs are:

- `results_vector` : 1d vector where the results will be stored;
- `integrand` : function to be integrated (e.g. `GaPSE.integrand_Lensing_Doppler`);
- `P1` and `P2`, `y` and `cosmo` : `Point`s, angle cosine and `Cosmology` to be given to the `integrand`;
- `N_χs` : number of points to be used sampling the interval `1e-6 ≤ χ1 ≤ P1.comdist`;
- `kwargs...` : other kwargs to be passed to `integrand`.

## Example

```julia
s1, s2, N_χs = cosmo.s_eff, 1.3*cosmo.s_eff, 50

χ1s = s1 .* range(1e-6, 1, length=N_χs)
P1, P2 = GaPSE.Point(s1, cosmo), GaPSE.Point(s2, cosmo)

int_ξs = KernelAbstractions.zeros(backend, Float64, N_χs)
kernel! = kernel_1d_P1!(backend)
kernel!(int_ξs, GaPSE.integrand_ξ_GNC_Lensing_Doppler, P1, P2, y, cosmo, N_χs, kwargs...; ndrange=size(int_ξs))
KernelAbstractions.synchronize(backend)

res = trapz(χ1s, int_ξs)
```

See also: [`kernel_1d_P2!`](@ref), [`kernel_2d!`](@ref)
"""
@kernel function kernel_1d_P1!(results_vector, integrand, P1, P2, y, cosmo, N_χs, kwargs...)
    i = @index(Global, Linear)
    IP = GaPSE.Point(P1.comdist * lr(1e-6, 1, N_χs, i), cosmo)
    results_vector[i] = integrand(IP, P1, P2, y, cosmo; kwargs...)
end



"""
    @kernel function kernel_1d_P2!(results_vector, integrand, P1, P2, y, cosmo, N_χs, kwargs...) ::Nothing

One-dimensional (`1d`) kernel used for the parallelisation of the 1d integrals over χ2 (`P2`).
The inputs are:

- `results_vector` : 1d vector where the results will be stored;
- `integrand` : function to be integrated (e.g. `GaPSE.integrand_ξ_GNC_Newtonian_IntegratedGP`);
- `P1` and `P2`, `y` and `cosmo` : `Point`s, angle cosine and `Cosmology` to be given to the `integrand`;
- `N_χs` : number of points to be used sampling the interval `1e-6 ≤ χ2 ≤ P2.comdist`;
- `kwargs...` : other kwargs to be passed to `integrand`.

## Example

```julia
s1, s2, N_χs = cosmo.s_eff, 1.3*cosmo.s_eff, 50

χ2s = s2 .* range(1e-6, 1, length=N_χs)
P1, P2 = GaPSE.Point(s1, cosmo), GaPSE.Point(s2, cosmo)

int_ξs = KernelAbstractions.zeros(backend, Float64, N_χs)
kernel! = kernel_1d_P2!(backend)
kernel!(int_ξs, GaPSE.integrand_ξ_GNC_Newtonian_IntegratedGP, P1, P2, y, cosmo, N_χs, kwargs...; ndrange=size(int_ξs))
KernelAbstractions.synchronize(backend)

res = trapz(χ2s, int_ξs)
```

See also: [`kernel_1d_P1!`](@ref), [`kernel_2d!`](@ref)
"""
@kernel function kernel_1d_P2!(results_vector, integrand, P1, P2, y, cosmo, N_χs, kwargs...)
    i = @index(Global, Linear)
    IP = GaPSE.Point(P2.comdist * lr(1e-6, 1, N_χs, i), cosmo)
    results_vector[i] = integrand(IP, P1, P2, y, cosmo; kwargs...)
end


##########################################################################################92



"""
    @kernel function kernel_2d!(results_vector, integrand, P1, P2, y, cosmo, N_χs_2, kwargs...) ::Nothing

Two-dimensional (`2d`) kernel used for the parallelisation of the 2d integrals over χ1 and χ2.
The inputs are:

- `results_vector` : 2d vector where the results will be stored;
- `integrand` : function to be integrated (e.g. `GaPSE.integrand_ξ_GNC_AutoLensing`);
- `P1` and `P2`, `y` and `cosmo` : `Point`s, angle cosine and `Cosmology` to be given to the `integrand`;
- `N_χs_2` : number of points to be used sampling each of the intervals `1e-6 ≤ χ1 ≤ P1.comdist` and `1e-6 ≤ χ2 ≤ P2.comdist`;
- `kwargs...` : other kwargs to be passed to `integrand`.

## Example

```julia
s1, s2, N_χs_2 = cosmo.s_eff, 1.3*cosmo.s_eff, 50

χ1s = s1 .* range(1e-6, 1, length=N_χs_2)
χ2s = s2 .* range(1e-6, 1, length=N_χs_2)
P1, P2 = GaPSE.Point(s1, cosmo), GaPSE.Point(s2, cosmo)

int_ξs = KernelAbstractions.zeros(backend, Float64, N_χs_2, N_χs_2)
kernel! = kernel_2d!(backend)
kernel!(int_ξs, GaPSE.integrand_ξ_GNC_Lensing, P1, P2, y, cosmo, N_χs_2, kwargs...; ndrange=size(int_ξs))
KernelAbstractions.synchronize(backend)

res = trapz((χ1s, χ2s), reshape(int_ξs, N_χs_2, N_χs_2))
```

See also: [`kernel_1d_P1!`](@ref), [`kernel_1d_P2!`](@ref)
"""
@kernel function kernel_2d!(results_vector, integrand, P1, P2, y, cosmo, N_χs_2, kwargs...)
    i, j = @index(Global, NTuple)
    IP1 = GaPSE.Point(P1.comdist * lr(1e-6, 1, N_χs_2, i), cosmo)
    IP2 = GaPSE.Point(P2.comdist * lr(1e-6, 1, N_χs_2, j), cosmo)
    #IP1 = GaPSE.Point(χ1s[i], cosmo)
    #IP2 = GaPSE.Point(χ2s[j], cosmo) 
    results_vector[i, j] = integrand(IP1, IP2, P1, P2, y, cosmo; kwargs...)
end

