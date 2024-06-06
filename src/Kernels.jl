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

"""
function lr(a, b, n, i)
    @assert (a<b) && (1≤i≤n) "Not valid inputs: a, b = $a, $b \t i, n = $i, $n"
    return a + (i - 1.0) / (n - 1.0) * (b - a)
end


@kernel function kernel_1d_P1!(results_vector, integrand, P1, P2, y, cosmo, N_χs, kwargs...)
    i = @index(Global, Linear)
    IP = GaPSE.Point(P1.comdist * lr(1e-6, 1, N_χs, i), cosmo)
    results_vector[i] = integrand(IP, P1, P2, y, cosmo; kwargs...)
end

@kernel function kernel_1d_P2!(results_vector, integrand, P1, P2, y, cosmo, N_χs, kwargs...)
    i = @index(Global, Linear)
    IP = GaPSE.Point(P2.comdist * lr(1e-6, 1, N_χs, i), cosmo)
    results_vector[i] = integrand(IP, P1, P2, y, cosmo; kwargs...)
end

@kernel function kernel_2d!(results_vector, integrand, P1, P2, y, cosmo, N_χs_2, kwargs...)
    i, j = @index(Global, NTuple)
    IP1 = GaPSE.Point(P1.comdist * lr(1e-6, 1, N_χs_2, i), cosmo)
    IP2 = GaPSE.Point(P2.comdist * lr(1e-6, 1, N_χs_2, j), cosmo)
    #IP1 = GaPSE.Point(χ1s[i], cosmo)
    #IP2 = GaPSE.Point(χ2s[j], cosmo) 
    results_vector[i, j] = integrand(IP1, IP2, P1, P2, y, cosmo; kwargs...)
end

