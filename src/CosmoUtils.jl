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
    func_z_eff(s_min, s_max, z_of_s) ::Float64

Given:
- the input comoving distances `s_min` and `s_max`
- the spline `z_of_s` that for an input comoving distance ``s`` return the associated redshift ``z(s)``,

this function return the effective redshift ``z_\\mathrm{eff}``, computed as follows:

```math
\\begin{split}
z_\\mathrm{eff} := 
\\frac{
    \\int \\mathrm{d}^3\\mathbf{s} \\, \\phi^2(\\mathbf{s}) \\, z(s)
    }{
        \\int \\mathrm{d}^3\\mathbf{s}\\, \\phi^2(\\mathbf{s}) 
    } &= \\frac{
        \\int_0^\\infty \\mathrm{d}s  \\, s^2 \\, \\phi^2(s) \\, z(s) \\times
        \\int_{4\\pi}\\mathrm{d}^2\\hat{\\mathbf{s}} \\, W^2(\\hat{\\mathbf{s}})
    }{
        \\int_0^\\infty \\mathrm{d}s \\, s^2 \\, \\phi^2(s)\\times
        \\int_{4\\pi}\\mathrm{d}^2\\hat{\\mathbf{s}} \\, W^2(\\hat{\\mathbf{s}})
    } \\\\[5pt]
    &= \\frac{
        \\int_0^\\infty \\mathrm{d}s  \\, s^2 \\, \\phi^2(s) \\, z(s)
    }{
        \\int_0^\\infty \\mathrm{d}s \\, s^2 \\, \\phi^2(s)
    } \\\\[4pt]
    &= \\frac{3}{s_\\mathrm{max}^3 - s_\\mathrm{min}^3} \\,
        \\int_{s_\\mathrm{min}}^{s_\\mathrm{max}} \\mathrm{d}s  \\, s^2 \\, z(s) \\; .
\\end{split}
```

We have used our assuption concerning the separability of the window function 
of the survey ``\\phi(\\mathbf{s})`` into a
radial and angular part, respectively ``\\phi(s)`` and ``W(\\mathbf{\\hat{s}})``:

```math
    \\phi(\\mathbf{s}) = \\phi(s) \\, W(\\mathbf{\\hat{s}}) \\; .
```

We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.
The final integration over ``\\mathrm{d}s`` is performed through the function `quadgk`
of the [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl) Julia package.

See also: [`ϕ`](@ref), [`W`](@ref)
"""
function func_z_eff(s_min, s_max, z_of_s)
    3.0 / (s_max^3 - s_min^3) * quadgk(s -> s^2 * z_of_s(s), s_min, s_max)[1]
end


"""
    s(s1, s2, y) ::Float64

Return the value ``s = s(s_1, s_2, y) = \\sqrt{s_1^2 + s_2^2 - 2 \\, s_1 \\, s_2 \\, y}``

See also: [`μ`](@ref), [`s2`](@ref), [`y`](@ref)
"""
s(s1, s2, y) = √(s1^2 + s2^2 - 2 * s1 * s2 * y)


"""
    μ(s1, s2, y) ::Float64

Return the cosine ``\\mu=\\hat{\\mathbf{s}}_1\\cdot\\hat{\\mathbf{s}}`` of the angle between 
the comoving distances ``\\mathbf{s}_1`` and ``\\mathbf{s} = \\mathbf{s}_2 - \\mathbf{s}_1``.
It can be easily shown that:
```math
\\mu = \\mu(s_1, s_2, y) = \\frac{y \\, s_2 - s_1}{s(s_1, s_2, y)} \\;,
\\quad s = s(s_1, s_2, y) = \\sqrt{s_1^2 + s^2 - 2 \\, s_1 \\, s_2 \\, y}
```
with ``y=\\cos\\theta=\\hat{\\mathbf{s}}_1\\cdot\\hat{\\mathbf{s}}_2`` and where ``s`` is 
obtained from the function `s`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

See also: [`s`](@ref), [`s2`](@ref), [`y`](@ref)
"""
μ(s1, s2, y) = (y * s2 - s1) / s(s1, s2, y)



"""
    s2(s1, s, μ) ::Float64

Return the value ``s_2 = \\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}``

See also: [`s`](@ref), [`μ`](@ref), [`y`](@ref)
"""
s2(s1, s, μ) = √(s1^2 + s^2 + 2 * s1 * s * μ)



"""
    y(s1, s, y) ::Float64

Return the cosine ``y=\\cos\\theta=\\hat{\\mathbf{s}}_1\\cdot\\hat{\\mathbf{s}}_2`` of the angle between 
the comoving distances ``\\mathbf{s}_1`` and ``\\mathbf{s}_2 = \\mathbf{s}_1 + \\mathbf{s}``.
It can be easily shown that:
```math
y = y(s_1, s, \\mu) = \\frac{\\mu \\, s + s_1}{s_2(s_1, s, \\mu)} \\;,
\\quad s_2 = s_2(s_1, s, \\mu) = \\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}
```
with ``\\mu=\\hat{\\mathbf{s}}_1\\cdot\\hat{\\mathbf{s}}`` and where ``s_2`` is 
obtained from the function `s2`.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

See also: [`s`](@ref), [`s2`](@ref), [`μ`](@ref)
"""
y(s1, s, μ) = (μ * s + s1) / s2(s, s1, μ)



##########################################################################################92




"""
    ϕ(s, s_min, s_max) ::Float64

Radial part of the survey window function. Return `1.0` if is true that
``s_\\mathrm{min} < s < s_\\mathrm{max}`` and `0.0` otherwise.

In this software we made the assuption that the survey window function can be
separated into a radial and angular part, i.e.:

```math
    \\phi(\\mathbf{s}) = \\phi(s) \\, W(\\mathbf{\\hat{s}})
```

See also: [`W`](@ref)
"""
ϕ(s, s_min, s_max) = s_min < s < s_max ? 1.0 : 0.0



"""
    W(θ, θ_max) ::Float64

Angular part of the survey window function. Return `1.0` if is true that
``0 \\leq \\theta < \\theta_\\mathrm{max}`` and `0.0` otherwise. It is
implicitly assumed an azimutal simmetry of the survey.

In this software we made the assuption that the survey window function can be
separated into a radial and angular part, i.e.:

```math
    \\phi(\\mathbf{s}) = \\phi(s) \\, W(\\mathbf{\\hat{s}})
```

See also: [`ϕ`](@ref)
"""
W(θ, θ_max) = 0.0 ≤ θ < θ_max ? 1.0 : 0.0


"""
    V_survey(s_min, s_max, θ_max) ::Float64

Return the volume of a survey with azimutal simmetry, i.e.:

```math
\\begin{split}
    V(s_\\mathrm{max}, s_\\mathrm{min}, \\theta_\\mathrm{max}) &= \\; C_\\mathrm{up} - C_\\mathrm{down} + TC \\\\
    &C_\\mathrm{up} = \\frac{\\pi}{3} s_\\mathrm{max}^3 \\, 
        (1 - \\cos\\theta_\\mathrm{max})^2 \\, (2 + \\cos\\theta_\\mathrm{max}) \\\\
    &C_\\mathrm{down} = \\frac{\\pi}{3} s_\\mathrm{min}^3 \\, 
        (1 - \\cos\\theta_\\mathrm{max})^2 \\, (2 + \\cos\\theta_\\mathrm{max}) \\\\
    &TC = \\frac{\\pi}{3} (s_\\mathrm{max}^2 + s_\\mathrm{min}^2 + 
        s_\\mathrm{max} \\,s_\\mathrm{min}) \\,  (s_\\mathrm{max} - s_\\mathrm{min})\\, 
        \\cos\\theta_\\mathrm{max}\\, \\sin^2\\theta_\\mathrm{max}
\\end{split}
```
"""
function V_survey(s_min, s_max, θ_max)
    sin_θ, cos_θ = sin(θ_max), cos(θ_max)
    diff_up_down = (s_max^3 - s_min^3) * (1 - cos_θ)^2 * (2 + cos_θ)
    tr = (s_max^2 + s_min^2 + s_max * s_min) * (s_max - s_min) * cos_θ * sin_θ^2
    #r1, r2 = s_min * sin(θ_max), s_max * sin(θ_max)
    #d1, d2 = s_min * cos(θ_max), s_max * cos(θ_max)
    #calotta_up = π / 3 * (s_max - d2)^2 * (2 * s_max + d2)
    #calotta_down = π / 3 * (s_min - d1)^2 * (2 * s_min + d1)
    #tronco_cono = π / 3 * (r1^2 + r1 * r2 + r2^2) * (s_max - s_min) * cos(θ_max)
    return π / 3.0 * (diff_up_down + tr)
end


"""
    A(s_min, s_max, θ_max) ::Float64

Return the Power Spectrum multipole normalization coefficient `A`, i.e.:
```math
    A(s_\\mathrm{max}, s_\\mathrm{min}, \\theta_\\mathrm{max})= 
    \\frac{
        V(s_\\mathrm{max}, s_\\mathrm{min}, \\theta_\\mathrm{max})
    }{4 \\, \\pi^2}
```
where ``V(s_\\mathrm{max}, s_\\mathrm{min}, \\theta_\\mathrm{max})`` is the 
survey volume.

Pay attention: this is NOT used for the normalization of [`InputPS`](@ref), see
instead [`A_prime`](@ref)

See also: [`V_survey`](@ref)
"""
function A(s_min, s_max, θ_max)
     V_survey(s_min, s_max, θ_max) / (4.0 * π^2)
end



"""
    A_prime :: Float64

It's the Power Spectrum multipole normalization coefficient ``A^{'}``, i.e.:
```math
    A^{'} = \\frac{3 \\, A}{ (s_\\mathrm{max}^3 - s_\\mathrm{min}^3)} = 
    \\frac{1}{4\\,\\pi}
```

See also: [`A`](@ref), [`V_survey`](@ref)
"""
const A_prime = 1.0 / (4.0 * π)



##########################################################################################92


"""
    corresponding_redshift(z, m, file_data::String; names_bg=NAMES_BACKGROUND, h_0=0.7, Z_MAX = 1e3)

For a given `file_data` containing the csomological background quantities (that will be managed
by the struct `BackgroundData`), return the redshift corresponding (for that cosmology) to the 
comoving distance equals to `m * s(z)`, where `z` is the input redshift and `m` a coeffient.

Essentially, in a sketch:

Input `z`, `m` -> Computes `s1 = s(z)`  -> Computes `s2 = m * s1 ` -> Return `z2 = z(s2)`

This function is not efficient, it should be used only in order to have a rough estimation for the
maximum sampling value of the Integrated Window Function.

## Optional arguments

- `names = NAMES_BACKGROUND` : the column names of the `file_data`. If the colum order change from
  the default one `NAMES_BACKGROUND`, you must set as input the vector of string with the correct
  one, with the SAME names. They are, with the default order:\n
  $(NAMES_BACKGROUND)

- `h = 0.7` : the adimensional hubble constant. By default, CLASS background data are measured with
  it numerically expressed (so distances are measured in `Mpc`, for example), while this code works
  with `h` in the unit of measure (so distances are measured in `Mpc/h`, for example).
  Change this value to `1.0` if the input data do not have this issue, or to your value of interest 
  (`0.67`, `0.5`, ...).

- `Z_MAX = 1e3` : maximum redshift to consider for BackgroundData; it should be high.

See also: [`BackgroundData`](@ref), [`WindowFIntegrated`](@ref)
"""
function corresponding_redshift(z, m, file_data::String; names_bg=NAMES_BACKGROUND, h_0=0.7, Z_MAX=1e3)
    @assert m > 0 "m > 0 must hold!"
    @assert 0.0 ≤ z < Z_MAX "0.0 ≤ z < Z_MAX must hold"

    BD = BackgroundData(file_data, Z_MAX; names=names_bg, h=h_0)
    s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
    z_of_s = GaPSE.MySpline(BD.comdist, BD.z; bc="error")

    s = s_of_z(z)
    return z_of_s(m * s)
end

