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
    func_ℛ_LD(s, ℋ; s_lim=0.01, ℋ_0 = ℋ0)

Given in inpuit a comoving distance `s` and a comoving Hubble parameter `ℋ`, this
function returns the following value:
```math
\\mathfrak{R}(s; s_{\\mathrm{lim}})=
    \\begin{cases}
    1 - \\frac{1}{\\mathcal{H} \\, s} \\; ,
        \\quad s > s_\\mathrm{lim}\\\\
    1 - \\frac{1}{\\mathcal{H}_0 \\, s_\\mathrm{lim}} \\; , 
        \\quad \\quad 0 \\leq s \\leq s_\\mathrm{lim}
    \\end{cases}
```

The ``0 \\leq s \\leq s_\\mathrm{lim}`` case is used in order to avoid 
the divergence of the denominator.
This function is used inside a `Cosmology` for the computations concering the
Two-Point Correlation Fuctions (TPCFs) relative to the perturbed Luminosity Distance (LD).
The default value of the comoving Hubble parameter nowadays is, in natural system
(where the speed of light c=1): 
``\\mathcal{H}_0 \\simeq 3.335641\\times10^{-4} \\; h_0^{-1}\\mathrm{Mpc}``

See also: [`func_ℛ_GNC`](@ref), [`Cosmology`](@ref), [`ℋ0`](@ref)
"""
function func_ℛ_LD(s, ℋ; s_lim=0.01, ℋ_0=ℋ0)
    if s > s_lim
        return 1.0 - 1.0 / (s * ℋ)
    else
        return 1.0 - 1.0 / (s_lim * ℋ_0)
    end
end



"""
    func_ℛ_GNC(s, ℋ, ℋ_p; s_b = 0.0, 𝑓_evo = 0.0, s_lim=0.01, ℋ_0 = ℋ0)


Given in input a comoving distance `s`, a comoving Hubble parameter `ℋ` and
its first derivative value `ℋ_p` wrt the comoving time ``\\tau``, 
this function returns the following value:

```math
\\mathcal{R}(s; s_{\\mathrm{lim}})=
    \\begin{cases}
        5 s_{\\mathrm{b}} + \\frac{2 - 5 s_{\\mathrm{b}}}{\\mathcal{H} \\, s} +  
            \\frac{\\dot{\\mathcal{H}}}{\\mathcal{H}^2} - \\mathit{f}_{\\mathrm{evo}}\\; ,
            \\quad s > s_\\mathrm{lim}\\\\
        5 s_{\\mathrm{b}} + 
            \\frac{2 - 5 s_{\\mathrm{b}}}{\\mathcal{H}_0 \\, s_\\mathrm{lim}} +  
            \\frac{\\dot{\\mathcal{H}}}{\\mathcal{H}_0^2} - \\mathit{f}_{\\mathrm{evo}}\\; , 
            \\quad \\quad 0 \\leq s \\leq s_\\mathrm{lim}
    \\end{cases}
```

where ``s_{\\mathrm{b}}`` is the magnification bias (i.e. the slope of the luminosity 
function at the luminosity threshold), ``\\mathit{f}_{\\mathrm{evo}}`` the evolution bias
and ``\\dot{\\mathcal{H}} = \\mathrm{d}\\mathcal{H} / \\mathrm{d}\\tau``.
 
The ``0 \\leq s \\leq s_\\mathrm{lim}`` case is used in order to avoid 
the divergence of the denominator.
This function is used inside a `Cosmology` for the computations concering the
Two-Point Correlation Fuctions (TPCFs) relative to the Galaxy Number Counts (GNC).
The default value of the comoving Hubble parameter nowadays is, in natural system
(where the speed of light c=1): 
``\\mathcal{H}_0 \\simeq 3.335641\\times10^{-4} \\; h_0^{-1}\\mathrm{Mpc}``

See also: [`func_ℛ_GNC`](@ref), [`Cosmology`](@ref), [`ℋ0`](@ref)
"""
function func_ℛ_GNC(s, ℋ, ℋ_p; s_b=0.0, 𝑓_evo=0.0, s_lim=0.01, ℋ_0=ℋ0)
    if s_b ≈ 2.0 / 5.0
        2.0 + ℋ_p / ℋ^2 - 𝑓_evo
    elseif s > s_lim
        return 5.0 * s_b + (2.0 - 5.0 * s_b) / (s * ℋ) + ℋ_p / ℋ^2 - 𝑓_evo
    else
        return 5.0 * s_b + (2.0 - 5.0 * s_b) / (s_lim * ℋ_0) + ℋ_p / ℋ_0^2 - 𝑓_evo
    end
end



##########################################################################################92




"""
    CosmoSplines(
        z_of_s::GaPSE.MySpline
        D_of_s::GaPSE.MySpline
        f_of_s::GaPSE.MySpline
        ℋ_of_s::GaPSE.MySpline
        ℋ_p_of_s::GaPSE.MySpline
        ℛ_LD_of_s::GaPSE.MySpline
        ℛ_GNC1_of_s::GaPSE.MySpline
        ℛ_GNC2_of_s::GaPSE.MySpline

        s_of_z::GaPSE.MySpline

        z_eff::Float64
        s_min::Float64
        s_max::Float64
        s_eff::Float64

        file_data::String
        names::Vector{String}
        z_min::Float64
        z_max::Float64
        h::Float64
        s_lim::Float64
        z_spline_lim::Float64
        s_spline_lim::Float64

        s_b1::Float64
        𝑓_evo1::Float64
        s_b2::Float64
        𝑓_evo2::Float64
    )

Struct that contains all the useful cosmological splines.
It is used only inside the creation of a `Cosmology`, check its documentation for further information.

## Constructors 

    CosmoSplines(
        file_data::String, z_min, z_max; 
        names::Vector{String} = NAMES_BACKGROUND, h=0.7, 
        s_lim = 0.01, z_spline_lim=1000.0, s_b1=0.0, s_b2=nothing,
        𝑓_evo1=0.0, 𝑓_evo2=nothing
        )

- `file_data::String` : file containing all the background data; it is expected that such file
  is a background output of the CLASS (link: https://github.com/lesgourg/class_public) code.
  It is managed through the struct `BackgroundData`.

- `z_min` and `z_max` : the minimum and maximum redshifts of the survey we want to study.

- `names = NAMES_BACKGROUND` : the column names of the `file_data`. If the colum order change from
  the default one `NAMES_BACKGROUND`, you must set as input the vector of string with the correct
  one, with the SAME names. They are, with the default order:\n
  $(GaPSE.NAMES_BACKGROUND)  

- `h::Float64` : today's Hubble adimensional parameter (`H_0 = h * 100 km/(s * Mpc)`).

- `s_lim::Float64` : the lower-bound value for the functions `func_ℛ_LD` and `func_ℛ_GNC`; it is necessary, because
  `ℛ_LD` and `ℛ_GNC` blows up for ``s \\rightarrow 0^{+}``. Consequently, if the `func_ℛ_LD`/`func_ℛ_GNC` input value is 
  `0 ≤ s < s_lim`, the returned value is always `func_ℛ_LD(s_lim)`/`func_ℛ_GNC(s_lim)`.

- `z_spline_lim::Float64` : the upper limit where to cut the cosmological splines (it will be used by `CosmoSplines`);
  the recommended value is the recombination era (i.e. ``z \\simeq 1000 ``).

- `s_spline_lim::Float64` : the comoving distance converted from the redshift `z_spline_lim` with the 
  spline `s_of_z`;

- `s_b1::Float64` and  `s_b2::Float64`: magnification biases, i.e. the slope of the luminosity function at the luminosity threshold; 
  you can choose to define both of them (if you are interested in the analysis of two galaxy species) or only 
  the former (and leave the latter as `nothing`, it will be set equal to the former).

- `𝑓_evo1::Float64` and `𝑓_evo2::Float64`: evolution biases; you can choose to define both of them (if you are interested
  in the analysis of two galaxy species) or only the former (and leave the latter as `nothing`, it will be set equal
  to the former).

See also: [`Cosmology`](@ref)
"""
struct CosmoSplines
    z_of_s::GaPSE.MySpline
    D_of_s::GaPSE.MySpline
    f_of_s::GaPSE.MySpline
    ℋ_of_s::GaPSE.MySpline
    ℋ_p_of_s::GaPSE.MySpline
    ℛ_LD_of_s::GaPSE.MySpline
    ℛ_GNC1_of_s::GaPSE.MySpline
    ℛ_GNC2_of_s::GaPSE.MySpline

    s_of_z::GaPSE.MySpline

    z_eff::Float64
    s_min::Float64
    s_max::Float64
    s_eff::Float64

    file_data::String
    names::Vector{String}
    z_min::Float64
    z_max::Float64
    h::Float64
    s_lim::Float64
    z_spline_lim::Float64
    s_spline_lim::Float64

    s_b1::Float64
    𝑓_evo1::Float64
    s_b2::Float64
    𝑓_evo2::Float64

    function CosmoSplines(
            file_data::String, z_min, z_max; 
            names::Vector{String} = NAMES_BACKGROUND, h=0.7, 
            s_lim = 0.01, z_spline_lim=1000.0, 
            s_b1=0.0, s_b2=nothing, 𝑓_evo1=0.0, 𝑓_evo2=nothing
            )

        BD = BackgroundData(file_data, z_spline_lim; names=names, h=h)
        s_b2 = isnothing(s_b2) ? s_b1 : s_b2
        𝑓_evo2 = isnothing(𝑓_evo2) ? 𝑓_evo1 : 𝑓_evo2

        z_of_s = GaPSE.MySpline(BD.comdist, BD.z; bc="error")
        s_of_z = GaPSE.MySpline(BD.z, BD.comdist; bc="error")
        D_of_s = GaPSE.MySpline(BD.comdist, BD.D; bc="error")
        f_of_s = GaPSE.MySpline(BD.comdist, BD.f; bc="error")
        ℋ_of_s = GaPSE.MySpline(BD.comdist, BD.ℋ; bc="error")

        ℋ_of_τ = GaPSE.MySpline(reverse(BD.conftime), reverse(BD.ℋ); bc="error")
        vec_ℋs_p = [GaPSE.derivative(ℋ_of_τ, t) for t in BD.conftime]
        ℋ_p_of_s = GaPSE.MySpline(BD.comdist, vec_ℋs_p; bc="error")

        #println(BD.z[end], " ",BD.comdist[end])
        first_ss = 10.0 .^ range(-4, log10(BD.comdist[end]), length=1000)
        ss = vcat(first_ss[begin:end-1], BD.comdist[end])
        ℛ_LDs = [func_ℛ_LD(s, ℋ_of_s(s); s_lim=s_lim) for s in ss]
        ℛ_LD_of_s = GaPSE.MySpline(vcat(0.0, ss), vcat(ℛ_LDs[begin], ℛ_LDs); bc="error")

        ℛ_GNC1s = [func_ℛ_GNC(s, ℋ_of_s(s), ℋ_p_of_s(s);
            s_b=s_b1, 𝑓_evo=𝑓_evo1, s_lim=s_lim) for s in ss]
        ℛ_GNC1_of_s = GaPSE.MySpline(vcat(0.0, ss), vcat(ℛ_GNC1s[begin], ℛ_GNC1s); bc="error")
        ℛ_GNC2s = [func_ℛ_GNC(s, ℋ_of_s(s), ℋ_p_of_s(s);
            s_b=s_b2, 𝑓_evo=𝑓_evo2, s_lim=s_lim) for s in ss]
        ℛ_GNC2_of_s = GaPSE.MySpline(vcat(0.0, ss), vcat(ℛ_GNC2s[begin], ℛ_GNC2s); bc="error")

        s_min = s_of_z(z_min)
        s_max = s_of_z(z_max)
        z_eff = GaPSE.func_z_eff(s_min, s_max, z_of_s)
        s_eff = s_of_z(z_eff)
        s_spline_lim = s_of_z(z_spline_lim)

        new(z_of_s, D_of_s, f_of_s, ℋ_of_s, ℋ_p_of_s, ℛ_LD_of_s, ℛ_GNC1_of_s, ℛ_GNC2_of_s,
            s_of_z,
            z_eff, s_min, s_max, s_eff,
            file_data, names, z_min, z_max, h, s_lim, z_spline_lim, s_spline_lim,
            s_b1, s_b2, 𝑓_evo1, 𝑓_evo2)
    end
end



##########################################################################################92


"""
    Cosmology(
        IPS::InputPS
        params::CosmoParams
        tools::IPSTools
        windowF::WindowF

        z_of_s::GaPSE.MySpline
        D_of_s::GaPSE.MySpline
        f_of_s::GaPSE.MySpline
        ℋ_of_s::GaPSE.MySpline
        ℋ_p_of_s::GaPSE.MySpline
        ℛ_LD_of_s::GaPSE.MySpline
        ℛ_GNC1_of_s::GaPSE.MySpline
        ℛ_GNC2_of_s::GaPSE.MySpline

        s_of_z::GaPSE.MySpline

        z_eff::Float64
        s_min::Float64
        s_max::Float64
        s_eff::Float64
        s_spline_lim::Float64

        volume::Float64

        file_data::String
        file_ips::String
        file_windowF::String
        )

Struct that contains all the information that may be used for the 
Two-Point Correlation Function (TPCF) computations.
We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

## Arguments 

- `IPS::InputPS` : the matter Input Power Spectrum at present day of the Universe we are focusiong on.

- `params::CosmoParams` : options and parameters decided for this Cosmology; check the documentation of
  `CosmoParams` for more information.

- `tools::IPSTools` : all the functions and integrals depending on the Input Power Spectrum; check the documentation
  of `IPSTools` for more information.

- `windowF::WindowF` : the window function ``F``, defined as:
  ```math
    \\begin{split}
    F(x,\\mu; \\theta_\\mathrm{max}) = & \\;4\\pi 
    \\int_0^{\\theta_\\mathrm{max}} \\mathrm{d}\\theta_1 \\int_0^\\pi \\mathrm{d} \\theta \\; 
    \\, \\Theta\\left(\\frac
        {x \\cos \\theta + \\cos \\theta_1}{\\sqrt{x^1+2+2x\\mu}} - 
        \\cos(\\theta_\\mathrm{max}) 
        \\right) 
    \\, \\Theta(\\mu-\\cos(\\theta+\\theta_1)) \\\\
    &\\Theta(\\cos(\\theta - \\theta_1)-\\mu) \\;
    \\frac{\\sin\\theta\\sin\\theta_1}
        {\\sqrt{(\\sin\\theta\\sin\\theta_1)^2-(\\cos\\theta\\cos\\theta_1-\\mu)^2}}
    \\end{split}
  ```

- `windowFint::WindowFIntegrated` : the Integrated Window Function ``\\mathcal{F}``, defined as:
  ```math
  \\mathcal{F}(s, \\mu) = 
  \\int_0^\\infty \\mathrm{d}s_1 \\, \\phi(s_1) \\,  
  \\phi\\left(\\sqrt{s_1^2 + s^2 + 2 \\, s_1 \\, s \\, \\mu}\\right) 
  \\, F\\left(\\frac{s}{s_1}, \\mu \\right)
  ```
  where ``\\phi`` is the angular part of the survey window function and ``F(x, μ)`` is the 
  window function.

- `WFI_norm::Float64` : the norm of the Integrate Window Function, obtained from:
  ```math
  \\mathrm{norm \\, of } \\, \\mathcal{F} = \\frac{1}{2} \\int_{-1}^{1} \\, \\mathrm{d}\\mu \\, 
  \\mathcal{F}\\left(s = 10 \\, h_0^{-1}\\, \\mathrm{Mpc}, \\mu\\right) 
  ```

- `z_of_s, D_of_s, f_of_s, ℋ_of_s, ℋ_p_of_s, ℛ_LD_of_s, ℛ_GNC1_of_s, ℛ_GNC2_of_s ::GaPSE.MySpline` :
  splines obtained from the data stored by `BackgroundData` applied to the input background 
  data file. Given an input comoving distance `s`, they return the corresponding value of,
  respectively:
  - the redshift `z`;
  - the linear growth factor `D` (normalized to 1.0 at present day);
  - the linear growth rate `f`;
  - the comoving Hubble parameter `ℋ`;
  - the derivative of the comoving Hubble parameter wrt the comoving time `ℋ_p`; 
  - `ℛ_LD`, obtained from `func_ℛ_LD` and defined as:
  ```math
     \\mathfrak{R} = 1 - \\frac{1}{\\mathcal{H} \\, s}
  ```
  where ``s`` is the comoving distance and \\mathcal{H} is comoving Hubble parameter.
  It's spline is obtained in a sample of point given by 
  `10.0 .^ range(-4, log10(max(comdist...)), length=1000)`.
  - `ℛ_GNC`, obtained from `func_ℛ_GNC` and defined as:
  ```math
     \\mathcal{R} = 5 s_{\\mathrm{b}} + \\frac{2 - 5 s_{\\mathrm{b}}}{\\mathcal{H} \\, s} +  
     \\frac{\\dot{\\mathcal{H}}}{\\mathcal{H}^2} - \\mathit{f}_{\\mathrm{evo}}
  ```
  where``s`` is the comoving distance, \\mathcal{H} is comoving Hubble parameter,
  ``s_{\\mathrm{b}}`` is the magnification bias (i.e. the slope of the luminosity 
  function at the luminosity threshold), ``\\mathit{f}_{\\mathrm{evo}}`` the evolution bias
  and ``\\dot{\\mathcal{H}} = \\mathrm{d}\\mathcal{H} / \\mathrm{d}\\tau`` the first derivative
  of the comoving Hubble parameter wrt the comoving time ``\\tau``.
  It's spline is obtained in a sample of point given by 
  `10.0 .^ range(-4, log10(max(comdist...)), length=1000)`.
  NOTE: there are two of these splines in case you are taking into account two galaxies species (which
  have different values for galaxy, magnification and evolutionary biases); if you don't (i.e. you set only
  the first species values in `CosmoParams`) the two splines coincide.

- `s_of_z ::GaPSE.MySpline` : spline that returns the value of the comoving distance `s`
  corresponding to an input redshift `z`. Also this spline is obtained from the data stored by 
  `BackgroundData` applied to the input background data file.

- `z_eff::Float64` : effective redshift of this survey; its value is obtained through
  the function `func_z_eff`, with inputs the `s_min`, `s_max` and `z_of_s` here stored.

- `s_min::Float64` and `s_max::Float64` : the minimum and maximum comoving distances of
  the survey considered; they are the corresponding comoving distance to the chosen minimum and
  maximum redshifts `z_min` and `z_max`, stored in the input `CosmoParams`.

- `s_eff::Float64` : the corresponding comoving distance to the computed effective 
  redshifts `z_eff`.

- `s_spline_lim::Float64` : the comoving distance converted from the redshift `z_spline_lim` with the 
  spline `s_of_z`; `z_spline_lim` is the upper limit where to cut the cosmological splines,
  the recommended value is the recombination era (i.e. ``z \\simeq 1000 ``); it is set in `CosmoParams`.

- `volume::Float64` : volume of this survey. It is computed applying the function `V_survey`
  with inputs `s_min`, `s_max` here stored and the `θ_max` in the input `CosmoParams`.

- `file_data, file_ips, file_windowF::String` : the file names used for this Cosmology.

## Constructors

    Cosmology(
        params::CosmoParams,
        file_data::String,
        file_ips::String,
        file_windowF::String,
        file_IntwindowF::String;
        names_bg = NAMES_BACKGROUND)

- `params::CosmoParams` : parameters to be used for this Cosmology. See the docstring
  of `CosmoParams` for more information on the possible inputs.

- `file_data::String` : file containing all the background data; it is expected that such file
  is a background output of the CLASS (link: https://github.com/lesgourg/class_public) code.
  It is managed through the struct `BackgroundData`.

- `file_ips::String` : file containing the Input Power Spectrum at present day; it is expected that such file
  is a Power Spectrum output of the CLASS (link: https://github.com/lesgourg/class_public) code.
  It is managed through the struct `InputPS`.

- `file_windowF::String` : file containing a map of the window function `F`.
  This file is managed through the struct `WindowF`, and can be produced with the function `print_map_F`; see their
  docstrings for more information.

- `file_IntwindowF::String` : file containing a map of the Integrated Window Function `\\mathcal{F}`.
  This file is managed through the struct `WindowFIntegrated`, and can be produced with the function 
  `print_map_IntegartedF`; see their docstrings for more information.

- `names = NAMES_BACKGROUND` : the column names of the `file_data`. If the colum order change from
  the default one `NAMES_BACKGROUND`, you must set as input the vector of string with the correct
  one, with the SAME names. They are, with the default order:\n
  $(GaPSE.NAMES_BACKGROUND)  

See also: [`CosmoParams`](@ref), [`InputPS`](@ref), [`IPSTools`](@ref),
[`BackgroundData`](@ref), [`WindowF`](@ref), [`WindowFIntegrated`](@ref), 
[`print_map_F`](@ref), [`print_map_IntegratedF`](@ref), [`func_z_eff`](@ref),
[`V_survey`](@ref), [`func_ℛ_LD`](@ref), [`func_ℛ_GNC`](@ref), 
"""
struct Cosmology
    IPS::InputPS
    #ξ_matter::EPLs
    params::CosmoParams
    tools::IPSTools
    windowF::WindowF
    windowFint::WindowFIntegrated
    WFI_norm::Float64

    z_of_s::GaPSE.MySpline
    D_of_s::GaPSE.MySpline
    f_of_s::GaPSE.MySpline
    ℋ_of_s::GaPSE.MySpline
    ℋ_p_of_s::GaPSE.MySpline
    ℛ_LD_of_s::GaPSE.MySpline
    ℛ_GNC1_of_s::GaPSE.MySpline
    ℛ_GNC2_of_s::GaPSE.MySpline

    s_of_z::GaPSE.MySpline

    z_eff::Float64
    s_min::Float64
    s_max::Float64
    s_eff::Float64
    s_spline_lim::Float64

    volume::Float64

    file_data::String
    file_ips::String
    file_windowF::String
    file_IWF::Union{String,Nothing}

    function Cosmology(
        params::CosmoParams,
        file_data::String,
        file_ips::String,
        file_windowF::String,
        file_IntwindowF::String,
        #file_IntwindowF::Union{String,Nothing}=nothing;
        names_bg::Vector{String} = NAMES_BACKGROUND
    )

        #BD = BackgroundData(file_data, params.z_max; names=names_bg, h=params.h_0)
        IPS = InputPS(file_ips; params.IPS...)
        windowF = WindowF(file_windowF)
        tools = IPSTools(IPS; params.IPSTools...)

        #ss_m, xis_m = ξ_from_PS(IPS; int_k_min=1e-6, int_k_max=1e3,
        #     L=0, N=1024, pr=false, s0=nothing, right=nothing)
        #ξ_matter = EPLs(ss_m, xis_m, [1.0, 1.0], [-1.0, 1.0])

        CS = CosmoSplines(file_data, params.z_min, params.z_max; 
            names=names_bg, h=params.h_0, 
            s_lim = params.s_lim, z_spline_lim = params.z_spline_lim,
            s_b1=params.s_b1, s_b2=params.s_b2, 
            𝑓_evo1=params.𝑓_evo1, 𝑓_evo2=params.𝑓_evo2)

        vol = V_survey(CS.s_min, CS.s_max, params.θ_max)

        #=
        windowFintegrated = isnothing(file_IntwindowF) ?
                            WindowFIntegrated(s_min, s_max, windowF; params.WFI...) :
                            WindowFIntegrated(file_IntwindowF)
        =#
        windowFintegrated = WindowFIntegrated(file_IntwindowF)

        #WFI_norm = sum([spline_integrF(0, μ, windowFintegrated) 
        #     for μ in range(-0.90, 0.90, length=100)]) / 100
        WFI_norm = quadgk(μ -> spline_integrF(10.0, μ, windowFintegrated), -1, 1; rtol=1e-2)[1] / 2

        new(
            IPS,
            #ξ_matter,
            params,
            tools,
            windowF,
            windowFintegrated,
            WFI_norm,
            CS.z_of_s, CS.D_of_s, CS.f_of_s, CS.ℋ_of_s, CS.ℋ_p_of_s, CS.ℛ_LD_of_s, CS.ℛ_GNC1_of_s, CS.ℛ_GNC2_of_s,
            CS.s_of_z,
            CS.z_eff, CS.s_min, CS.s_max, CS.s_eff, CS.s_spline_lim,
            vol,
            file_data,
            file_ips,
            file_windowF,
            file_IntwindowF,
        )
    end
end



"""
    Point(
        z::Float64
        comdist::Float64
        D::Float64
        f::Float64
        ℋ::Float64
        ℋ_p::Float64
        ℛ_LD::Float64
        ℛ_GNC::Float64
        a::Float64
        )
     
A point in the Universe, placed at redshift `z` from us.
It contains all the relevant cosmological information at that redshift, respectively:
- the redshift `z`;
- the comoving distance `s`;
- the linear growth factor `D` (normalized to 1.0 at present day);
- the linear growth rate `f`;
- the comoving Hubble parameter `ℋ`;
- the first derivative `ℋ_p` of the comoving Hubble parameter `ℋ` wrt the comoving time ``\\tau``;
- the derivative of the comoving Hubble parameter wrt the comoving time `ℋ_p`; 
- `ℛ_LD`, obtained from `func_ℛ_LD` and defined as:
  ```math
  \\mathfrak{R} = 1 - \\frac{1}{\\mathcal{H} \\, s}
  ```
  where ``s`` is the comoving distance and \\mathcal{H} is comoving Hubble parameter.
- `ℛ_GNC`, obtained from `func_ℛ_GNC` and defined as:
  ```math
  \\mathcal{R} = 5 s_{\\mathrm{b}} + \\frac{2 - 5 s_{\\mathrm{b}}}{\\mathcal{H} \\, s} +  
  \\frac{\\dot{\\mathcal{H}}}{\\mathcal{H}^2} - \\mathit{f}_{\\mathrm{evo}}
  ```
  where``s`` is the comoving distance, \\mathcal{H} is comoving Hubble parameter,
  ``s_{\\mathrm{b}}`` is the magnification bias (i.e. the slope of the luminosity 
  function at the luminosity threshold), ``\\mathit{f}_{\\mathrm{evo}}`` the evolution bias
  and ``\\dot{\\mathcal{H}} = \\mathrm{d}\\mathcal{H} / \\mathrm{d}\\tau`` the first derivative
  of the comoving Hubble parameter wrt the comoving time ``\\tau``.
  NOTE: there are two of these values in case you are taking into account two galaxies species (which
  have different values for galaxy, magnification and evolutionary biases); if you don't (i.e. you set only
  the first species values in `CosmoParams`) the two splines coincide.

- the scale factor `a` (normalized to 1.0 at present day);

We remember that all the distances are measured in ``h_0^{-1}\\mathrm{Mpc}``.

## Constructors

`Point(s, cosmo::Cosmology)` : given a comoving distance `s`, it extrapolates all 
the data from the given input `Cosmology`.

See also: [`Cosmology`](@ref)
"""
struct Point
    z::Float64
    #conftime::Float64
    comdist::Float64
    #angdist::Float64
    #lumdist::Float64
    D::Float64
    f::Float64
    ℋ::Float64
    ℋ_p::Float64
    ℛ_LD::Float64
    ℛ_GNC1::Float64
    ℛ_GNC2::Float64
    a::Float64

    #Point(z, comdist, D, f, ℋ, ℛ_LD) = new(z, comdist, D, f, ℋ, ℛ_LD, 1.0/(1.0+z))
    function Point(s, cosmo::Cosmology)
        z = cosmo.z_of_s(s)
        new(z, s, cosmo.D_of_s(s), cosmo.f_of_s(s), cosmo.ℋ_of_s(s),
            cosmo.ℋ_p_of_s(s), cosmo.ℛ_LD_of_s(s), cosmo.ℛ_GNC1_of_s(s), cosmo.ℛ_GNC2_of_s(s),
            1.0 / (1.0 + z))
    end
end


function println_point(P::Point)
    println(
        "The input point contains the following data:\n",
        "  redshift = $(P.z), \t com. dist. = $(P.comdist) Mpc/h_0, \n"*
        "  D = $(P.D),  \t f = $(P.f), \t H com. = $(P.ℋ), \t deriv. H com. = $(P.ℋ_p), \n"*
        "  R_GNC_1 = $(P.ℛ_GNC1), \t R_GNC_2 = $(P.ℛ_GNC2), R_LD = $(P.ℛ_LD)"
    )
end
