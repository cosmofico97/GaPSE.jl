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

module GaPSE


using TwoFAST # Licence: MIT "Expat" (o GPL ?)
using FFTW
using Base: @kwdef
using SpecialFunctions: gamma
import Base: *

include("FFTLog.jl")
using .FFTLog

using Dierckx # Licence: BSD
using HCubature, QuadGK, WignerSymbols # Licence: MIT "Expat"
using LegendrePolynomials, AssociatedLegendrePolynomials # Licence: MIT "Expat"
using SpecialFunctions, Trapz, LsqFit, FastGaussQuadrature, LinearAlgebra  # Licence: MIT
using GridInterpolations  # Licence: MIT "Expat"
using ProgressMeter, Printf  # Licence: MIT "Expat"

using Test, Documenter, DelimitedFiles  # Licence: MIT "Expat"

using KernelAbstractions
onlycpu = true





const BRAND_simple = """
        ###############
        #    GaPSE    #
        ############### \n#"""
 
const BRAND= """
###################################
#   ____       ____  ____  _____  #
#  / ___| __ _|  _ \\/ ___|| ____| #
# | |  _ / _` | |_) \\___ \\|  _|   #
# | |_| | (_| |  __/ ___) | |___  # 
#  \\____|\\__,_|_|   |____/|_____| #
###################################\n# \
"""



####### DO NOT MODIFY THESE NAMES AND/OR THEIR ORDER IF YOU ARE NOT SURE! ###############92

const NAMES_F_MAP = ["x", "mu", "F", "F_error"]
const NAMES_PS = ["k (h/Mpc)", "P (Mpc/h)^3"]
const NAMES_BACKGROUND = ["z", "proper time [Gyr]", "conf. time [Mpc]", "H [1/Mpc]",
     "comov. dist.", "ang.diam.dist.", "lum. dist.", "comov.snd.hrz.",
     "(.)rho_g", "(.)rho_b", "(.)rho_cdm", "(.)rho_lambda", "(.)rho_ur",
     "(.)rho_crit", "gr.fac. D", "gr.fac. f"]
const VALID_GROUPS = ["LD", "GNC", "GNCxLD", "LDxGNC", "generic"] # DO NOT MODIFY!
const LENGTH_VALID_GROUPS = [18, 27, 22, 22, nothing]

const HUBBLE_0 = 1e5 / 299792458.0

include("Spline.jl")
include("OtherUtils.jl")
include("MathUtils.jl")
#include("FFTLog.jl")
include("Wllnn.jl")
include("WindowF.jl")
include("WindowFIntegrated.jl")
include("BackgroundData.jl")
include("CosmoParams.jl")
include("CosmoUtils.jl")
include("IPSTools.jl")
include("XiMatter.jl")
include("Cosmology.jl")
include("PPXiGalaxies.jl")
include("PNG.jl")


##################### PERTURBED LUMINOSITY DISTANCE #############################


include("LD_AutoCorrelations/LD_AutoDoppler.jl")
include("LD_AutoCorrelations/LD_AutoLensing.jl")
include("LD_AutoCorrelations/LD_AutoLocalGP.jl")
include("LD_AutoCorrelations/LD_AutoIntegratedGP.jl")

include("LD_CrossCorrelations/LD_LensingDoppler.jl")
include("LD_CrossCorrelations/LD_DopplerLocalGP.jl")
include("LD_CrossCorrelations/LD_DopplerIntegratedGP.jl")
include("LD_CrossCorrelations/LD_LensingLocalGP.jl")
include("LD_CrossCorrelations/LD_LensingIntegratedGP.jl")
include("LD_CrossCorrelations/LD_LocalGPIntegratedGP.jl")

include("PPDoppler.jl")


####################################### GALAXY NUMBER COUNTS #############################92


include("GNC_AutoCorrelations/GNC_AutoNewtonian.jl")
include("GNC_AutoCorrelations/GNC_AutoDoppler.jl")
include("GNC_AutoCorrelations/GNC_AutoLensing.jl")
include("GNC_AutoCorrelations/GNC_AutoLocalGP.jl")
include("GNC_AutoCorrelations/GNC_AutoIntegratedGP.jl")

include("GNC_CrossCorrelations/GNC_NewtonianDoppler.jl")
include("GNC_CrossCorrelations/GNC_NewtonianLensing.jl")
include("GNC_CrossCorrelations/GNC_NewtonianLocalGP.jl")
include("GNC_CrossCorrelations/GNC_NewtonianIntegratedGP.jl")
include("GNC_CrossCorrelations/GNC_LensingDoppler.jl")
include("GNC_CrossCorrelations/GNC_DopplerLocalGP.jl")
include("GNC_CrossCorrelations/GNC_DopplerIntegratedGP.jl")
include("GNC_CrossCorrelations/GNC_LensingLocalGP.jl")
include("GNC_CrossCorrelations/GNC_LensingIntegratedGP.jl")
include("GNC_CrossCorrelations/GNC_LocalGPIntegratedGP.jl")


##################### GALAXY NUMBER COUNTS x LUMINOSITY DISTANCE #########################92


include("GNCxLD_CrossCorrelations/GNCxLD_NewtonianDoppler.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_NewtonianLensing.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_NewtonianLocalGP.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_NewtonianIntegratedGP.jl")

include("GNCxLD_CrossCorrelations/GNCxLD_DopplerDoppler.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_DopplerLensing.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_DopplerLocalGP.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_DopplerIntegratedGP.jl")

include("GNCxLD_CrossCorrelations/GNCxLD_LensingDoppler.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_LensingLensing.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_LensingLocalGP.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_LensingIntegratedGP.jl")

include("GNCxLD_CrossCorrelations/GNCxLD_LocalGPDoppler.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_LocalGPLensing.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_LocalGPLocalGP.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_LocalGPIntegratedGP.jl")

include("GNCxLD_CrossCorrelations/GNCxLD_IntegratedGPDoppler.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_IntegratedGPLensing.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_IntegratedGPLocalGP.jl")
include("GNCxLD_CrossCorrelations/GNCxLD_IntegratedGPIntegratedGP.jl")


##########################################################################################92


include("Dicts.jl")


include("LD_XiMultipoles.jl")
include("LD_SumXiMultipoles.jl")
include("GNC_XiMultipoles.jl")
include("GNC_SumXiMultipoles.jl")
include("GNCxLD_XiMultipoles.jl")
include("GNCxLD_SumXiMultipoles.jl")
include("LDxGNC_XiMultipoles.jl")
include("LDxGNC_SumXiMultipoles.jl")
include("PS_FFTLog.jl")
include("PS_TwoFAST.jl")
include("PowerSpectra.jl")
include("PowerSpectraGenWin.jl")
include("WindowF_QMultipoles.jl")


function parameters_used(io::IO, cosmo::Cosmology; logo::Union{Bool,String} = true)
    
    if typeof(logo) == Bool 
        logo && println(io, BRAND)
    elseif logo=="simple" 
        println(io, BRAND_simple)
    else
        throw(AssertionError(("bug to be fixed in parameters_used")))
    end

    println(io, "# The Cosmology considered had the following paremeters:\n# ")
    println(io, "# - Matter Power Spectrum input file: \"$(cosmo.file_ips)\"")
    println(io, "# - Background data input file: \"$(cosmo.file_data)\"")
    println(io, "# - F window function input file: \"$(cosmo.file_windowF)\"")
    println(io, "# - Integrated F window function input file: \"$(cosmo.file_IWF)\"")
    println(io, "#")

    println(io, "# - Basic CosmoParams considered: ")
    println(io, "#\t z_min = $(cosmo.params.z_min) \t z_max = $(cosmo.params.z_max)")
    println(io, "#\t θ_max = $(cosmo.params.θ_max) [rad] \t h_0 = $(cosmo.params.h_0)")
    println(io, "#\t Ω_b = $(cosmo.params.Ω_b) \t " *
                "Ω_cdm = $(cosmo.params.Ω_cdm) \t Ω_M0 = $(cosmo.params.Ω_M0)")
    println(io, "#\t b1 = $(cosmo.params.b1) \t " *
                "f_evo1 = $(cosmo.params.𝑓_evo1) \t s_b1 = $(cosmo.params.s_b1)")
    println(io, "#\t b2 = $(cosmo.params.b2) \t " *
                "f_evo2 = $(cosmo.params.𝑓_evo2) \t s_b2 = $(cosmo.params.s_b2)")
    println(io, "#")

    println(io, "# - CosmoParams about the Input Power Spectrum: ")
    my_println_dict(io, cosmo.params.IPS; pref="#\t ", N=2)
    println(io, "#")

    println(io, "# - CosmoParams about the Input Power Spectrum Tools: ")
    my_println_dict(io, cosmo.params.IPSTools; pref="#\t ", N=3)
    println(io, "#")

    #=
    println(io, "# - CosmoParams about the Integrated Window Function F: ")
    my_println_dict(io, cosmo.params.WFI; pref="#\t ", N=3)
    println(io, "#")
    =#

    println(io, "# - Computed quantities: ")
    println(io, "# \t effective redshift z_eff = $(cosmo.z_eff) ")
    println(io, "# \t comoving s_min = $(cosmo.s_min) Mpc/h_0")
    println(io, "# \t comoving s_max = $(cosmo.s_max) Mpc/h_0")
    println(io, "# \t comoving s_eff = $(cosmo.s_eff) Mpc/h_0")
    println(io, "# \t comoving s_spline_lim = $(cosmo.s_spline_lim) Mpc/h_0")
    println(io, "# \t Volume of the survey V_survey = $(cosmo.volume) Mpc^3/h_0^3")
    println(io, "# \t σ_0 = $(cosmo.tools.σ_0)")
    println(io, "# \t σ_1 = $(cosmo.tools.σ_1)")
    println(io, "# \t σ_2 = $(cosmo.tools.σ_2)")
    println(io, "# \t σ_3 = $(cosmo.tools.σ_3)")
    println(io, "# \t σ_4 = $(cosmo.tools.σ_4)")
    println(io, "# \t (where σ_i = \\int_{k_{min}}^{k_{max}}\\frac{dq}{2 π^2} q^{2-i} P(q))")
    println(io, "# ")
end

parameters_used(cosmo::Cosmology; kwargs...) = parameters_used(stdout, cosmo; kwargs...)


"""
    parameters_used(io::IO, cosmo::Cosmology; logo::Union{Bool,String} = true)
    parameters_used(cosmo::Cosmology; kwargs...) = parameters_used(stdout, cosmo; kwargs...)

Writes in the `io` stream all the information concerning the input Cosmology `cosmo`.
There are currently three options for `logo`: `true`, `"simple"` and `false`.
With `true`, the logo that is printed to `io` is the following:

```julia
$(string(BRAND))
```

With `false`, the logo that is printed to `io` is instead the following:

```julia
$(string(BRAND_simple))
```

Finally, with `false`, not logo is printed.

See also: [`Cosmology`](@ref)
"""
parameters_used


end # module
