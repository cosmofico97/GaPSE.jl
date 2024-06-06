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


using GaPSE, Test
using Dierckx, DelimitedFiles, QuadGK, Suppressor, NPZ


const FILE_F_MAP = "datatest/F_REFERENCE_pi2.txt"
const FILE_PS = "datatest/file_pk.txt"
const FILE_IF_MAP = "datatest/IntegrF_REFERENCE_pi2.txt"
const FILE_ILN = "datatest/table_Iln.txt"
const FILE_BACKGROUND = "datatest/WideA_ZA_background.dat"

const Z_MIN = 0.05
const Z_MAX = 0.20
const S_MIN = 148.1920001343431
const S_MAX = 571.7022420911966
const S_EFF = 435.37470960794167
const Z_EFF = 0.15045636097417317
const S_SPLINE_LIM = 9519.354776660555
const VOLUME = 3.845366167993746e8
const HUBBLE_0 = 1e5 / 299792458.0

include("TEST_DATA.jl")


##########################################################################################92


@testset "test FFTLog" begin
    include("test_FFTLog.jl")
end

@testset "test Spline" begin
    include("test_Spline.jl")
end

@testset "test MathUtils" begin
    include("test_MathUtils.jl")
end

@testset "test OtherUtils" begin
    include("test_OtherUtils.jl")
end


@testset "test CosmoUtils" begin
    include("test_CosmoUtils.jl")
end

@testset "test IPSTools" begin
    include("test_IPSTools.jl")
end


@testset "test BackgroundData" begin
    include("test_BackgroundData.jl")
end

@testset "test CosmoParams" begin
    include("test_CosmoParams.jl")
end

@testset "test Cosmology" begin
    include("test_Cosmology.jl")
end

@testset "test WindowF" begin
    include("test_WindowF.jl")
end


@testset "test WindowFIntegrated" begin
    include("test_WindowFIntegrated.jl")
end


@testset "test Dicts" begin
    include("test_Dicts.jl")
end


@testset "test PowerSpectra" begin
    include("test_PowerSpectra.jl")
end


@testset "test PowerSpectraGenWin" begin
    include("test_PowerSpectraGenWin.jl")
end

include("test_PowerSpectraeBOSS.jl")

@testset "test WindowF_QMultipoles" begin
    include("test_WindowF_QMultipoles.jl")
end


################################### COSMOLOGY IMPLEMENTATION #############################92


const PARAMS = GaPSE.CosmoParams(Z_MIN, Z_MAX, π / 2.0;
    Ω_b=0.0489, Ω_cdm=0.251020, h_0=0.7, s_lim=1e-2,
    b1=1.0, b2=nothing, s_b1=0.0, s_b2=nothing, 𝑓_evo1=0.0, 𝑓_evo2=nothing,
    IPS_opts=Dict(
        :fit_left_min => 1e-6, :fit_left_max => 3e-6,
        :fit_right_min => 1e1, :fit_right_max => 2e1,
    ),
    IPSTools_opts=Dict(
        :N => 1024, :fit_min => 0.05, :fit_max => 0.5,
        :con => true, :k_min => 1e-8, :k_max => 10.0,
    )
    #=
    WFI_opts=Dict(:llim => 0.0, :rlim => Inf, :N => 1000,
        :trap => true, :rtol => 1e-2, :atol => 0.0,
        :ss_start => 0.0, :ss_step => 21.768735478453323,
        :ss_stop => 0.0)
    =#
)

const COSMO = GaPSE.Cosmology(PARAMS, FILE_BACKGROUND, FILE_PS, FILE_F_MAP, FILE_IF_MAP)

@test_throws AssertionError GaPSE.parameters_used(COSMO; logo = "idk")
@test GaPSE.parameters_used(COSMO; logo="simple") isa Any
@test GaPSE.parameters_used(stdout, COSMO; logo=true) isa Any
@test GaPSE.parameters_used(stdout, COSMO; logo=false) isa Any


@testset "test_Point" begin
    P = GaPSE.Point(150.0, COSMO)

    z, s, D, f = 0.050617270275786205, 150.0, 0.9742776339718845, 0.5428326879328796         
    H, H_p = 0.00032500943586492186, 5.302066243063981e-8 #5.302174066399274e-8
    #R_GNC_1, R_GNC_2, R_LD = 41.526401599493504, 41.526401599493504, -19.512224871790487
    R_GNC_1, R_GNC_2, R_LD = 41.5263913750122, 41.5263913750122, -19.512224871790487

    @test z ≈ P.z
    @test s ≈ P.comdist
    @test D ≈ P.D
    @test f ≈ P.f
    @test H ≈ P.ℋ
    @test H_p ≈ P.ℋ_p
    @test R_LD ≈ P.ℛ_LD
    @test R_GNC_1 ≈ P.ℛ_GNC1
    @test R_GNC_2 ≈ P.ℛ_GNC2
    @test 1/(1+z) ≈ P.a
end

#=
common_kwargs = Dict(
    :pr => false,
    :use_windows => false,
    :enhancer => 1e8, :N_μs => 30,
    :μ_atol => 0.0, :μ_rtol => 1e-2,
    :N_log => 100,
);

spec_effect = [
    "auto_lensing", "auto_integratedgp",
    "lensing_doppler", "doppler_lensing",
    "doppler_integratedgp", "integratedgp_doppler",
    "lensing_localgp", "localgp_lensing",
    "lensing_integratedgp", "integratedgp_lensing",
    "localgp_integratedgp", "integratedgp_localgp",
];

specific_kwargs = [effect ∈ spec_effect ? Dict(
    :en => 1e12, :N_χs => 30) : nothing for effect in GaPSE.GR_EFFECTS_LD]

joint_kwargs = [isnothing(spec) ? common_kwargs : merge(common_kwargs, spec)
            for spec in specific_kwargs];
=#

dict_L_dir = Dict(0 => "monopoles", 1 => "dipoles", 2 => "tripoles",
    3 => "quadrupoles", 4 => "pentapoles")

KWARGS_LD = Dict(
    :pr => true,
    :use_windows => false,
    :enhancer => 1e8, :N_trap => 30, :N_lob => 30,
    :atol_quad => 0.0, :rtol_quad => 1e-2,
    :N_χs => 100, :N_χs_2 => 60,
    :N_log => 3,
);
SS_LD = 10 .^ range(0, log10(2.0 * COSMO.s_max), length=100);

KWARGS_GNC = Dict(
    :pr => true,
    :use_windows => false,
    :enhancer => 1e8, :N_trap => 30, :N_lob => 30,
    :atol_quad => 0.0, :rtol_quad => 1e-2,
    :N_χs => 100, :N_χs_2 => 60,
    :N_log => 3,
);
SS_GNC = 10 .^ range(-1, log10(2.0 * COSMO.s_max), length=100);

KWARGS_GNCxLD = Dict(
    :pr => true,
    :use_windows => false,
    :enhancer => 1e8,
    :N_trap => 30, :N_lob => 30,
    :atol_quad => 0.0, :rtol_quad => 1e-2,
    :N_χs => 100, :N_χs_2 => 60,
    :N_log => 3,
);
SS_GNCxLD = 10 .^ range(0, log10(2.0 * COSMO.s_max), length=100);

KWARGS_LDxGNC = Dict(
    :pr => true,
    :use_windows => false,
    :enhancer => 1e8,
    :N_trap => 30, :N_lob => 30,
    :atol_quad => 0.0, :rtol_quad => 1e-2,
    :N_χs => 100, :N_χs_2 => 60,
    :N_log => 3,
);
SS_LDxGNC = 10 .^ range(0, log10(2.0 * COSMO.s_max), length=100);



################################### TEST PLANE-PARALLEL APPROXIMATIONS ###################92


@testset "test PPDoppler" begin
    include("test_PPDoppler.jl")
end

@testset "test XiMatter" begin
    include("test_XiMatter.jl")
end

@testset "test PPXiGalaxies" begin
    include("test_PPXiGalaxies.jl")
end





################################### TEST PRIMORDIAL NON-GAUSSIANITES #####################92


@testset "test PNG" begin
    include("test_PNG.jl")
end


################################### TEST LUMINOSITY DISTANCE PERTURBATIONS ###############92

ss_LD_L0_noF, xis_sum_LD_L0_noF, all_xis_LD_L0_noF =
    GaPSE.readxyall("datatest/LD_SumXiMultipoles/xis_LD_L0_noF.txt", comments=true)

@testset "test LD_AutoDoppler" begin
    include("test_LD_AutoCorrelations/test_LD_AutoDoppler.jl")
end

@testset "test LD_AutoIntegratedGP" begin
    include("test_LD_AutoCorrelations/test_LD_AutoIntegratedGP.jl")
end

@testset "test LD_AutoLocalGP" begin
    include("test_LD_AutoCorrelations/test_LD_AutoLocalGP.jl")
end

@testset "test LD_AutoLensing" begin
    include("test_LD_AutoCorrelations/test_LD_AutoLensing.jl")
end


##############################



@testset "test LD_DopplerLensing" begin
    include("test_LD_CrossCorrelations/test_LD_DopplerLensing.jl")
end

@testset "test LD_DopplerLocalGP" begin
    include("test_LD_CrossCorrelations/test_LD_DopplerLocalGP.jl")
end

@testset "test LD_DopplerIntegratedGP" begin
    include("test_LD_CrossCorrelations/test_LD_DopplerIntegratedGP.jl")
end

@testset "test LD_LensingIntegratedGP" begin
    include("test_LD_CrossCorrelations/test_LD_LensingIntegratedGP.jl")
end

@testset "test LD_LensingLocalGP" begin
    include("test_LD_CrossCorrelations/test_LD_LensingLocalGP.jl")
end

@testset "test LD_LocalGPIntegratedGP" begin
    include("test_LD_CrossCorrelations/test_LD_LocalGPIntegratedGP.jl")
end


##############################


@testset "test LD_XiMultipoles" begin
    include("test_LD_XiMultipoles.jl")
end



@testset "test LD_SumXiMultipoles_P1" begin
    include("test_LD_SumXiMultipoles_P1.jl")
end


@testset "test LD_SumXiMultipoles_P2" begin
    include("test_LD_SumXiMultipoles_P2.jl")
end


################################### TEST RELATIVISTIC GALAXY NUMBER COUNTS ###############92

ss_GNC_L0_noF_noobs, xis_sum_GNC_L0_noF_noobs, all_xis_GNC_L0_noF_noobs =
    GaPSE.readxyall("datatest/GNC_SumXiMultipoles/xis_GNC_L0_noF_noobs.txt", comments=true)
ss_GNC_L0_noF_noobsvel, xis_sum_GNC_L0_noF_noobsvel, all_xis_GNC_L0_noF_noobsvel =
    GaPSE.readxyall("datatest/GNC_SumXiMultipoles/xis_GNC_L0_noF_noobsvel.txt", comments=true)
ss_GNC_L0_noF_withobs, xis_sum_GNC_L0_noF_withobs, all_xis_GNC_L0_noF_withobs =
    GaPSE.readxyall("datatest/GNC_SumXiMultipoles/xis_GNC_L0_noF_withobs.txt", comments=true)

@testset "test GNC_AutoNewton" begin
    include("test_GNC_AutoCorrelations/test_GNC_AutoNewton.jl")
end

@testset "test GNC_AutoDoppler" begin
    include("test_GNC_AutoCorrelations/test_GNC_AutoDoppler.jl")
end


@testset "test GNC_AutoIntegratedGP" begin
    include("test_GNC_AutoCorrelations/test_GNC_AutoIntegratedGP.jl")
end


@testset "test GNC_AutoLocalGP" begin
    include("test_GNC_AutoCorrelations/test_GNC_AutoLocalGP.jl")
end


@testset "test GNC_AutoLensing" begin
    include("test_GNC_AutoCorrelations/test_GNC_AutoLensing.jl")
end


##############################

@testset "test GNC_NewtonDoppler" begin
    include("test_GNC_CrossCorrelations/test_GNC_NewtonDoppler.jl")
end


@testset "test GNC_NewtonLensing" begin
    include("test_GNC_CrossCorrelations/test_GNC_NewtonLensing.jl")
end


@testset "test GNC_NewtonLocalGP" begin
    include("test_GNC_CrossCorrelations/test_GNC_NewtonLocalGP.jl")
end

@testset "test GNC_NewtonIntegratedGP" begin
    include("test_GNC_CrossCorrelations/test_GNC_NewtonIntegratedGP.jl")
end


@testset "test GNC_DopplerLensing" begin
    include("test_GNC_CrossCorrelations/test_GNC_DopplerLensing.jl")
end


@testset "test GNC_DopplerLocalGP" begin
    include("test_GNC_CrossCorrelations/test_GNC_DopplerLocalGP.jl")
end

@testset "test GNC_DopplerIntegratedGP" begin
    include("test_GNC_CrossCorrelations/test_GNC_DopplerIntegratedGP.jl")
end

@testset "test GNC_LensingIntegratedGP" begin
    include("test_GNC_CrossCorrelations/test_GNC_LensingIntegratedGP.jl")
end

@testset "test GNC_LensingLocalGP" begin
    include("test_GNC_CrossCorrelations/test_GNC_LensingLocalGP.jl")
end

@testset "test GNC_LocalGPIntegratedGP" begin
    include("test_GNC_CrossCorrelations/test_GNC_LocalGPIntegratedGP.jl")
end

##############################


@testset "test GNC_XiMultipoles" begin
    include("test_GNC_XiMultipoles.jl")
end


@testset "test GNC_SumXiMultipoles_P1" begin
    include("test_GNC_SumXiMultipoles_P1.jl")
end


@testset "test GNC_SumXiMultipoles_P2" begin
    include("test_GNC_SumXiMultipoles_P2.jl")
end


##### TEST RELATIVISTIC GALAXY NUMBER COUNTS X LUMINOSITY DISTANCE PERT. and viceversa ###92


ss_GNCxLD_L0_noF, xis_sum_GNCxLD_L0_noF, all_xis_GNCxLD_L0_noF =
    GaPSE.readxyall("datatest/GNCxLD_SumXiMultipoles/xis_GNCxLD_L0_noF.txt",
        comments=true);
ss_LDxGNC_L0_noF, xis_sum_LDxGNC_L0_noF, all_xis_LDxGNC_L0_noF =
    GaPSE.readxyall("datatest/LDxGNC_SumXiMultipoles/xis_LDxGNC_L0_noF.txt",
        comments=true);


@testset "test GNCxLD_NewtonDoppler" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_NewtonDoppler.jl")
end

@testset "test GNCxLD_NewtonLensing" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_NewtonLensing.jl")
end

@testset "test GNCxLD_NewtonLocalGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_NewtonLocalGP.jl")
end

@testset "test GNCxLD_NewtonIntegratedGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_NewtonIntegratedGP.jl")
end

############

@testset "test GNCxLD_DopplerDoppler" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_DopplerDoppler.jl")
end

@testset "test GNCxLD_DopplerLensing" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_DopplerLensing.jl")
end

@testset "test GNCxLD_DopplerLocalGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_DopplerLocalGP.jl")
end

@testset "test GNCxLD_DopplerIntegratedGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_DopplerIntegratedGP.jl")
end

############

@testset "test GNCxLD_LensingDoppler" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LensingDoppler.jl")
end

@testset "test GNCxLD_LensingLensing" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LensingLensing.jl")
end

@testset "test GNCxLD_LensingLocalGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LensingLocalGP.jl")
end

@testset "test GNCxLD_LensingIntegratedGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LensingIntegratedGP.jl")
end

############

@testset "test GNCxLD_LocalGPDoppler" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LocalGPDoppler.jl")
end

@testset "test GNCxLD_LocalGPLensing" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LocalGPLensing.jl")
end

@testset "test GNCxLD_LocalGPLocalGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LocalGPLocalGP.jl")
end

@testset "test GNCxLD_LocalGPIntegratedGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_LocalGPIntegratedGP.jl")
end

############

@testset "test GNCxLD_IntegratedGPDoppler" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_IntegratedGPDoppler.jl")
end

@testset "test GNCxLD_IntegratedGPLensing" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_IntegratedGPLensing.jl")
end

@testset "test GNCxLD_IntegratedGPLocalGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_IntegratedGPLocalGP.jl")
end

@testset "test GNCxLD_IntegratedGPIntegratedGP" begin
    include("test_GNCxLD_CrossCorrelations/test_GNCxLD_IntegratedGPIntegratedGP.jl")
end


##############################


@testset "test GNCxLD_XiMultipoles" begin
    include("test_GNCxLD_XiMultipoles.jl")
end


@testset "test LDxGNC_XiMultipoles" begin
    include("test_LDxGNC_XiMultipoles.jl")
end



@testset "test GNCxLD_SumXiMultipoles_P1" begin
    include("test_GNCxLD_SumXiMultipoles_P1.jl")
end

@testset "test GNCxLD_SumXiMultipoles_P2" begin
    include("test_GNCxLD_SumXiMultipoles_P2.jl")
end


@testset "test LDxGNC_SumXiMultipoles_P1" begin
    include("test_LDxGNC_SumXiMultipoles_P1.jl")
end

@testset "test LDxGNC_SumXiMultipoles_P2" begin
    include("test_LDxGNC_SumXiMultipoles_P2.jl")
end

##############################



##### TEST TWO SPECIES GNC,GNCxLD and LDxGNC (L=0) #######################################92


@testset "test TwoSpecies" begin
    include("test_TwoSpecies.jl")
end



##############################
