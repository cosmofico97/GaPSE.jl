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

using Pkg

PATH_TO_GAPSE = "../"
Pkg.activate(normpath(PATH_TO_GAPSE)) # or @__DIR__
using GaPSE


const global B = 1.5     # Galaxy bias
const global S_B = 0.0   # Magnification bias
const global F_EVO = 0.0 # Evolution bias
const global filenames_appendix = "_b1p5-sb0-fevo0"
COMPUTE_XIS_GNC = false # set this to true if you want to compute the TPCFs of the GNC!
# This is the directory name where to put the files computed in the 
# following steps; makes sure that it's name ends with "/" !
DIR = "ARTICLE" * filenames_appendix * "/"
@assert isdir(DIR) "ERROR: DIR=$DIR DOESN'T EXIST!!!"

using DelimitedFiles, Dierckx, Printf

FILE_NAME = split(PROGRAM_FILE, "/")[end]

#main(x::Union{String, Float64, Int64}...) = main([string(var) for var in [x...]])
function main()

     # Have you followed the "ipynb/TUTORIAL.ipynb" file? 
     # All this stuff is there explained!
     # If you already know it, just wrote the code you want execute and
     # then run in the command line:
     #
     #    $ julia GaPSE-exe.jl
     #
     # Have a nice day!


     ########## This is the basic configuration for a Cosmology ###########

     ps_kwargs(alg::Symbol=:fftlog) = alg == :twofast ?
                                      Dict(
          :alg => :twofast, :epl => true, :pr => false,
          :N_left => 12, :N_right => 12,
          :p0_left => [-2.0, 1.0], :p0_right => [-2.0, 1.0],
          :int_s_min => 1e0, :int_s_max => 1200.0,
          :cut_first_n => 0, :cut_last_n => 0
     ) : alg == :fftlog ?
                                      Dict(
          :alg => :fftlog, :pr => true, :ν => 1.5,
          :n_extrap_low => 300, :n_extrap_high => 300,
          :n_pad => 500, :cut_first_n => 0, :cut_last_n => 0,
     ) : throw(AssertionError("alg = :fftlog (recommended) or alg = :twofast !"))
     tf = :fftlog

     FILE_F_MAP = PATH_TO_GAPSE * "data/NEW_F_pi2.txt"
     #FILE_F_MAP =  PATH_TO_GAPSE * "data/F_REFERENCE_pi2.txt";
     #=
     kwargs_map_F_hcub = Dict(
          :θ_max => π / 2.0, :tolerance => 1e-10, 
          :rtol => 1e-2, :atol => 1e-3, :pr => true,
     );

     kwargs_map_F_trap = Dict(
          :θ_max => π / 2.0, :tolerance => 1e-10, 
          :N => 1000, :pr => true,
     );

     xs = [x for x in 0:0.02:5]
     μs = union(
     [μ for μ in range(-1.0, -0.98, length = 50)], 
     [μ for μ in range(-0.98, 0.98, length = 102)],
     [μ for μ in range(0.98, 1.0, length = 50)]);
     GaPSE.print_map_F(FILE_F_MAP, xs, μs; 
     alg = :trap, Fmap_opts = kwargs_map_F_trap # we recommend to use :trap
     #alg = :hcub, Fmap_opts = kwargs_map_F_hcub # but you can use also :hcub if you prefer
     )
     =#

     z_min, z_max, θ_max = 1.0, 1.5, π / 2.0

     FILE_BACKGROUND = PATH_TO_GAPSE * "data/WideA_ZA_background.dat"

     #=
     WFI_opts = Dict(
         :llim => nothing, :rlim => nothing, 
         :rtol => 1e-2, :atol => 0.0, 
         :N => 1000, :pr => true,
     )
     =#

     params = GaPSE.CosmoParams(z_min, z_max, θ_max;
          Ω_b=0.0489, Ω_cdm=0.251020, 
          h_0 = 0.70, s_lim = 1e-2, z_spline_lim = 1000.0,
          b1=B, s_b1=S_B, 𝑓_evo1=F_EVO, 
          b2 = nothing, s_b2 = nothing, 𝑓_evo2 = nothing,
          IPS_opts=Dict(
               :fit_left_min => 1e-6, :fit_left_max => 3e-6,
               :fit_right_min => 1e1, :fit_right_max => 2e1),
          IPSTools_opts=Dict(
               :N => 1024, :fit_min => 0.05, :fit_max => 0.5,
               :con => true, :k_min => 1e-8, :k_max => 10.0)
          #=
          WFI_opts = Dict(
              :llim => nothing, :rlim => nothing, 
              :rtol => 1e-2, :atol => 0.0, 
              :N => 1000, :pr => true,)
          =#
     )

     #FILE_F_MAP =  PATH_TO_GAPSE * "data/NEW_F_pi2.txt";
     FILE_IF_MAP = PATH_TO_GAPSE * "data/NEW_IntegrF_pi2_z115.txt"

     #FILE_F_MAP = PATH_TO_GAPSE*"data/F_REFERENCE_pi2.txt";
     #FILE_IF_MAP = PATH_TO_GAPSE*"data/IntegrF_REFERENCE_pi2_z115.txt";

     #=
     calc_μs = union(
     [μ for μ in range(-1.0, -0.98, length = 50)], 
     [μ for μ in range(-0.98, 0.98, length = 102)],
     [μ for μ in range(0.98, 1.0, length = 50)]);

     GaPSE.print_map_IntegratedF(
     z_min, z_max, calc_μs,
     FILE_F_MAP, FILE_IF_MAP, 
     FILE_BACKGROUND;
     alg = :trap, N_ss = 200, m = 2.1,
     Dict(
          :llim => nothing, :rlim => nothing, 
          :rtol => 1e-2, :atol => 0.0, 
          :N => 1000, :pr => true,
     )...
     )
     =#


     FILE_PS = PATH_TO_GAPSE * "test/datatest/file_pk.txt"
     cosmo = GaPSE.Cosmology(params, FILE_BACKGROUND, FILE_PS, FILE_F_MAP, FILE_IF_MAP)

     GaPSE.parameters_used(stdout, cosmo)





     ###### Creation of a CosmoPNG struct ##########

     cosmopngparams = GaPSE.CosmoPNGParams(
          cosmo.D_of_s(cosmo.s_eff);
          bf=1.0,
          Dict(
               :flm_0 => 5e-2, :flM_0 => 1e-1, :s0_0 => 1e-4,
               :kmin_0 => 1e-6, :kmax_0 => 1e4, :N_0 => 1024,
               :flm_2 => 5e-1, :flM_2 => 1e0, :s0_2 => 1e-4,
               :kmin_2 => 1e-6, :kmax_2 => 1e4, :N_2 => 1024
          )...
     )
     cosmopng = GaPSE.CosmoPNG(cosmopngparams, cosmo, PATH_TO_GAPSE * "data/Tk.dat")





     ###### Code for TPCF and PS of S ##########

     GaPSE.print_map_ξ_S_multipole(cosmo, cosmopng, DIR * "xi_S_L0_noF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500); L=0, use_windows=false)
     GaPSE.print_map_ξ_S_multipole(cosmo, cosmopng, DIR * "xi_S_L2_noF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500); L=2, use_windows=false)
     GaPSE.print_PS_multipole(DIR * "xi_S_L0_noF" * filenames_appendix * ".txt",
          DIR * "ps_S_L0_noF" * filenames_appendix * ".txt"; L=0, ps_kwargs(tf)...)
     GaPSE.print_PS_multipole(DIR * "xi_S_L2_noF" * filenames_appendix * ".txt",
          DIR * "ps_S_L2_noF" * filenames_appendix * ".txt"; L=2, ps_kwargs(tf)...)

     GaPSE.print_map_ξ_S_multipole(cosmo, cosmopng, DIR * "xi_S_L0_withF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500); L=0, use_windows=true)
     GaPSE.print_map_ξ_S_multipole(cosmo, cosmopng, DIR * "xi_S_L2_withF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500); L=2, use_windows=true)
     GaPSE.print_PS_multipole(DIR * "xi_S_L0_withF" * filenames_appendix * ".txt",
          DIR * "ps_S_L0_withF" * filenames_appendix * ".txt"; L=0, ps_kwargs(tf)...)
     GaPSE.print_PS_multipole(DIR * "xi_S_L2_withF" * filenames_appendix * ".txt",
          DIR * "ps_S_L2_withF" * filenames_appendix * ".txt"; L=2, ps_kwargs(tf)...)


     ss_S_L0_withF, xis_S_L0_withF = GaPSE.readxy(DIR * "xi_S_L0_withF" * filenames_appendix * ".txt")
     ss_S_L2_withF, xis_S_L2_withF = GaPSE.readxy(DIR * "xi_S_L2_withF" * filenames_appendix * ".txt")
     ks_S_L0_withF, pks_S_L0_withF = GaPSE.readxy(DIR * "ps_S_L0_withF" * filenames_appendix * ".txt")
     ks_S_L2_withF, pks_S_L2_withF = GaPSE.readxy(DIR * "ps_S_L2_withF" * filenames_appendix * ".txt")

     spline_S_xis_L0_withF = Dierckx.Spline1D(ss_S_L0_withF, xis_S_L0_withF; bc="error")
     spline_S_xis_L2_withF = Dierckx.Spline1D(ss_S_L2_withF, xis_S_L2_withF; bc="error")
     spline_S_pks_L0_withF = Dierckx.Spline1D(ks_S_L0_withF, pks_S_L0_withF; bc="error")
     spline_S_pks_L2_withF = Dierckx.Spline1D(ks_S_L2_withF, pks_S_L2_withF; bc="error")


     ########## LIST OF GNC EFFECTS THAT CANNOT BE CONFUSED WITH THE PNG SIGNAL ##############

     LIST_GNC_SECURE = [
          "auto_newton", "auto_lensing", "lensing_newton", "newton_lensing",
          "doppler_newton", "newton_doppler",
          "doppler_lensing", "lensing_doppler",
     ]

     LIST_GNC_NON_SECURE = [
          "auto_doppler", "auto_localgp", "auto_integratedgp", "newton_localgp", "localgp_newton",
          "newton_integratedgp", "integratedgp_newton", "doppler_localgp", "localgp_doppler",
          "doppler_integratedgp", "integratedgp_doppler", "lensing_localgp", "localgp_lensing",
          "lensing_integratedgp", "integratedgp_lensing", "localgp_integratedgp", "integratedgp_localgp",
     ]


     for effect in LIST_GNC_SECURE
          @assert effect in GaPSE.GR_EFFECTS_GNC "1: $effect"
     end

     for effect in LIST_GNC_NON_SECURE
          @assert effect in GaPSE.GR_EFFECTS_GNC "2: $effect"
     end

     for effect in LIST_GNC_SECURE
          @assert !(effect in LIST_GNC_NON_SECURE) "3: $effect"
     end

     for effect in GaPSE.GR_EFFECTS_GNC
          if effect in LIST_GNC_SECURE || effect in LIST_GNC_NON_SECURE
               nothing
          else
               throw(AssertionError("$effect is not included!"))
          end
     end





     ######## Computation of the GNC TPCFs and Newtonian TPCF (and their PS) #########

     name_xis_GNC_L0_noF_noobsvel_file = "xis_GNC_L0_noF_noobsvel" * filenames_appendix * ".txt"
     name_ps_GNC_L0_noF_noobsvel_file = "ps_GNC_L0_noF_noobsvel" * filenames_appendix * ".txt"
     name_xis_GNC_L0_noF_noobs_file = "xis_GNC_L0_noF_noobs" * filenames_appendix * ".txt"
     name_ps_GNC_L0_noF_noobs_file = "ps_GNC_L0_noF_noobs" * filenames_appendix * ".txt"

     name_xis_GNC_L0_withF_noobsvel_file = "xis_GNC_L0_withF_noobsvel" * filenames_appendix * ".txt"
     name_ps_GNC_L0_withF_noobsvel_file = "ps_GNC_L0_withF_noobsvel" * filenames_appendix * ".txt"
     name_xis_GNC_L0_withF_noobs_file = "xis_GNC_L0_withF_noobs" * filenames_appendix * ".txt"
     name_ps_GNC_L0_withF_noobs_file = "ps_GNC_L0_withF_noobs" * filenames_appendix * ".txt"

     if COMPUTE_XIS_GNC == true
          GaPSE.print_map_sum_ξ_GNC_multipole(
               cosmo, DIR * name_xis_GNC_L0_noF_noobsvel_file,
               10 .^ range(0, log10(2 * cosmo.s_max), length=500);
               use_windows=false, L=0, alg=:quad, obs=:noobsvel,
               single=true, enhancer=1e8,
               N_trap=200, N_lob=200, atol_quad=0.0, rtol_quad=1e-2,
               N_χs=100, N_χs_2=60
          )
     end

     if COMPUTE_XIS_GNC == true
          GaPSE.print_map_sum_ξ_GNC_multipole(
               cosmo, DIR * name_xis_GNC_L0_withF_noobsvel_file,
               10 .^ range(0, log10(2 * cosmo.s_max), length=500);
               use_windows=true, L=0, alg=:quad, obs=:noobsvel,
               single=true, enhancer=1e8,
               N_trap=200, N_lob=200, atol_quad=0.0, rtol_quad=1e-2,
               N_χs=100, N_χs_2=60
          )
     end

     if COMPUTE_XIS_GNC == true
          GaPSE.print_map_sum_ξ_GNC_multipole(
               cosmo, DIR * name_xis_GNC_L0_noF_noobs_file,
               10 .^ range(0, log10(2 * cosmo.s_max), length=500);
               use_windows=false, L=0, alg=:quad, obs=:no,
               single=true, enhancer=1e8,
               N_trap=200, N_lob=200, atol_quad=0.0, rtol_quad=1e-2,
               N_χs=100, N_χs_2=60)
     end

     if COMPUTE_XIS_GNC == true
          GaPSE.print_map_sum_ξ_GNC_multipole(
               cosmo, DIR * name_xis_GNC_L0_withF_noobs_file,
               10 .^ range(0, log10(2 * cosmo.s_max), length=500);
               use_windows=true, L=0, alg=:quad, obs=:no,
               single=true, enhancer=1e8,
               N_trap=200, N_lob=200, atol_quad=0.0, rtol_quad=1e-2,
               N_χs=100, N_χs_2=60)
     end


     GaPSE.print_all_PS_multipole(DIR * name_xis_GNC_L0_noF_noobsvel_file,
          DIR * name_ps_GNC_L0_noF_noobsvel_file, "GNC"; L=0, ps_kwargs(tf)...)
     ks_GNC_L0_noF_noobsvel, pks_sum_GNC_L0_noF_noobsvel, pks_all_GNC_L0_noF_noobsvel =
          GaPSE.readxyall(DIR * name_ps_GNC_L0_noF_noobsvel_file)
     spline_GNCsum_pks_L0_noF_noobsvel = Dierckx.Spline1D(ks_GNC_L0_noF_noobsvel, pks_sum_GNC_L0_noF_noobsvel; bc="error")

     GaPSE.print_all_PS_multipole(DIR * name_xis_GNC_L0_noF_noobs_file,
          DIR * name_ps_GNC_L0_noF_noobs_file, "GNC"; L=0, ps_kwargs(tf)...)
     ks_GNC_L0_noF_noobs, pks_sum_GNC_L0_noF_noobs, pks_all_GNC_L0_noF_noobs =
          GaPSE.readxyall(DIR * name_ps_GNC_L0_noF_noobs_file)
     spline_GNCsum_pks_L0_noF_noobs = Dierckx.Spline1D(ks_GNC_L0_noF_noobs, pks_sum_GNC_L0_noF_noobs; bc="error")

     GaPSE.print_all_PS_multipole(DIR * name_xis_GNC_L0_withF_noobsvel_file,
          DIR * name_ps_GNC_L0_withF_noobsvel_file, "GNC"; L=0, ps_kwargs(tf)...)
     ks_GNC_L0_withF_noobsvel, pks_sum_GNC_L0_withF_noobsvel, pks_all_GNC_L0_withF_noobsvel =
          GaPSE.readxyall(DIR * name_ps_GNC_L0_withF_noobsvel_file)
     spline_GNCsum_pks_L0_withF_noobsvel = Dierckx.Spline1D(ks_GNC_L0_withF_noobsvel, pks_sum_GNC_L0_withF_noobsvel; bc="error")

     GaPSE.print_all_PS_multipole(DIR * name_xis_GNC_L0_withF_noobs_file,
          DIR * name_ps_GNC_L0_withF_noobs_file, "GNC"; L=0, ps_kwargs(tf)...)
     ks_GNC_L0_withF_noobs, pks_sum_GNC_L0_withF_noobs, pks_all_GNC_L0_withF_noobs =
          GaPSE.readxyall(DIR * name_ps_GNC_L0_withF_noobs_file)
     spline_GNCsum_pks_L0_withF_noobs = Dierckx.Spline1D(ks_GNC_L0_withF_noobs, pks_sum_GNC_L0_withF_noobs; bc="error")


     ks_Newtonian_L0_noF_noobsvel, pks_Newtonian_L0_noF_noobsvel =
          ks_GNC_L0_noF_noobsvel, pks_all_GNC_L0_noF_noobsvel[GaPSE.INDEX_GR_EFFECT_GNC["auto_newton"]]
     spline_Newt_pks_L0_noF_noobsvel = Dierckx.Spline1D(ks_Newtonian_L0_noF_noobsvel, pks_Newtonian_L0_noF_noobsvel; bc="error")
     ks_Newtonian_L0_noF_noobs, pks_Newtonian_L0_noF_noobs =
          ks_GNC_L0_noF_noobs, pks_all_GNC_L0_noF_noobs[GaPSE.INDEX_GR_EFFECT_GNC["auto_newton"]]
     spline_Newt_pks_L0_noF_noobs = Dierckx.Spline1D(ks_Newtonian_L0_noF_noobs, pks_Newtonian_L0_noF_noobs; bc="error")

     ks_Newtonian_L0_withF_noobsvel, pks_Newtonian_L0_withF_noobsvel =
          ks_GNC_L0_withF_noobsvel, pks_all_GNC_L0_withF_noobsvel[GaPSE.INDEX_GR_EFFECT_GNC["auto_newton"]]
     spline_Newt_pks_L0_withF_noobsvel = Dierckx.Spline1D(ks_Newtonian_L0_withF_noobsvel, pks_Newtonian_L0_withF_noobsvel; bc="error")
     ks_Newtonian_L0_withF_noobs, pks_Newtonian_L0_withF_noobs =
          ks_GNC_L0_withF_noobs, pks_all_GNC_L0_withF_noobs[GaPSE.INDEX_GR_EFFECT_GNC["auto_newton"]]
     spline_Newt_pks_L0_withF_noobs = Dierckx.Spline1D(ks_Newtonian_L0_withF_noobs, pks_Newtonian_L0_withF_noobs; bc="error")

     pks_PARTIAL_GNC_L0_withF_noobsvel = zeros(length(pks_sum_GNC_L0_withF_noobsvel))
     pks_PARTIAL_GNC_L0_withF_noobsvel += pks_sum_GNC_L0_withF_noobsvel
     for effect in GaPSE.GR_EFFECTS_GNC
          if effect in LIST_GNC_SECURE
               pks_PARTIAL_GNC_L0_withF_noobsvel .-= pks_all_GNC_L0_withF_noobsvel[GaPSE.INDEX_GR_EFFECT_GNC[effect]]
          end
     end
     spline_GNCsumPARTIAL_pks_L0_withF_noobsvel = Dierckx.Spline1D(ks_GNC_L0_withF_noobsvel, pks_PARTIAL_GNC_L0_withF_noobsvel; bc="error")

     pks_PARTIAL_GNC_L0_withF_noobs = zeros(length(pks_sum_GNC_L0_withF_noobs))
     pks_PARTIAL_GNC_L0_withF_noobs += pks_sum_GNC_L0_withF_noobs
     for effect in GaPSE.GR_EFFECTS_GNC
          if effect in LIST_GNC_SECURE
               pks_PARTIAL_GNC_L0_withF_noobs .-= pks_all_GNC_L0_withF_noobs[GaPSE.INDEX_GR_EFFECT_GNC[effect]]
          end
     end
     spline_GNCsumPARTIAL_pks_L0_withF_noobs = Dierckx.Spline1D(ks_GNC_L0_withF_noobs, pks_PARTIAL_GNC_L0_withF_noobs; bc="error")




     ############# Code for TPCFs and PS of PP ###########

     GaPSE.print_map_ξ_PPGalaxies_multipole(cosmo, DIR * "xi_ppg_L0_noF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500);
          L=0, use_windows=false, pr=true, enhancer=1e6,
          atol_quad=0.0, rtol_quad=1e-2)
     GaPSE.print_map_ξ_PPGalaxies_multipole(cosmo, DIR * "xi_ppg_L2_noF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500);
          L=2, use_windows=false, pr=true, enhancer=1e6,
          atol_quad=0.0, rtol_quad=1e-2)
     GaPSE.print_PS_multipole(DIR * "xi_ppg_L0_noF" * filenames_appendix * ".txt",
          DIR * "ps_ppg_L0_noF" * filenames_appendix * ".txt"; L=0, ps_kwargs(tf)...)
     GaPSE.print_PS_multipole(DIR * "xi_ppg_L2_noF" * filenames_appendix * ".txt",
          DIR * "ps_ppg_L2_noF" * filenames_appendix * ".txt"; L=2, ps_kwargs(tf)...)

     ss_ppg_L0_noF, xis_ppg_L0_noF = GaPSE.readxy(DIR * "xi_ppg_L0_noF" * filenames_appendix * ".txt")
     ss_ppg_L2_noF, xis_ppg_L2_noF = GaPSE.readxy(DIR * "xi_ppg_L2_noF" * filenames_appendix * ".txt")
     ks_ppg_L0_noF, pks_ppg_L0_noF = GaPSE.readxy(DIR * "ps_ppg_L0_noF" * filenames_appendix * ".txt")
     ks_ppg_L2_noF, pks_ppg_L2_noF = GaPSE.readxy(DIR * "ps_ppg_L2_noF" * filenames_appendix * ".txt")

     spline_ppg_xis_L0_noF = Dierckx.Spline1D(ss_ppg_L0_noF, xis_ppg_L0_noF; bc="error")
     spline_ppg_xis_L2_noF = Dierckx.Spline1D(ss_ppg_L2_noF, xis_ppg_L2_noF; bc="error")
     spline_ppg_pks_L0_noF = Dierckx.Spline1D(ks_ppg_L0_noF, pks_ppg_L0_noF; bc="error")
     spline_ppg_pks_L2_noF = Dierckx.Spline1D(ks_ppg_L2_noF, pks_ppg_L2_noF; bc="error")

     GaPSE.print_map_ξ_PPGalaxies_multipole(cosmo, DIR * "xi_ppg_L0_withF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500);
          L=0, use_windows=true, pr=true, enhancer=1e6,
          atol_quad=0.0, rtol_quad=1e-2)
     GaPSE.print_map_ξ_PPGalaxies_multipole(cosmo, DIR * "xi_ppg_L2_withF" * filenames_appendix * ".txt",
          10 .^ range(0, log10(2 * cosmo.s_max), length=500);
          L=2, use_windows=true, pr=true, enhancer=1e6,
          atol_quad=0.0, rtol_quad=1e-2)
     GaPSE.print_PS_multipole(DIR * "xi_ppg_L0_withF" * filenames_appendix * ".txt",
          DIR * "ps_ppg_L0_withF" * filenames_appendix * ".txt"; L=0, ps_kwargs(tf)...)
     GaPSE.print_PS_multipole(DIR * "xi_ppg_L2_withF" * filenames_appendix * ".txt",
          DIR * "ps_ppg_L2_withF" * filenames_appendix * ".txt"; L=2, ps_kwargs(tf)...)

     ss_ppg_L0_withF, xis_ppg_L0_withF = GaPSE.readxy(DIR * "xi_ppg_L0_withF" * filenames_appendix * ".txt")
     ss_ppg_L2_withF, xis_ppg_L2_withF = GaPSE.readxy(DIR * "xi_ppg_L2_withF" * filenames_appendix * ".txt")
     ks_ppg_L0_withF, pks_ppg_L0_withF = GaPSE.readxy(DIR * "ps_ppg_L0_withF" * filenames_appendix * ".txt")
     ks_ppg_L2_withF, pks_ppg_L2_withF = GaPSE.readxy(DIR * "ps_ppg_L2_withF" * filenames_appendix * ".txt")

     spline_ppg_xis_L0_withF = Dierckx.Spline1D(ss_ppg_L0_withF, xis_ppg_L0_withF; bc="error")
     spline_ppg_xis_L2_withF = Dierckx.Spline1D(ss_ppg_L2_withF, xis_ppg_L2_withF; bc="error")
     spline_ppg_pks_L0_withF = Dierckx.Spline1D(ks_ppg_L0_withF, pks_ppg_L0_withF; bc="error")
     spline_ppg_pks_L2_withF = Dierckx.Spline1D(ks_ppg_L2_withF, pks_ppg_L2_withF; bc="error")





     ########## Code for the differences: Newtonian - PPG , GNCsum - Newtonian , GNCsum - PPG #######

     ks = ks_GNC_L0_withF_noobsvel
     diff_Newt_ppg = [spline_Newt_pks_L0_withF_noobsvel(k) - spline_ppg_pks_L0_noF(k) for k in ks]
     #diff_ALLGRsum_ppg = pks_ALLGRsum_L0_withF .- pks_ppg_L0_withF;
     #diff_ALLGRsum_Newt = pks_ALLGRsum_L0_withF .- pks_Newtonian_L0_withF_noobsvel;
     diff_GNCsum_ppg = [spline_GNCsum_pks_L0_withF_noobsvel(k) - spline_ppg_pks_L0_withF(k) for k in ks]
     diff_GNCsum_Newt = [spline_GNCsum_pks_L0_withF_noobsvel(k) - spline_Newt_pks_L0_withF_noobsvel(k) for k in ks]



     ######### Write in a file the results ##########

     open(DIR * "RESULTS" * filenames_appendix * ".txt", "w") do io
          GaPSE.parameters_used(io, cosmo)
          println(io, "#\n#\n#")
          println(io,
               "# Here we report the Primordial Non-Gaussianities (PNG) Signal (S) in the Power Spectrum (PS or P) monopoles\n" *
               "# and the General Relativistic (GR) effects arising in the Galaxy Number Counts (GNC). \n" *
               "# All the PS are measured in h_0^{-3} Mpc^3 and are computed from the Two-Point Correlation Functions (TPCFs) \n" *
               "# considering an azimuthally simmetric window function; check the code for more information.\n" *
               "# As a short legend: \n" *
               "# \t - ppg = plane-parallel galaxies : PS monopole of the Galaxies using the Plane-Parallel approximation \n" *
               "# \t - S = Signal : signal of the PNG, using the f_{NL} prescription \n" *
               "# \t - delta delta : autocorrelation of the Newtonian term (= sum of density and redshift perturbations) \n" *
               "# \t - GNCsum : sum of all the 25 PS arising from the 25 GNC TPCFs\n" *
               "# \t - noobs = no-observer : all the terms in the TPCFs analytical expressions related to the observer are neglected \n" *
               "# \t - noobsvel = no-observer-velocity : only the observer terms related to his velocity are neglected (i.e. the ones \n" *
               "# \t     in the Doppler term), while the others are still taken into account"
          )
          println(io, "# LIST = " * string(LIST_GNC_SECURE[begin:end-1] .* " + "...) * LIST_GNC_SECURE[end])
          println(io, "# " *
                      "1: k [h_0/Mpc] \t " *
                      "2: P_0^{ppg} \t " *
                      "3: P_0^{S} \t " *
                      "4: P_0^{delta delta, noobsvel} \t " *
                      "5: P_0^{GNCsum, noobsvel} \t " *
                      "6: P_0^{GNCsum - LIST, noobsvel} \t " *
                      "7: P_0^{delta delta, noobs} \t " *
                      "8: P_0^{GNCsum, noobs} \t " *
                      "9: P_0^{GNCsum - LIST, noobs}"
          )

          for k in ks_GNC_L0_withF_noobsvel
               println(io,
                    "$k \t " *
                    "$(spline_ppg_pks_L0_withF(k)) \t " *
                    "$(spline_S_pks_L0_withF(k)) \t " *
                    "$(spline_Newt_pks_L0_withF_noobsvel(k)) \t " *
                    "$(spline_GNCsum_pks_L0_withF_noobsvel(k)) \t " *
                    "$(spline_GNCsumPARTIAL_pks_L0_withF_noobsvel(k)) \t " *
                    "$(spline_Newt_pks_L0_withF_noobs(k)) \t " *
                    "$(spline_GNCsum_pks_L0_withF_noobs(k)) \t " *
                    "$(spline_GNCsumPARTIAL_pks_L0_withF_noobs(k))"
               )
          end
     end
end







if (ARGS == String[])
     #println("\nwithout input arguments/commands, show this help message and exit\n")
     #main(["--help"])
     main()
else
     #main(ARGS)
     return 0
end
