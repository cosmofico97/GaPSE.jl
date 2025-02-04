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


@testset "test create_file_for_XiMultipoles" begin
    RTOL = 1e-5

    @testset "zeros" begin
        name = "datatest/PowerSpectrumGenWin/xis_LD_L0123_lob_noF_specific_ss.txt"
        calc_name = "spec_ss.txt"
        isfile(calc_name) && rm(calc_name)

        names = "datatest/LD_SumXiMultipoles/" .* [
            "xis_LD_L0_lob_noF_specific_ss.txt",
            "xis_LD_L1_lob_noF_specific_ss.txt",
            "xis_LD_L2_lob_noF_specific_ss.txt",
            "xis_LD_L3_lob_noF_specific_ss.txt",
        ]

        @test_throws AssertionError GaPSE.create_file_for_XiMultipoles(calc_name, names, 1, "LDxGNC")
        @test_throws AssertionError GaPSE.create_file_for_XiMultipoles(calc_name, names, "newton_doppler", "LD")
        @test_throws AssertionError GaPSE.create_file_for_XiMultipoles(calc_name, names, "idk", "GNC")
        @test_throws AssertionError GaPSE.create_file_for_XiMultipoles(calc_name, names, "auto_newton", "LDxGNC")
        @test_throws AssertionError GaPSE.create_file_for_XiMultipoles(calc_name, names, "auto_newton", "GNCxLD")

        isfile(calc_name) && rm(calc_name)
    end

    @testset "first" begin
        name = "datatest/PowerSpectrumGenWin/xis_LD_L0123_lob_noF_specific_ss.txt"
        calc_name = "spec_ss.txt"
        isfile(calc_name) && rm(calc_name)

        names = "datatest/LD_SumXiMultipoles/" .* [
            "xis_LD_L0_lob_noF_specific_ss.txt",
            "xis_LD_L1_lob_noF_specific_ss.txt",
            "xis_LD_L2_lob_noF_specific_ss.txt",
            "xis_LD_L3_lob_noF_specific_ss.txt",
        ]

        GaPSE.create_file_for_XiMultipoles(calc_name, names, "auto_doppler", "LD")
        calc_ss, calc_all_xis = GaPSE.readxall(name)

        ss, all_xis = GaPSE.readxall(name)
        @test all(isapprox(t, c; rtol=RTOL) for (t, c) in zip(ss, calc_ss))
        @test all(isapprox(t, c; rtol=RTOL) for (t, c) in zip(all_xis, calc_all_xis))

        rm(calc_name)
    end

    @testset "second" begin

        name = "datatest/PowerSpectrumGenWin/xis_LD_auto_doppler_L0123_lob_noF.txt"
        calc_name = "aut_dop.txt"
        isfile(calc_name) && rm(calc_name)

        names = "datatest/LD_doppler_multipoles/" .* [
            "xi_LD_auto_doppler_lobatto_noF_L0.txt",
            "xi_LD_auto_doppler_lobatto_noF_L1.txt",
            "xi_LD_auto_doppler_lobatto_noF_L2.txt",
            "xi_LD_auto_doppler_lobatto_noF_L3.txt",
        ]

        GaPSE.create_file_for_XiMultipoles(calc_name, names, 2, "LD")
        calc_ss, calc_all_xis = GaPSE.readxall(name)

        ss, all_xis = GaPSE.readxall(name)
        @test all(isapprox(t, c; rtol=RTOL) for (t, c) in zip(ss, calc_ss))
        for (col, calc_col) in zip(all_xis, calc_all_xis)
            @test all(isapprox(t, c; rtol=RTOL) for (t, c) in zip(col, calc_col))
        end

        rm(calc_name)
    end
end


@testset "test XiMultipoles" begin
    RTOL = 1e-5

    name = "datatest/PowerSpectrumGenWin/xis_LD_auto_doppler_L0123_lob_noF.txt"
    ss, all_xis = GaPSE.readxall(name)

    ximult = GaPSE.XiMultipoles(name)

    @test all(isapprox(t, c; rtol=RTOL) for (t, c) in zip(ss, ximult.comdist))
    for (col, calc_col) in zip(all_xis, ximult.multipoles)
        @test all(isapprox(t, c; rtol=RTOL) for (t, c) in zip(col, calc_col))
    end


end


@testset "test PS_multipole_GenWin" begin
    RTOL = 1e-2

    xi_filenames = "datatest/PowerSpectrumGenWin/" .* [
        "xis_GNC_L0_noF_noobsvel_GenWin.txt",
        "xis_GNC_L1_noF_noobsvel_GenWin.txt",
        "xis_GNC_L2_noF_noobsvel_GenWin.txt",
        "xis_GNC_L3_noF_noobsvel_GenWin.txt",
        "xis_GNC_L4_noF_noobsvel_GenWin.txt",
    ]

    calc_name_file_ximultipoles = "calc_xis_sum_GNC_L01234_noF_noobsvel.txt"
    name_file_Qmultipoles = "datatest/PowerSpectrumGenWin/Ql_multipoles_of_F_REFERENCE_z115.txt"
    isfile(calc_name_file_ximultipoles) && rm(calc_name_file_ximultipoles)
    GaPSE.create_file_for_XiMultipoles(calc_name_file_ximultipoles, xi_filenames, 2, "GNC")


    ps_kwargs(alg::Symbol=:fftlog) = alg == :twofast ? begin
        Dict(
            :alg => :twofast, :epl => false, :pr => false,
            :N_left => 12, :N_right => 12,
            :p0_left => [-2.0, 1.0], :p0_right => [-2.0, 1.0],
            :int_s_min => 1e0, :int_s_max => 1200.0,
            :cut_first_n => 0, :cut_last_n => 0
        )
    end : alg == :fftlog ? begin
        Dict(
            :alg => :fftlog, :pr => true, :ν => 1.5,
            :n_extrap_low => 0, :n_extrap_high => 0,
            :n_pad => 500, :cut_first_n => 0, :cut_last_n => 0,
        )
    end : throw(AssertionError("alg = :fftlog (recommended) or alg = :twofast !"))

    @testset "fftlog" begin
        ks_fftlog, pks_fftlog = GaPSE.PS_multipole_GenWin(calc_name_file_ximultipoles, name_file_Qmultipoles;
            L=0, ps_kwargs(:fftlog)...)

        true_ks_fftlog, true_pks_fftlog, _ = GaPSE.readxyall("datatest/PowerSpectrumGenWin/ps_GNC_L0_withF_noobsvel_GenWin_fftlog.txt")

        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(ks_fftlog, true_ks_fftlog)])
        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(pks_fftlog, true_pks_fftlog)])

        #println("ks=$ks_fftlog;")
        #println("pks_fftlog=$pks_fftlog;")
        #println("true_pks_fftlog=$true_pks_fftlog;")

    end

    @testset "twofast" begin
        ks_twofast, pks_twofast = GaPSE.PS_multipole_GenWin(calc_name_file_ximultipoles, name_file_Qmultipoles;
            L=0, ps_kwargs(:twofast)...)

        true_ks_twofast, true_pks_twofast, _ = GaPSE.readxyall("datatest/PowerSpectrumGenWin/ps_GNC_L0_withF_noobsvel_GenWin_twofast.txt")

        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(ks_twofast, true_ks_twofast)])
        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(pks_twofast, true_pks_twofast)])

        #println("ks=$ks_twofast;")
        #println("pks_twofast=$pks_twofast;")
        #println("true_pks_twofast=$true_pks_twofast;")

    end

    rm(calc_name_file_ximultipoles)
end


@testset "test print_PS_multipole_GenWin" begin
    RTOL = 1e-2

    xi_filenames = "datatest/PowerSpectrumGenWin/" .* [
        "xis_GNC_L0_noF_noobsvel_GenWin.txt",
        "xis_GNC_L1_noF_noobsvel_GenWin.txt",
        "xis_GNC_L2_noF_noobsvel_GenWin.txt",
        "xis_GNC_L3_noF_noobsvel_GenWin.txt",
        "xis_GNC_L4_noF_noobsvel_GenWin.txt",
    ]

    calc_name_file_ximultipoles = "calc_xis_sum_GNC_L01234_noF_noobsvel.txt"
    name_file_Qmultipoles = "datatest/PowerSpectrumGenWin/Ql_multipoles_of_F_REFERENCE_z115.txt"
    isfile(calc_name_file_ximultipoles) && rm(calc_name_file_ximultipoles)
    GaPSE.create_file_for_XiMultipoles(calc_name_file_ximultipoles, xi_filenames, 2, "GNC")


    ps_kwargs(alg::Symbol=:fftlog) = alg == :twofast ? begin
        Dict(
            :alg => :twofast, :epl => false, :pr => false,
            :N_left => 12, :N_right => 12,
            :p0_left => [-2.0, 1.0], :p0_right => [-2.0, 1.0],
            :int_s_min => 1e0, :int_s_max => 1200.0,
            :cut_first_n => 0, :cut_last_n => 0
        )
    end : alg == :fftlog ? begin
        Dict(
            :alg => :fftlog, :pr => true, :ν => 1.5,
            :n_extrap_low => 0, :n_extrap_high => 0,
            :n_pad => 500, :cut_first_n => 0, :cut_last_n => 0,
        )
    end : throw(AssertionError("alg = :fftlog (recommended) or alg = :twofast !"))

    @testset "fftlog" begin
        calc_name_file_pks_fftlog = "calc_pks_genwin_fftlog.txt"
        isfile(calc_name_file_pks_fftlog) && rm(calc_name_file_pks_fftlog)
        GaPSE.print_PS_multipole_GenWin(calc_name_file_ximultipoles, name_file_Qmultipoles, calc_name_file_pks_fftlog;
            L=0, ps_kwargs(:fftlog)...)
        ks_fftlog, pks_fftlog = GaPSE.readxy(calc_name_file_pks_fftlog)

        true_ks_fftlog, true_pks_fftlog, _ = GaPSE.readxyall("datatest/PowerSpectrumGenWin/ps_GNC_L0_withF_noobsvel_GenWin_fftlog.txt")

        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(ks_fftlog, true_ks_fftlog)])
        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(pks_fftlog, true_pks_fftlog)])

        #println("ks=$ks_fftlog;")
        #println("pks_fftlog=$pks_fftlog;")
        #println("true_pks_fftlog=$true_pks_fftlog;")

        rm(calc_name_file_pks_fftlog)
    end

    @testset "twofast" begin
        calc_name_file_pks_twofast = "calc_pks_genwin_twofast.txt"
        isfile(calc_name_file_pks_twofast) && rm(calc_name_file_pks_twofast)
        GaPSE.print_PS_multipole_GenWin(calc_name_file_ximultipoles, name_file_Qmultipoles, calc_name_file_pks_twofast;
            L=0, ps_kwargs(:twofast)...)
        ks_twofast, pks_twofast = GaPSE.readxy(calc_name_file_pks_twofast)

        true_ks_twofast, true_pks_twofast, _ = GaPSE.readxyall("datatest/PowerSpectrumGenWin/ps_GNC_L0_withF_noobsvel_GenWin_twofast.txt")

        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(ks_twofast, true_ks_twofast)])
        @test all([isapprox(t, c; rtol=RTOL) for (t, c) in zip(pks_twofast, true_pks_twofast)])

        #println("ks=$ks_twofast;")
        #println("pks_twofast=$pks_twofast;")
        #println("true_pks_twofast=$true_pks_twofast;")
        rm(calc_name_file_pks_twofast)
    end

    rm(calc_name_file_ximultipoles)
end


