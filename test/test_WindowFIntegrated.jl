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

kwargs_map_F_int = Dict(
    :llim => 0.0, :rlim => Inf, :pr => true,
    :rtol => 5e-2, :atol => 0.0, :N => 300,
);

kwargs_F_int_quad = Dict(
    :llim => kwargs_map_F_int[:llim],
    :rlim => kwargs_map_F_int[:rlim],
    :rtol => kwargs_map_F_int[:rtol],
    :atol => kwargs_map_F_int[:atol],
)

kwargs_F_int_trap = Dict(
    :llim => kwargs_map_F_int[:llim],
    :rlim => kwargs_map_F_int[:rlim],
    :N => kwargs_map_F_int[:N],
)

calc_μs = vcat([-1.0, -0.98, -0.95],
    [μ for μ in -0.9:0.3:0.9],
    [0.95, 0.98, 1.0]);

windF = GaPSE.WindowF(FILE_F_MAP);

##########

@testset "test integrated_F_quadgk" begin
    RTOL = 1e-2
    s_min, s_max = 148.1920001465757, 571.7022420258767
    z_min, z_max = 0.05, 0.20

    @test isapprox(GaPSE.integrated_F_quadgk(100, 0, s_min, s_max, windF;
            kwargs_F_int_quad...), 2.12335e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(150, 0, s_min, s_max, windF;
            kwargs_F_int_quad...), 1.93936e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(200, 0, s_min, s_max, windF;
            kwargs_F_int_quad...), 1.68321e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(250, 0, s_min, s_max, windF;
            kwargs_F_int_quad...), 1.45684e+09; rtol=RTOL)

    @test isapprox(GaPSE.integrated_F_quadgk(100, 0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 1.42493e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(150, 0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 1.06046e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(200, 0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 7.53131e+08; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(250, 0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 4.82146e+08; rtol=RTOL)

    @test isapprox(GaPSE.integrated_F_quadgk(100, -0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 2.14873e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(150, -0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 2.08741e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(200, -0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 1.98355e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_quadgk(250, -0.8, s_min, s_max, windF;
            kwargs_F_int_quad...), 1.93041e+09; rtol=RTOL)
end

@testset "test integrated_F_trapz" begin
    RTOL = 1e-2
    s_min, s_max = 148.1920001465757, 571.7022420258767
    z_min, z_max = 0.05, 0.20

    @test isapprox(GaPSE.integrated_F_trapz(100, 0, s_min, s_max, windF;
            kwargs_F_int_trap...), 2.15022e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(150, 0, s_min, s_max, windF;
            kwargs_F_int_trap...), 1.93769e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(200, 0, s_min, s_max, windF;
            kwargs_F_int_trap...), 1.68718e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(250, 0, s_min, s_max, windF;
            kwargs_F_int_trap...), 1.41394e+09; rtol=RTOL)

    @test isapprox(GaPSE.integrated_F_trapz(100, 0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 1.44814e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(150, 0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 1.04627e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(200, 0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 7.29342e+08; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(250, 0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 4.84891e+08; rtol=RTOL)

    @test isapprox(GaPSE.integrated_F_trapz(100, -0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 2.18180e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(150, -0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 2.05673e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(200, -0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 1.93539e+09; rtol=RTOL)
    @test isapprox(GaPSE.integrated_F_trapz(250, -0.8, s_min, s_max, windF;
            kwargs_F_int_trap...), 1.90523e+09; rtol=RTOL)
end

##########

@testset "test print_map_IntegratedF with com dist" begin
    name_trap = "datatest/WindowFIntegrated/IntF_trap.txt"
    name_quad = "datatest/WindowFIntegrated/IntF_quad.txt"
    out_trap = "calc_IntF_trap.txt"
    out_quad = "calc_IntF_quad.txt"
    #z_min, z_max = 0.05, 0.20;
    s_min, s_max = 148.1920001465757, 571.7022420258767
    ref_ss = [s for s in 0.0:50.0:500.0]
    ref_μs = vcat([-1.0, -0.98, -0.95], [μ for μ in -0.9:0.3:0.9], [0.95, 0.98, 1.0])

    isfile(out_trap) && rm(out_trap)
    isfile(out_quad) && rm(out_quad)

    @testset "zeros" begin
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, FILE_F_MAP, out_trap; alg=:anything)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, FILE_F_MAP, out_trap; llim=-1.0)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, FILE_F_MAP, out_trap; rlim=0.0)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, FILE_F_MAP, out_trap; llim=1.0, rlim=0.5)

        @test_throws AssertionError GaPSE.print_map_IntegratedF(-1.0, 10.0, ref_ss, ref_μs, FILE_F_MAP, out_trap)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(1.0, 0.5, ref_ss, ref_μs, FILE_F_MAP, out_trap)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(1.0, 1.0, ref_ss, ref_μs, FILE_F_MAP, out_trap)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, [-1.0, 0.0, 50.0, 100.0], ref_μs, FILE_F_MAP, out_trap)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, [0.0, 50.0, 30.0, 100.0], ref_μs, FILE_F_MAP, out_trap)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, [-1.5, -1.0, 0.0, 0.5], FILE_F_MAP, out_trap)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, [-1.0, -0.5, 0.5, 0.0, 1.0], FILE_F_MAP, out_trap)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, FILE_F_MAP, "nonexistingdir/file.txt")
    end


    @testset "test trap 1" begin
        GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, FILE_F_MAP, out_trap;
            alg=:trap, kwargs_map_F_int...)

        table_output_F = readdlm(out_trap, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_trap, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([s1 ≈ s2 for (s1, s2) in zip(ss, output_ss)])
        @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
        @test all([IF1 ≈ IF2 for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_trap)
    end

    @testset "test trap 2" begin
        wf = GaPSE.WindowF(FILE_F_MAP)
        GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, wf, out_trap;
            alg=:trap, kwargs_map_F_int...)

        table_output_F = readdlm(out_trap, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_trap, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([s1 ≈ s2 for (s1, s2) in zip(ss, output_ss)])
        @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
        @test all([IF1 ≈ IF2 for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_trap)
    end

    @testset "test quad 1" begin
        GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, FILE_F_MAP, out_quad;
            alg=:quad, kwargs_map_F_int...)

        table_output_F = readdlm(out_quad, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_quad, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([s1 ≈ s2 for (s1, s2) in zip(ss, output_ss)])
        @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
        @test all([IF1 ≈ IF2 for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_quad)
    end

    @testset "test quad 2" begin
        wf = GaPSE.WindowF(FILE_F_MAP)
        GaPSE.print_map_IntegratedF(s_min, s_max, ref_ss, ref_μs, wf, out_quad;
            alg=:quad, kwargs_map_F_int...)

        table_output_F = readdlm(out_quad, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_quad, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([s1 ≈ s2 for (s1, s2) in zip(ss, output_ss)])
        @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
        @test all([IF1 ≈ IF2 for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_quad)
    end
end

@testset "test print_map_IntegratedF with redshifts" begin
    name_trap = "datatest/WindowFIntegrated/IntF_trap.txt"
    name_quad = "datatest/WindowFIntegrated/IntF_quad.txt"
    out_trap = "calc_IntF_trap.txt"
    out_quad = "calc_IntF_quad.txt"
    # The following vector of redshifts are the values corresponding for the ""future""
    # Cosmology to the following comoving distances
    #s_min, s_max = 148.1920001465757, 571.7022420258767
    # ref_ss = [s for s in 100.0:50.0:500.0]
    z_min, z_max = 0.05, 0.20
    ref_zs = [0.0, 0.01674166576924665,
        0.03361259170114134, 0.050617270275786205, 0.06776014179168262,
        0.08504575380657542, 0.10247876620072593, 0.12006395442156174,
        0.1378062128024508, 0.15571055793431468, 0.17378213220269997]
    ATOL, RTOL = 1e-12, 1e-4

    ref_μs = vcat([-1.0, -0.98, -0.95], [μ for μ in -0.9:0.3:0.9], [0.95, 0.98, 1.0])

    isfile(out_trap) && rm(out_trap)
    isfile(out_quad) && rm(out_quad)

    @testset "zeros" begin
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; alg=:anything)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; llim=-1.0)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; rlim=0.0)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; llim=1.0, rlim=0.5)

        @test_throws AssertionError GaPSE.print_map_IntegratedF(-1.0, 1.0, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(1.0, 0.5, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(1.0, 1.0, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, [-1.0, 0.0, 0.1, 0.2], ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, [0.0, 0.2, 0.1, 0.3], ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, [-1.5, -1.0, 0.0, 0.5], FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, [-1.0, -0.5, 0.5, 0.0, 1.0], FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, FILE_F_MAP, "nonexistingdir/file.txt", FILE_BACKGROUND)
    end


    @testset "test trap 1" begin
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND;
            alg=:trap, kwargs_map_F_int...)

        table_output_F = readdlm(out_trap, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_trap, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_trap)
    end

    @testset "test trap 2" begin
        wf = GaPSE.WindowF(FILE_F_MAP)
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, wf, out_trap, FILE_BACKGROUND;
            alg=:trap, kwargs_map_F_int...)

        table_output_F = readdlm(out_trap, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_trap, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_trap)
    end

    @testset "test quad 1" begin
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, FILE_F_MAP, out_quad, FILE_BACKGROUND;
            alg=:quad, kwargs_map_F_int...)

        table_output_F = readdlm(out_quad, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_quad, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_quad)
    end

    @testset "test quad 2" begin
        wf = GaPSE.WindowF(FILE_F_MAP)
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_zs, ref_μs, wf, out_quad, FILE_BACKGROUND;
            alg=:quad, kwargs_map_F_int...)

        table_output_F = readdlm(out_quad, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_quad, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_quad)
    end
end

@testset "test print_map_IntegratedF with automatic com dist" begin
    name_trap = "datatest/WindowFIntegrated/IntF_trap_another.txt"
    name_quad = "datatest/WindowFIntegrated/IntF_quad_another.txt"
    out_trap = "calc_IntF_trap.txt"
    out_quad = "calc_IntF_quad.txt"
    # The following vector of redshifts are the values corresponding for the ""future""
    # Cosmology to the following comoving distances
    #s_min, s_max = 148.1920001465757, 571.7022420258767
    # ref_ss = [s for s in 100.0:50.0:500.0]
    ATOL, RTOL = 1e-12, 1e-4
    z_min, z_max = 1.0, 1.5
    ref_μs = vcat([-1.0, -0.98, -0.95], [μ for μ in -0.9:0.3:0.9], [0.95, 0.98, 1.0])

    isfile(out_trap) && rm(out_trap)
    isfile(out_quad) && rm(out_quad)

    @testset "zeros" begin
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; alg=:anything)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; llim=-1.0)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; rlim=0.0)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; llim=1.0, rlim=0.5)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; N_ss=3)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; m=0.0)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND; m=20.0)

        @test_throws AssertionError GaPSE.print_map_IntegratedF(-1.0, 1.0, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(1.0, 0.5, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(1.0, 1.0, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, [-1.5, -1.0, 0.0, 0.5], FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, [-1.0, -0.5, 0.5, 0.0, 1.0], FILE_F_MAP, out_trap, FILE_BACKGROUND)
        @test_throws AssertionError GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, "nonexistingdir/file.txt", FILE_BACKGROUND)
    end


    @testset "test trap 1" begin
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_trap, FILE_BACKGROUND;
            alg=:trap, N_ss=20, m=2.1, kwargs_map_F_int...)

        table_output_F = readdlm(out_trap, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_trap, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        #println("ss = $ss;")
        #println("μs = $μs;")
        #println("IFs = $IFs;")

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_trap)
    end

    @testset "test trap 2" begin
        wf = GaPSE.WindowF(FILE_F_MAP)
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, wf, out_trap, FILE_BACKGROUND;
            alg=:trap, N_ss=20, m=2.1, kwargs_map_F_int...)

        table_output_F = readdlm(out_trap, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_trap, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_trap)
    end

    @testset "test quad 1" begin
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, FILE_F_MAP, out_quad, FILE_BACKGROUND;
            alg=:quad, N_ss=20, m=2.1, kwargs_map_F_int...)

        table_output_F = readdlm(out_quad, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_quad, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_quad)
    end

    @testset "test quad 2" begin
        wf = GaPSE.WindowF(FILE_F_MAP)
        GaPSE.print_map_IntegratedF(z_min, z_max, ref_μs, wf, out_quad, FILE_BACKGROUND;
            alg=:quad, N_ss=20, m=2.1, kwargs_map_F_int...)

        table_output_F = readdlm(out_quad, comments=true)
        output_ss = convert(Vector{Float64}, table_output_F[:, 1])
        output_μs = convert(Vector{Float64}, table_output_F[:, 2])
        output_IFs = convert(Vector{Float64}, table_output_F[:, 3])

        table_F = readdlm(name_quad, comments=true)
        ss = convert(Vector{Float64}, table_F[:, 1])
        μs = convert(Vector{Float64}, table_F[:, 2])
        IFs = convert(Vector{Float64}, table_F[:, 3])

        @test all([isapprox(s1, s2; atol=ATOL, rtol=RTOL) for (s1, s2) in zip(ss, output_ss)])
        @test all([isapprox(μ1, μ2; atol=ATOL, rtol=RTOL) for (μ1, μ2) in zip(μs, output_μs)])
        @test all([isapprox(IF1, IF2; atol=ATOL, rtol=RTOL) for (IF1, IF2) in zip(IFs, output_IFs)])

        rm(out_quad)
    end
end

##########


@testset "test WindowFIntegrated: first convection" begin
    ss = 100 .* [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] .+ 30
    μs = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
    IFs = [0, 1, 2, 0, 2, 4, 0, 4, 8, 0, 8, 16]

    unique_ss = 100 .* [0, 1, 2, 3] .+ 30
    unique_μs = [-1, 0, 1]
    table_IFs = [0 1 2; 0 2 4; 0 4 8; 0 8 16]

    name = "test_WindowFIntegrated_fc.txt"
    isfile(name) && rm(name)
    open(name, "w") do io
        println(io, "# line of comment")
        println(io, "# another one")
        for (s, μ, IF) in zip(ss, μs, IFs)
            println(io, "$s \t $μ \t $IF")
        end
    end

    F_fc = GaPSE.WindowFIntegrated(name)

    @test size(F_fc.ss) == size(unique_ss)
    @test size(F_fc.μs) == size(unique_μs)
    @test size(F_fc.IFs) == size(table_IFs)
    @test all(F_fc.ss .== unique_ss)
    @test all(F_fc.μs .== unique_μs)
    @test all(F_fc.IFs .== table_IFs)

    rm(name)
end

@testset "test WindowFIntegrated: second convection" begin
    ss = 100 .* [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] .+ 30
    μs = [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    IFs = [0, 0, 0, 0, 1, 2, 4, 8, 2, 4, 8, 16]

    unique_ss = 100 .* [0, 1, 2, 3] .+ 30
    unique_μs = [-1, 0, 1]
    table_IFs = [0 1 2; 0 2 4; 0 4 8; 0 8 16]

    name = "test_WindowFIntegrated_sc.txt"
    isfile(name) && rm(name)
    open(name, "w") do io
        println(io, "# line of comment")
        println(io, "# another one")
        for (s, μ, IF) in zip(ss, μs, IFs)
            println(io, "$s \t $μ \t $IF")
        end
    end

    F_sc = GaPSE.WindowFIntegrated(name)

    @test size(F_sc.ss) == size(unique_ss)
    @test size(F_sc.μs) == size(unique_μs)
    @test size(F_sc.IFs) == size(table_IFs)
    @test all(F_sc.ss .== unique_ss)
    @test all(F_sc.μs .== unique_μs)
    @test all(F_sc.IFs .== table_IFs)

    rm(name)
end


##########

@testset "test spline windowFintegrated" begin
    wfi_trap = GaPSE.WindowFIntegrated("datatest/WindowFIntegrated/IntF_trap.txt")
    spline_trap(s, μ) = GaPSE.spline_integrF(s, μ, wfi_trap)

    @test isapprox(spline_trap(375.0, -0.8), 1.599944823071156e9; rtol=1e-5)
    @test isapprox(spline_trap(499.0, -0.8), 1.2760483651794796e9; rtol=1e-5)
    @test isapprox(spline_trap(375.0, 0.1), 6.21811893461226e8; rtol=1e-5)
    @test isapprox(spline_trap(499.0, 0.1), 1.0480317459427744e8; rtol=1e-5)
end

@testset "test second method print_map_integrated" begin
    name_trap = "datatest/WindowFIntegrated/IntF_trap.txt"
    orig_wfi = GaPSE.WindowFIntegrated(name_trap)

    print_trap = "test_print_wfi.txt"
    isfile(print_trap) && rm(print_trap)
    GaPSE.print_map_IntegratedF(print_trap, orig_wfi)
    other_wfi = GaPSE.WindowFIntegrated(print_trap)

    @test all([s1 ≈ s2 for (s1, s2) in zip(orig_wfi.ss, other_wfi.ss)])
    @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(orig_wfi.μs, other_wfi.μs)])
    @test all([IF1 ≈ IF2 for (IF1, IF2) in zip(orig_wfi.IFs, other_wfi.IFs)])
    rm(print_trap)
end

