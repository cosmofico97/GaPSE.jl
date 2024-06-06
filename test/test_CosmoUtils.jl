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

@testset "test func_z_eff - Dierckx Spline" begin
    z_of_s = Dierckx.Spline1D(COM_DIST, ZS)
    @test isapprox(GaPSE.func_z_eff(S_MAX, S_MIN, z_of_s), Z_EFF; rtol=1e-6)
end

@testset "test func_z_eff - MySpline" begin
    z_of_s = GaPSE.MySpline(COM_DIST, ZS; ic="ThirdDerivative")
    @test isapprox(GaPSE.func_z_eff(S_MAX, S_MIN, z_of_s), Z_EFF; rtol=1e-6)
end


@testset "test s-mu-s2-y" begin
    RTOL = 1e-6

    @test isapprox(GaPSE.s(1.2, 3.5, 0.73), 2.749181696432595; rtol=RTOL)
    @test isapprox(GaPSE.s(3.5, 1.2, 0.73), 2.749181696432595; rtol=RTOL)

    @test isapprox(GaPSE.μ(1.2, 3.5, 0.73), 0.49287393472693375; rtol=RTOL)
    @test isapprox(GaPSE.μ(3.5, 1.2, 0.73), -0.9544658337442616; rtol=RTOL)

    @test isapprox(GaPSE.s2(1.2, 3.5, 0.73), 4.452190472115944; rtol=RTOL)
    @test isapprox(GaPSE.s2(3.5, 1.2, 0.73), 4.452190472115944; rtol=RTOL)

    @test isapprox(GaPSE.y(1.2, 3.5, 0.73), 0.8434050662292086; rtol=RTOL)
    @test isapprox(GaPSE.y(3.5, 1.2, 0.73), 0.9828869693259701; rtol=RTOL)
end

@testset "test ϕ" begin
    @test GaPSE.ϕ(0, 1.0, 2.0) ≈ 0.0
    @test GaPSE.ϕ(0.5, 1.0, 2.0) ≈ 0.0
    @test GaPSE.ϕ(1.0, 1.0, 2.0) ≈ 0.0
    @test GaPSE.ϕ(1.0 + 1e-5, 1.0, 2.0) ≈ 1.0
    @test GaPSE.ϕ(1.5, 1.0, 2.0) ≈ 1.0
    @test GaPSE.ϕ(2.0 - 1e-5, 1.0, 2.0) ≈ 1.0
    @test GaPSE.ϕ(2.0, 1.0, 2.0) ≈ 0.0
    @test GaPSE.ϕ(2.5, 1.0, 2.0) ≈ 0.0
end

@testset "test W" begin
    @test GaPSE.W(0, π / 3) ≈ 1.0
    @test GaPSE.W(1.0, π / 3) ≈ 1.0
    @test GaPSE.W(π / 3, π / 3) ≈ 0.0
    @test GaPSE.W(π, π / 3) ≈ 0.0
end

@testset "test A" begin
    RTOL = 1e-4
    @test isapprox(GaPSE.A(150, 500, π / 2), 6.452406651183921e6, rtol=RTOL)
    @test isapprox(GaPSE.A(1, 2, π / 2), 0.371361533881089, rtol=RTOL)

    @test isapprox(GaPSE.A(1, 2, 0.0), 0.0, rtol=RTOL)
    @test isapprox(GaPSE.A(1, 2, π / 6), 0.049753011551710406, rtol=RTOL)

    @test isapprox(GaPSE.A(S_MIN, S_MAX, π / 2), 9.740426295429418e6, rtol=RTOL)
end


@testset "corresponding_redshift" begin
    RTOL = 1e-5

    @test_throws AssertionError GaPSE.corresponding_redshift(-1.0, 2, FILE_BACKGROUND)
    @test_throws AssertionError GaPSE.corresponding_redshift(1.0, 0.0, FILE_BACKGROUND)

    @test isapprox(GaPSE.corresponding_redshift(0.1, 2, FILE_BACKGROUND), 0.20521035318209763; rtol=RTOL)
    @test isapprox(GaPSE.corresponding_redshift(0.1, 1, FILE_BACKGROUND), 0.09999999998047998; rtol=RTOL)
    @test isapprox(GaPSE.corresponding_redshift(2.0, 2, FILE_BACKGROUND), 16.015246541511207; rtol=RTOL)
end
