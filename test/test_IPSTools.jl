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


@testset "test InputPS" begin
    tab_pk = readdlm(FILE_PS, comments = true)
    ks = convert(Vector{Float64}, tab_pk[:, 1])
    k_min, k_max = ks[begin], ks[end]
    pks = convert(Vector{Float64}, tab_pk[:, 2])
    #PK = Dierckx.Spline1D(ks, pks)

    @testset "first" begin
        ips = GaPSE.InputPS(FILE_PS; 
            fit_left_min = 1e-6, fit_left_max = 3e-6,
            fit_right_min = 1e1, fit_right_max = 2e1)
        
        @test ips.l_si ≈ 0.9599998007678217
        @test ips.l_b ≈ 3.0123089778963984e6
        @test ips.l_a ≈ 0.0
        @test ips.left ≈ 1e-6

        @test ips.r_si ≈ -2.6531284143857454
        @test ips.r_b ≈ 68.30720789490483
        @test ips.r_a ≈ 0.0
        @test ips.right ≈ 2e1

        @test all([isapprox(ips(k), pk, rtol=1e-3)  for (k, pk) in 
            zip(ks[ks .> 0.7e-7], pks[ks .> 0.7e-7])])
    end

    @testset "second" begin
        ips = GaPSE.InputPS(ks, pks; 
            fit_left_min = 1e-6, fit_left_max = 3e-6,
            fit_right_min = 1e1, fit_right_max = 2e1)

        @test ips.l_si ≈ 0.9599998007678217
        @test ips.l_b ≈ 3.0123089778963984e6
        @test ips.l_a ≈ 0.0
        @test ips.left ≈ 1e-6

        @test ips.r_si ≈ -2.6531284143857454
        @test ips.r_b ≈ 68.30720789490483
        @test ips.r_a ≈ 0.0
        @test ips.right ≈ 2e1

        @test all([isapprox(ips(k), pk, rtol=1e-3)  for (k, pk) in 
            zip(ks[ks .> 0.7e-7], pks[ks .> 0.7e-7])])
    end
end


@testset "test IPSTools" begin
    ips = GaPSE.InputPS(FILE_PS; )
    k_min, k_max = 1e-6, 10.0

    tools = GaPSE.IPSTools(ips; k_min = k_min, k_max = k_max, N = 1024,
        fit_min = 0.05, fit_max = 0.5, con = true)
    PK = ips

    tab_Is = readdlm(FILE_ILN, comments = true)
    ss = convert(Vector{Float64}, tab_Is[:, 1])
    I00s = convert(Vector{Float64}, tab_Is[:, 2])
    I20s = convert(Vector{Float64}, tab_Is[:, 3])
    I40s = convert(Vector{Float64}, tab_Is[:, 4])
    I02s = convert(Vector{Float64}, tab_Is[:, 5])
    I22s = convert(Vector{Float64}, tab_Is[:, 6])
    I31s = convert(Vector{Float64}, tab_Is[:, 7])
    I11s = convert(Vector{Float64}, tab_Is[:, 8])
    I13s = convert(Vector{Float64}, tab_Is[:, 9])
    I04_tildes = convert(Vector{Float64}, tab_Is[:, 10])
    #=
    true_ss = convert(Vector{Float64}, tab_Is[2:end, 1])
    ss = true_ss[1e-3 .< true_ss .< 1e3]
    I00s = convert(Vector{Float64}, tab_Is[2:end, 2])[1e-3 .< true_ss .< 1e3]
    I20s = convert(Vector{Float64}, tab_Is[2:end, 3])[1e-3 .< true_ss .< 1e3]
    I40s = convert(Vector{Float64}, tab_Is[2:end, 4])[1e-3 .< true_ss .< 1e3]
    I02s = convert(Vector{Float64}, tab_Is[2:end, 5])[1e-3 .< true_ss .< 1e3] ./ ss .^ 2
    I22s = convert(Vector{Float64}, tab_Is[2:end, 6])[1e-3 .< true_ss .< 1e3] ./ ss .^ 2
    I31s = convert(Vector{Float64}, tab_Is[2:end, 7])[1e-3 .< true_ss .< 1e3] ./ ss
    I11s = convert(Vector{Float64}, tab_Is[2:end, 8])[1e-3 .< true_ss .< 1e3] ./ ss
    =#


    @testset "test sigmas" begin
        σ_0, _ = quadgk(q -> PK(q) * q^2 / (2 * π^2), k_min, k_max)
        σ_1, _ = quadgk(q -> PK(q) * q / (2 * π^2), k_min, k_max)
        σ_2, _ = quadgk(q -> PK(q) / (2 * π^2), k_min, k_max)
        σ_3, _ = quadgk(q -> PK(q) / (2 * π^2 * q), k_min, k_max)

        @test isapprox(tools.σ_0, σ_0, rtol=1e-4) && isapprox(σ_0, 15.593462966741178, rtol=1e-4)
        @test isapprox(tools.σ_1, σ_1, rtol=1e-4) && isapprox(σ_1, 15.074895881392285, rtol=1e-4)
        @test isapprox(tools.σ_2, σ_2, rtol=1e-4) && isapprox(σ_2, 100.85852368830221, rtol=1e-4)
        @test isapprox(tools.σ_3, σ_3, rtol=1e-4) && isapprox(σ_3, 3734.849690975012, rtol=1e-4)
    end

    RTOL = 5e-4

    @testset "test I00" begin
        @test isapprox(tools.I00.l_si , -0.3480008484088528  ; rtol=RTOL)
        @test isapprox(tools.I00.l_b , 11.575473646894222  ; rtol=RTOL)
        @test isapprox(tools.I00.l_a ,  -6.492685627424222  ; rtol=RTOL)
        @test isapprox(tools.I00.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I00.r_si , 2.8758516589192706  ; rtol=RTOL)
        @test isapprox(tools.I00.r_b , 8.345990355535678e-29  ; rtol=RTOL)
        @test isapprox(tools.I00.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I00.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I00(s), i, rtol = 1e-3) for (s, i) in zip(ss, I00s)])
    end

    @testset "test I20" begin
        @test isapprox(tools.I20.l_si , -0.3434449376790657  ; rtol=RTOL)
        @test isapprox(tools.I20.l_b , 1.5505826777939757  ; rtol=RTOL)
        @test isapprox(tools.I20.l_a ,  -0.035265713349630266  ; rtol=RTOL)
        #                               -0.03525587182760787
        @test isapprox(tools.I20.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I20.r_si , -3.9578746738388397  ; rtol=RTOL)
        @test isapprox(tools.I20.r_b , 1.1832931457198192e6  ; rtol=RTOL)
        @test isapprox(tools.I20.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I20.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I20(s), i, rtol = 1e-3) for (s, i) in zip(ss, I20s)])
    end

    @testset "test I40" begin
        @test isapprox(tools.I40.l_si , -0.3468022218825587  ; rtol=RTOL)
        @test isapprox(tools.I40.l_b , 0.7681399976144987  ; rtol=RTOL)
        #@test isapprox(tools.I40.l_a ,  -0.00028247463623223634  ; rtol=RTOL)
        @test isapprox(tools.I40.l_a, -0.0002803788459337459; rtol=RTOL)
        #                               -0.0002803788459337459
        @test isapprox(tools.I40.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I40.r_si , -3.9597415622888192  ; rtol=RTOL)
        @test isapprox(tools.I40.r_b , 6.92151268083521e6  ; rtol=RTOL)
        @test isapprox(tools.I40.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I40.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I40(s), i, rtol = 1e-3) for (s, i) in zip(ss, I40s)])
    end

    @testset "test I02" begin
        @test isapprox(tools.I02.l_si , -2.00164755613052  ; rtol=RTOL)
        @test isapprox(tools.I02.l_b , 100.44877199997933  ; rtol=RTOL)
        @test isapprox(tools.I02.l_a ,  0.0  ; rtol=RTOL)
        @test isapprox(tools.I02.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I02.r_si , -5.866385247535877  ; rtol=RTOL)
        @test isapprox(tools.I02.r_b , 1.7556551321550488e14  ; rtol=RTOL)
        @test isapprox(tools.I02.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I02.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I02(s), i, rtol = 1e-3) for (s, i) in zip(ss, I02s)])
    end

    @testset "test I22" begin
        @test isapprox(tools.I22.l_si , -0.34727171081772734  ; rtol=RTOL)
        @test isapprox(tools.I22.l_b , 0.9412777254767503  ; rtol=RTOL)
        @test isapprox(tools.I22.l_a , -0.43616927284862367  ; rtol=RTOL)
        @test isapprox(tools.I22.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I22.r_si , -3.431171897505612  ; rtol=RTOL)
        @test isapprox(tools.I22.r_b , 851.4607831995294  ; rtol=RTOL)
        @test isapprox(tools.I22.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I22.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I22(s), i, rtol = 1e-3) for (s, i) in zip(ss, I22s)])
    end

    @testset "test I31" begin
        @test isapprox(tools.I31.l_si ,  -0.3445696461864395  ; rtol=RTOL)
        @test isapprox(tools.I31.l_b , 0.33123234468023927  ; rtol=RTOL)
        @test isapprox(tools.I31.l_a , -0.005064156263986402  ; rtol=RTOL)
        #                              -0.005062453703102121
        @test isapprox(tools.I31.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I31.r_si , -3.9594640605240348  ; rtol=RTOL)
        @test isapprox(tools.I31.r_b , 1.157797082289167e6  ; rtol=RTOL)
        @test isapprox(tools.I31.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I31.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I31(s), i, rtol = 1e-3) for (s, i) in zip(ss, I31s)])
    end

    @testset "test I11" begin
        @test isapprox(tools.I11.l_si , -0.3474733407904323  ; rtol=RTOL)
        @test isapprox(tools.I11.l_b , 4.375198496682168  ; rtol=RTOL)
        @test isapprox(tools.I11.l_a , -2.17582565657694  ; rtol=RTOL)
        @test isapprox(tools.I11.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I11.r_si , -2.270802848837033  ; rtol=RTOL)
        @test isapprox(tools.I11.r_b , 0.002613631263376589  ; rtol=RTOL)
        @test isapprox(tools.I11.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I11.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I11(s), i, rtol = 1e-3) for (s, i) in zip(ss, I11s)])
    end

    @testset "test I13" begin
        @test isapprox(tools.I13.l_si ,  -2.0001163229203924  ; rtol=RTOL)
        @test isapprox(tools.I13.l_b , 33.62113907833843  ; rtol=RTOL)
        @test isapprox(tools.I13.l_a , -0.5498036524608388  ; rtol=RTOL)
        @test isapprox(tools.I13.left , 0.05  ; rtol=RTOL)

        @test isapprox(tools.I13.r_si , -4.387393755963048  ; rtol=RTOL)
        @test isapprox(tools.I13.r_b , 1.5388770882263558e7  ; rtol=RTOL)
        @test isapprox(tools.I13.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I13.right , 96466.16199111959  ; rtol=RTOL)

        @test all([isapprox(tools.I13(s), i, rtol = 1e-3) for (s, i) in zip(ss, I13s)])
    end

    @testset "test I04_tilde" begin
        @test isapprox(tools.I04_tilde.l_si , -2.0002445476746975  ; rtol=RTOL)
        @test isapprox(tools.I04_tilde.l_b , -16.80636730509123  ; rtol=RTOL)
        @test isapprox(tools.I04_tilde.l_a , 0.10615797439188279  ; rtol=RTOL)
        @test isapprox(tools.I04_tilde.left , 0.1  ; rtol=RTOL)

        @test isapprox(tools.I04_tilde.r_si , -3.746166614932489  ; rtol=RTOL)
        @test isapprox(tools.I04_tilde.r_b , -83772.05493095529  ; rtol=RTOL)
        @test isapprox(tools.I04_tilde.r_a , 0.0  ; rtol=RTOL)
        @test isapprox(tools.I04_tilde.right , 9888.080416307215  ; rtol=RTOL)

        @test all([isapprox(tools.I04_tilde(s), i, rtol = 1e-3) for (s, i) in zip(ss, I04_tildes)])
    end
end



