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



@testset "test constant values" begin
    @test isapprox(GaPSE.ℋ0, HUBBLE_0; rtol = 1e-6)
end

@testset "test BackgroundData" begin
    BD = GaPSE.BackgroundData(FILE_BACKGROUND, 1000.0)
    L = length(CONF_TIME)

    @test all(isapprox(BD.z[begin:L], ZS, rtol = 1e-8))
    @test all(isapprox.(BD.conftime[begin:L], CONF_TIME, rtol = 1e-8))
    @test all(isapprox.(BD.comdist[begin:L], COM_DIST, rtol = 1e-8))
    @test all(isapprox.(BD.angdist[begin:L], ANG_DIST, rtol = 1e-8))
    @test all(isapprox.(BD.lumdist[begin:L], LUM_DIST, rtol = 1e-8))
    @test all(isapprox.(BD.D[begin:L], GROWTH_FACTOR_D, rtol = 1e-8))
    @test all(isapprox.(BD.f[begin:L], GROWTH_FACTOR_F, rtol = 1e-8))
    @test all(isapprox.(BD.ℋ[begin:L], COM_H, rtol = 1e-8))
    @test all(isapprox.(BD.ℋ_p[3:L], COM_H_P[3:end], rtol = 1e-3))
end
