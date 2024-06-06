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

@testset "derivate_point" begin
    @testset "first" begin
        f(x) = 1.35 + 2.54 * x
        xp, x1, x2 = 2.0, 1.0, 3.0
        yp, y1, y2 = f(xp), f(x1), f(x2)

        @test GaPSE.derivate_point(xp, yp, x1, y1, x2, y2) ≈ 2.54
    end

    @testset "second" begin
        f(x) = 2.634 + 4.65 * x^3.0
        xp, x1, x2 = 3.0, 3.0 - 1e-8, 3.0 + 1e-8
        yp, y1, y2 = f(xp), f(x1), f(x2)
        @test isapprox(GaPSE.derivate_point(xp, yp, x1, y1, x2, y2), 125.55, atol=1e-4)
    end
end



@testset "test derivate_vector" begin
    RTOL = 1e-2

    @testset "zeros" begin
        xs = 1:0.1:2
        ys = [2.5 + 3.65 * x^(-3.5) for x in xs]
        @test_throws AssertionError GaPSE.derivate_vector(xs, ys[begin+1:end])
        @test_throws AssertionError GaPSE.derivate_vector(xs[begin+1:end], ys)
        @test_throws AssertionError GaPSE.derivate_vector(xs, ys; N=6)
    end

    @testset "first" begin
        xs = 1:0.1:2
        ys = [2.5 + 3.65 * x for x in xs]
        calc = GaPSE.derivate_vector(xs, ys; N=1)
        @test all([isapprox(3.65, c; rtol=1e-4) for c in calc])
    end

    @testset "second" begin
        xs = 1:0.01:2
        ys = [2.5 + 3.65 * x^2.0 for x in xs]
        calc = GaPSE.derivate_vector(xs, ys)
        @test all([isapprox(1.0, 3.65 * 2.0 * x / c; rtol=1e-4)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "third" begin
        xs = 1:0.01:2
        ys = [2.5 + 3.65 * x^3.5 for x in xs]
        calc = GaPSE.derivate_vector(xs, ys)
        @test all([isapprox(1.0, 3.65 * 3.5 * x^2.5 / c; rtol=1e-4)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "fourth" begin
        xs = 1:0.01:2
        ys = [2.5 + 3.65 * x^(-3.5) for x in xs]
        calc = GaPSE.derivate_vector(xs, ys)
        @test all([isapprox(1.0, -3.65 * 3.5 * x^(-4.5) / c; rtol=RTOL)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "fifth" begin
        xs = 10 .^ range(4, 6, length=100)
        ys = [2.5e99 + 3.65 * x^(3.5) for x in xs]
        @test_throws AssertionError GaPSE.derivate_vector(xs, ys; N=1)
        @test_throws AssertionError GaPSE.derivate_vector(xs, ys; N=6)
    end

    @testset "sixth" begin
        xs = 10 .^ range(4, 6, length=100)
        ys = [2.5e23 + 3.65 * x^(3.5) for x in xs]
        calc = GaPSE.derivate_vector(xs, ys)
        @test all([isapprox(1.0, 3.65 * 3.5 * x^(2.5) / c; rtol=RTOL)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "seventh" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [3.6e-22 + 3.65 * x^(3.5) for x in xs]
        calc = GaPSE.derivate_vector(xs, ys)
        @test all([isapprox(1.0, 3.65 * 3.5 * x^(2.5) / c; rtol=RTOL)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "eigth" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [3.6e-22 + 3.65 * x^(-3.5) for x in xs]
        calc = GaPSE.derivate_vector(xs, ys)
        @test all([isapprox(1.0, -3.65 * 3.5 * x^(-4.5) / c; rtol=RTOL)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end
end



@testset "test spectral_index" begin
    RTOL = 1e-2

    @testset "zeros" begin
        xs = 1:0.1:2
        ys = [2.5 + 3.65 * x^(-3.5) for x in xs]
        @test_throws AssertionError GaPSE.spectral_index(xs, ys[begin+1:end])
        @test_throws AssertionError GaPSE.spectral_index(xs[begin+1:end], ys)
        @test_throws AssertionError GaPSE.spectral_index(xs, ys; N=6)
    end


    @testset "first" begin
        xs = 10:0.1:20
        ys = [3.65 * x^(-1.5) for x in xs]
        calc = GaPSE.spectral_index(xs, ys; N=1, con=false)
        @test all([isapprox(-1.5, c; rtol=1e-3) for c in calc])
    end

    @testset "second" begin
        xs = 10:0.1:20
        ys = [1434.5 + 3.65 * x^(-1.5) for x in xs]
        calc = GaPSE.spectral_index(xs, ys; N=1, con=true)
        @test all([isapprox(-1.5, c; rtol=1e-3) for c in calc])
    end

    @testset "third" begin
        xs = 10 .^ range(4, 6, length=100)
        ys = [2.5e23 + 3.65 * x^(3.5) for x in xs]
        calc = GaPSE.spectral_index(xs, ys; N=1, con=true)
        @test all([isapprox(3.5, c; rtol=1e-3) for c in calc])
    end

    @testset "fourth" begin
        xs = 10 .^ range(4, 6, length=100)
        ys = [2.5e99 + 3.65 * x^(3.5) for x in xs]
        @test_throws AssertionError GaPSE.spectral_index(xs, ys; N=1, con=true)
        @test_throws AssertionError GaPSE.spectral_index(xs, ys; N=1, con=false)
    end

    @testset "fifth" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [3.65 * x^(3.5) for x in xs]
        calc = GaPSE.spectral_index(xs, ys; N=1, con=false)
        @test all([isapprox(1.0, 3.5 / c; rtol=RTOL)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "sixth" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [3.6e-22 + 3.65 * x^(3.5) for x in xs]
        calc = GaPSE.spectral_index(xs, ys; N=1, con=true)
        @test all([isapprox(1.0, 3.5 / c; rtol=RTOL)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "seventh" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [3.65 * x^(-3.5) for x in xs]
        calc = GaPSE.spectral_index(xs, ys; N=1, con=false)
        @test all([isapprox(1.0, -3.5 / c; rtol=RTOL)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end

    @testset "eigth" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [3.6e-22 + 3.65 * x^(-3.5) for x in xs]
        calc = GaPSE.spectral_index(xs, ys; N=1, con=true)
        @test all([isapprox(1.0, -3.5 / c; rtol=3e-2)
                    for (x, c) in zip(xs[begin+1:end-1], calc[begin+1:end-1])])
    end
end



@testset "mean_spectral_index" begin
    @testset "first" begin
        xs = range(1.0, 10.0, 100)
        ys = [3.54 * x^1.856 for x in xs]

        @test isapprox(GaPSE.mean_spectral_index(xs, ys; con=true), 1.856; atol=1e-4)
        @test isapprox(GaPSE.mean_spectral_index(xs, ys; con=false), 1.856; atol=1e-4)
    end

    @testset "second" begin
        xs = range(1.0, 10.0, 100)
        ys = [256.75 - 3.54 * x^1.856 for x in xs]

        @test isapprox(GaPSE.mean_spectral_index(xs, ys; con=true), 1.856; atol=1e-4)
    end

    @testset "third" begin
        xs = 10 .^ range(4, 6, length=100)
        ys = [3.54e6 * x^4.856 for x in xs]

        @test isapprox(GaPSE.mean_spectral_index(xs, ys; con=false), 4.856; atol=3e-2)
    end

    @testset "fourth" begin
        xs = 10 .^ range(4, 6, length=100)
        ys = [8.3e30 - 3.54e6 * x^4.856 for x in xs]

        @test isapprox(GaPSE.mean_spectral_index(xs, ys; con=true) / 4.856, 1.0; atol=3e-2)
    end

    @testset "fifth" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [3.54e6 * x^(-4.856) for x in xs]

        @test isapprox(GaPSE.mean_spectral_index(xs, ys; con=false) / (-4.856), 1.0; atol=3e-2)
    end

    @testset "sixth" begin
        xs = 10 .^ range(-6, -4, length=100)
        ys = [8.3e-27 - 3.54e-6 * x^(-4.856) for x in xs]

        @test isapprox(GaPSE.mean_spectral_index(xs, ys; con=true) / (-4.856), 1.0; atol=3e-2)
    end

end



##########################################################################################92



@testset "test power_law" begin
    x, y = 2.0, 31.0
    si, b, a = 3.0, 2.0, 15.0
    @test isapprox(y, GaPSE.power_law(x, si, b, a), rtol=1e-6)
end


@testset "test two_power_laws" begin
    xs = range(1, 20, length=100)

    @testset "first" begin
        switch, si_1, si_2, b, a = 5.0, 1.0, 2.0, 1.0, π
        new_b = b / (switch^(si_2 - si_1))
        calc_ys = [GaPSE.two_power_laws(x; switch=switch, si_1=si_1,
            si_2=si_2, b=b, a=a) for x in xs]

        @test isapprox(new_b, 1 / 5; rtol=1e-4)
        @test all([isapprox(GaPSE.power_law(x, si_1, b, a), y; rtol=1e-4)
                    for (x, y) in zip(xs[xs.<=switch], calc_ys[xs.<=switch])])
        @test all([isapprox(GaPSE.power_law(x, si_2, new_b, a), y; rtol=1e-4)
                    for (x, y) in zip(xs[xs.>switch], calc_ys[xs.>switch])])
    end

    @testset "second" begin
        switch, si_1, si_2, b, a = 5.0, -1.0, -2.0, -6.0, 12.0
        new_b = b / (switch^(si_2 - si_1))
        calc_ys = [GaPSE.two_power_laws(x; switch=switch, si_1=si_1,
            si_2=si_2, b=b, a=a) for x in xs]

        @test isapprox(new_b, -30; rtol=1e-4)
        @test all([isapprox(GaPSE.power_law(x, si_1, b, a), y; rtol=1e-4)
                    for (x, y) in zip(xs[xs.<=switch], calc_ys[xs.<=switch])])
        @test all([isapprox(GaPSE.power_law(x, si_2, new_b, a), y; rtol=1e-4)
                    for (x, y) in zip(xs[xs.>switch], calc_ys[xs.>switch])])
    end
end

@testset "test power_law_from_data" begin
    RTOL = 1e-2

    @testset "zeros" begin
        si, b, a = 2.69, 3.45, 0.0
        xs = 1:0.1:10
        ys = [a + b * x^(si) for x in xs]
        p0 = [1.0, 1.0]
        @test_throws AssertionError GaPSE.power_law_from_data(xs[begin:end-1], ys, p0; con=false)
        @test_throws AssertionError GaPSE.power_law_from_data(xs, ys[begin:end-1], p0; con=false)
        @test_throws AssertionError GaPSE.power_law_from_data(xs, ys, [1.0, 1.0, 1.0]; con=false)
        @test_throws AssertionError GaPSE.power_law_from_data(xs, ys, [1.0, 1.0]; con=true)
        @test_throws AssertionError GaPSE.power_law_from_data(xs, ys, [1.0, 1.0], 0.0, 2.0; con=true)
        @test_throws AssertionError GaPSE.power_law_from_data(xs, ys, [1.0, 1.0], 1.0, 111.0; con=true)
    end

    @testset "first" begin
        si, b, a = 2.69, 3.45, 0.0
        xs = 1:0.1:10
        ys = [a + b * x^(si) for x in xs]
        p0 = [1.0, 1.0]
        c_si, c_b, c_a = GaPSE.power_law_from_data(xs, ys, p0; con=false)
        @test isapprox(si, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "second" begin
        si, b, a = -2.69, 3.45, 0.0
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x^(si) for x in xs]
        p0 = [-1.0, 1.0]
        c_si, c_b, c_a = GaPSE.power_law_from_data(xs, ys, p0; con=false)
        @test isapprox(si, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "third" begin
        si, b, a = -2.69, 3.45, 13245.23
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x^(si) for x in xs]
        p0 = [-1.0, 1.0, 1.0]
        @test_throws AssertionError GaPSE.power_law_from_data(xs, ys, p0; con=true)
    end

    @testset "fourth" begin
        si, b, a = -2.65, 1.54e3, 2.34563e-15
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x^(si) for x in xs]
        p0 = [-1.0, 1.0, 1.0]
        c_si, c_b, c_a = GaPSE.power_law_from_data(xs, ys, p0; con=true)
        @test isapprox(si, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end
end



##########################################################################################92



@testset "test expand_left_log" begin
    @testset "zeros" begin
        si, b, a = -2.65, 1.54e3, 2.34563e-15
        lim, fit_min, fit_max = 1e2, 1e3, 1e4
        p0 = [1.0, 1.0]

        xs = 10 .^ range(2, 4; length=100)
        ys = [a + b * x^si for x in xs]
        cutted_xs = xs[fit_min.<xs.<fit_max]
        cutted_ys = ys[fit_min.<xs.<fit_max]

        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs[begin:end-1], ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs, ys[begin:end-1];
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=[1.0, 1.0, 1.0], con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=[1.0, 1.0], con=true)
        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs, ys;
            lim=fit_max, fit_min=fit_min, fit_max=lim, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_max, fit_max=fit_min, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs, ys;
            lim=0.5, fit_min=1.0, fit_max=fit_max, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=1e12, p0=p0, con=false)
    end


    @testset "first" begin
        si, b, a = -2.65, 1.54e3, 0.0
        lim, fit_min, fit_max = 1e2, 1e3, 1e4
        p0 = [-1.0, 1.0]

        xs = 10 .^ range(2, 4; length=100)
        ys = [a + b * x^si for x in xs]
        new_left_xs, new_left_ys = GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_left_xs, xs[xs.<=fit_min])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_left_ys, ys[xs.<=fit_min])])
    end

    @testset "second" begin
        si, b, a = -2.65, 1.54e3, 2.34563e-15
        lim, fit_min, fit_max = 1e2, 1e3, 1e4
        p0 = [-1.0, 1.0, 1.0]

        xs = 10 .^ range(2, 4; length=100)
        ys = [a + b * x^si for x in xs]
        new_left_xs, new_left_ys = GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=true)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_left_xs, xs[xs.<=fit_min])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_left_ys, ys[xs.<=fit_min])])
    end

    @testset "third" begin
        si, b, a = 2.65, 1.54e3, 0.0
        lim, fit_min, fit_max = 1e-6, 1e-5, 1e-4
        p0 = [-1.0, 1.0]

        xs = 10 .^ range(-6, -4; length=100)
        ys = [a + b * x^si for x in xs]
        new_left_xs, new_left_ys = GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_left_xs, xs[xs.<=fit_min])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_left_ys, ys[xs.<=fit_min])])
    end


    @testset "fourth" begin
        si, b, a = 2.65, 1.54e3, 1.4e-12
        lim, fit_min, fit_max = 1e-6, 1e-5, 1e-4
        p0 = [-1.0, 1.0, 0.0]

        xs = 10 .^ range(-6, -4; length=100)
        ys = [a + b * x^si for x in xs]
        new_left_xs, new_left_ys = GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=true)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_left_xs, xs[xs.<=fit_min])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_left_ys, ys[xs.<=fit_min])])
    end

    @testset "fifth" begin
        si, b, a = 2.65, 1.54e3, 1.4e30
        lim, fit_min, fit_max = 1e-6, 1e-5, 1e-4
        p0 = [-1.0, 1.0, 0.0]

        xs = 10 .^ range(-6, -4; length=100)
        ys = [a + b * x^si for x in xs]
        @test_throws AssertionError GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=true)

    end

    @testset "sixth" begin
        xs = 10 .^ range(2, 4; length=9)
        ys = [2 * x for x in xs]
        lim, fit_1, fit_2 = 1e1, 1e3 - 1e-5, 1e4
        new_left_xs, new_left_ys = GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_1, fit_max=fit_2, con=true)

        @test isapprox(new_left_xs[end] / (xs[xs.<fit_1][end]), 1.0, rtol=1e-5)
        @test isapprox(new_left_xs[end] / (562.341325190349), 1.0, rtol=1e-5)
        @test isapprox(new_left_ys[end] / (ys[xs.<fit_1][end]), 1.0, rtol=1e-5)
    end

    @testset "seventh" begin
        xs = 10 .^ range(2, 4; length=9)
        ys = [2 * x for x in xs]
        lim, fit_1, fit_2 = 1e1, 1e3, 1e4
        new_left_xs, new_left_ys = GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_1, fit_max=fit_2, con=true)

        @test isapprox(new_left_xs[end] / (xs[xs.<fit_1][end]), 1.0, rtol=1e-5)
        @test isapprox(new_left_xs[end] / (562.341325190349), 1.0, rtol=1e-5)
        @test isapprox(new_left_ys[end] / (ys[xs.<fit_1][end]), 1.0, rtol=1e-5)
    end

    @testset "eighth" begin
        xs = 10 .^ range(2, 4; length=9)
        ys = [2 * x for x in xs]
        lim, fit_1, fit_2 = 1e1, 1e3 + 1e-5, 1e4
        new_left_xs, new_left_ys = GaPSE.expand_left_log(xs, ys;
            lim=lim, fit_min=fit_1, fit_max=fit_2, con=true)

        @test isapprox(new_left_xs[end] / (xs[xs.<fit_1][end]), 1.0, rtol=1e-5)
        @test isapprox(new_left_xs[end] / (1000.0), 1.0, rtol=1e-5)
        @test isapprox(new_left_ys[end] / (ys[xs.<fit_1][end]), 1.0, rtol=1e-5)
    end
end

@testset "test expand_right_log" begin
    @testset "zeros" begin
        si, b, a = -2.65, 1.54e3, 2.34563e-15
        fit_min, fit_max, lim = 1e2, 1e3, 1e4
        p0 = [1.0, 1.0]

        xs = 10 .^ range(2, 4; length=100)
        ys = [a + b * x^si for x in xs]
        cutted_xs = xs[fit_min.<xs.<fit_max]
        cutted_ys = ys[fit_min.<xs.<fit_max]

        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs[begin:end-1], ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs, ys[begin:end-1];
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=[1.0, 1.0, 1.0], con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=[1.0, 1.0], con=true)
        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs, ys;
            lim=fit_min, fit_min=fit_min, fit_max=lim, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_max, fit_max=fit_min, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=1.0, fit_max=fit_max, p0=p0, con=false)
        @test_throws AssertionError GaPSE.GaPSE.expand_right_log(xs, ys;
            lim=2e12, fit_min=fit_min, fit_max=1e12, p0=p0, con=false)
    end


    @testset "first" begin
        si, b, a = -2.65, 1.54e3, 0.0
        fit_min, fit_max, lim = 1e2, 1e3, 1e4
        p0 = [-1.0, 1.0]

        xs = 10 .^ range(2, 4; length=100)
        ys = [a + b * x^si for x in xs]
        new_right_xs, new_right_ys = GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_right_xs, xs[xs.>=fit_max])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_right_ys, ys[xs.>=fit_max])])
    end

    @testset "second" begin
        si, b, a = -2.65, 1.54e3, 2.34563e-15
        fit_min, fit_max, lim = 1e2, 1e3, 1e4
        p0 = [-1.0, 1.0, 1.0]

        xs = 10 .^ range(2, 4; length=100)
        ys = [a + b * x^si for x in xs]
        new_right_xs, new_right_ys = GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=true)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_right_xs, xs[xs.>=fit_max])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_right_ys, ys[xs.>=fit_max])])
    end

    @testset "third" begin
        si, b, a = 2.65, 1.54e3, 0.0
        fit_min, fit_max, lim = 1e-6, 1e-5, 1e-4
        p0 = [-1.0, 1.0]

        xs = 10 .^ range(-6, -4; length=100)
        ys = [a + b * x^si for x in xs]
        new_right_xs, new_right_ys = GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=false)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_right_xs, xs[xs.>=fit_max])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_right_ys, ys[xs.>=fit_max])])
    end


    @testset "fourth" begin
        si, b, a = 2.65, 1.54e3, 1.4e-12
        fit_min, fit_max, lim = 1e-6, 1e-5, 1e-4
        p0 = [-1.0, 1.0, 0.0]

        xs = 10 .^ range(-6, -4; length=100)
        ys = [a + b * x^si for x in xs]
        new_right_xs, new_right_ys = GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=true)

        @test all([isapprox(x1 / x2, 1.0, rtol=1e-4) for (x1, x2) in zip(new_right_xs, xs[xs.>=fit_max])])
        @test all([isapprox(y1 / y2, 1.0, rtol=1e-4) for (y1, y2) in zip(new_right_ys, ys[xs.>=fit_max])])
    end

    @testset "fifth" begin
        si, b, a = 2.65, 1.54e3, 1.4e30
        fit_min, fit_max, lim = 1e-6, 1e-5, 1e-4
        p0 = [-1.0, 1.0, 0.0]

        xs = 10 .^ range(-6, -4; length=100)
        ys = [a + b * x^si for x in xs]
        @test_throws AssertionError GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_min, fit_max=fit_max, p0=p0, con=true)

    end

    @testset "sixth" begin
        xs = 10 .^ range(2, 4; length=9)
        ys = [2 * x for x in xs]
        fit_1, fit_2, lim = 1e2, 1e3 + 1e-5, 1e6
        new_right_xs, new_right_ys = GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_1, fit_max=fit_2, con=true)

        @test isapprox(new_right_xs[1] / (xs[xs.>fit_2][1]), 1.0, rtol=1e-5)
        @test isapprox(new_right_xs[1] / (1778.2794100389228), 1.0, rtol=1e-5)
        @test isapprox(new_right_ys[1] / (ys[xs.>fit_2][1]), 1.0, rtol=1e-5)
    end

    @testset "seventh" begin
        xs = 10 .^ range(2, 4; length=9)
        ys = [2 * x for x in xs]
        fit_1, fit_2, lim = 1e2, 1e3, 1e6
        new_right_xs, new_right_ys = GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_1, fit_max=fit_2, con=true)

        @test isapprox(new_right_xs[1] / (xs[xs.>fit_2][1]), 1.0, rtol=1e-5)
        @test isapprox(new_right_xs[1] / (1778.2794100389228), 1.0, rtol=1e-5)
        @test isapprox(new_right_ys[1] / (ys[xs.>fit_2][1]), 1.0, rtol=1e-5)
    end

    @testset "eighth" begin
        xs = 10 .^ range(2, 4; length=9)
        ys = [2 * x for x in xs]
        fit_1, fit_2, lim = 1e2, 1e3 - 1e-5, 1e6
        new_right_xs, new_right_ys = GaPSE.expand_right_log(xs, ys;
            lim=lim, fit_min=fit_1, fit_max=fit_2, con=true)

        @test isapprox(new_right_xs[1] / (xs[xs.>fit_2][1]), 1.0, rtol=1e-5)
        @test isapprox(new_right_xs[1] / (1000.0), 1.0, rtol=1e-5)
        @test isapprox(new_right_ys[1] / (ys[xs.>fit_2][1]), 1.0, rtol=1e-5)
    end
end


##########################################################################################92


@testset "test expanded_IPS" begin
    tab_IPS = readdlm("datatest/WideA_ZA_pk.dat", comments=true)
    ks = convert(Vector{Float64}, tab_IPS[:, 1])
    pks = convert(Vector{Float64}, tab_IPS[:, 2])

    tab_expanded_IPS = readdlm("datatest/expanded_IPS_CLASS.txt", comments=true)
    ex_ks = convert(Vector{Float64}, tab_expanded_IPS[:, 1])
    ex_pks = convert(Vector{Float64}, tab_expanded_IPS[:, 2])

    calc_ks, calc_pks = GaPSE.expanded_IPS(ks, pks; k_in=1e-12, k_end=1e5,
        k1=1e-6, k2=3e-6, k3=1e1, k4=2e1)

    @test all([isapprox(x1, x2, rtol=1e-6) for (x1, x2) in zip(calc_ks, ex_ks)])
    @test all([isapprox(x1, x2, rtol=1e-6) for (x1, x2) in zip(calc_pks, ex_pks)])
end

@testset "test expanded_Iln - Dierckx spline" begin
    tab_IPS = readdlm(FILE_PS, comments=true)
    ks = convert(Vector{Float64}, tab_IPS[:, 1])
    pks = convert(Vector{Float64}, tab_IPS[:, 2])
    PK_dierckx = Dierckx.Spline1D(ks, pks)

    table_Iln = readdlm(FILE_ILN, comments=true)
    ss = convert(Vector{Float64}, table_Iln[:, 1])
    Ilns = [convert(Vector{Float64}, table_Iln[:, i]) for i in 2:10]

    lim = 1e-4
    N = 1024
    fit_min, fit_max = 0.05, 0.5
    kmin, kmax, s0 = 1e-5, 1e3, 1e-3
    con = true
    p0 = [-1.0, 1.0, 0.0]

    I00_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 0, 0; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")
    I20_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 2, 0; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")
    I40_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 4, 0; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")
    I02_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 0, 2; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")
    I22_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 2, 2; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")
    I31_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 3, 1; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")
    I13_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 1, 3; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")
    I11_dierckx = Dierckx.Spline1D(GaPSE.expanded_Iln(PK_dierckx, 1, 1; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error")

    RTOL = 1e-3
    @test all([isapprox(I00_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[1])])
    @test all([isapprox(I20_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[2])])
    @test all([isapprox(I40_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[3])])
    @test all([isapprox(I02_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[4])])
    @test all([isapprox(I22_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[5])])
    @test all([isapprox(I31_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[6])])
    @test all([isapprox(I11_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[7])])
    @test all([isapprox(I13_dierckx(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[8])])
end

@testset "test expanded_Iln - MySpline" begin
    tab_IPS = readdlm(FILE_PS, comments=true)
    ks = convert(Vector{Float64}, tab_IPS[:, 1])
    pks = convert(Vector{Float64}, tab_IPS[:, 2])
    PK_myspline = GaPSE.MySpline(ks, pks; ic="ThirdDerivative")

    table_Iln = readdlm(FILE_ILN, comments=true)
    ss = convert(Vector{Float64}, table_Iln[:, 1])
    Ilns = [convert(Vector{Float64}, table_Iln[:, i]) for i in 2:10]

    lim = 1e-4
    N = 1024
    fit_min, fit_max = 0.05, 0.5
    kmin, kmax, s0 = 1e-5, 1e3, 1e-3
    con = true
    p0 = [-1.0, 1.0, 0.0]

    I00_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 0, 0; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")
    I20_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 2, 0; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")
    I40_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 4, 0; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")
    I02_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 0, 2; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")
    I22_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 2, 2; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")
    I31_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 3, 1; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")
    I13_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 1, 3; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")
    I11_myspline = GaPSE.MySpline(GaPSE.expanded_Iln(PK_myspline, 1, 1; lim=lim, N=N, kmin=kmin, kmax=kmax, s0=s0,
            fit_left_min=fit_min, fit_left_max=fit_max, p0_left=p0, con=con)...; bc="error", ic="ThirdDerivative")

    RTOL = 1e-3
    @test all([isapprox(I00_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[1])])
    @test all([isapprox(I20_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[2])])
    @test all([isapprox(I40_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[3])])
    @test all([isapprox(I02_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[4])])
    @test all([isapprox(I22_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[5])])
    @test all([isapprox(I31_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[6])])
    @test all([isapprox(I11_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[7])])
    @test all([isapprox(I13_myspline(s), I, rtol=RTOL) for (s, I) in zip(ss, Ilns[8])])
end

@testset "test func_I04_tilde - Dierckx Spline" begin
    table_ips = readdlm(FILE_PS)
    ks = convert(Vector{Float64}, table_ips[:, 1])
    pks = convert(Vector{Float64}, table_ips[:, 2])
    PK_dierckx = Dierckx.Spline1D(ks, pks)
    kmin, kmax = 1e-5, 1e3

    ss = [3.0387609674820664]
    res = [-1.7693316431457908]
    my_res = [GaPSE.func_I04_tilde(PK_dierckx, s, kmin, kmax) for s in ss]

    @test all([isapprox(my, tr, rtol=1e-3) for (my, tr) in zip(my_res, res)])
end

@testset "test func_I04_tilde - MySpline" begin
    table_ips = readdlm(FILE_PS)
    ks = convert(Vector{Float64}, table_ips[:, 1])
    pks = convert(Vector{Float64}, table_ips[:, 2])
    PK_myspline = GaPSE.MySpline(ks, pks; ic="ThirdDerivative")
    kmin, kmax = 1e-5, 1e3

    ss = [3.0387609674820664]
    res = [-1.7693316431457908]
    my_res = [GaPSE.func_I04_tilde(PK_myspline, s, kmin, kmax) for s in ss]

    @test all([isapprox(my, tr, rtol=1e-3) for (my, tr) in zip(my_res, res)])
end


@testset "test expanded_I04_tilde - Dierckx Spline" begin
    tab_I04_tildes = readdlm("datatest/I04_tilde_extended_NO_CONST.txt", comments=true)
    ss = convert(Vector{Float64}, tab_I04_tildes[:, 1])
    I04_tildes = convert(Vector{Float64}, tab_I04_tildes[:, 2])

    table_ips = readdlm(FILE_PS)
    ks = convert(Vector{Float64}, table_ips[:, 1])
    pks = convert(Vector{Float64}, table_ips[:, 2])
    PK_dierckx = Dierckx.Spline1D(ks, pks)
    kmin, kmax = 1e-5, 1e3
    calc_I04_tildes = GaPSE.expanded_I04_tilde(PK_dierckx, ss; kmin=kmin, kmax=kmax)

    @test all([isapprox(my, tr, rtol=1e-3) for (my, tr) in zip(calc_I04_tildes, I04_tildes)])
end


@testset "test expanded_I04_tilde - MySpline" begin
    tab_I04_tildes = readdlm("datatest/I04_tilde_extended_NO_CONST.txt", comments=true)
    ss = convert(Vector{Float64}, tab_I04_tildes[:, 1])
    I04_tildes = convert(Vector{Float64}, tab_I04_tildes[:, 2])

    table_ips = readdlm(FILE_PS)
    ks = convert(Vector{Float64}, table_ips[:, 1])
    pks = convert(Vector{Float64}, table_ips[:, 2])
    PK_myspline = GaPSE.MySpline(ks, pks; ic="ThirdDerivative")
    kmin, kmax = 1e-5, 1e3
    calc_I04_tildes = GaPSE.expanded_I04_tilde(PK_myspline, ss; kmin=kmin, kmax=kmax)

    @test all([isapprox(my, tr, rtol=1e-3) for (my, tr) in zip(calc_I04_tildes, I04_tildes)])
end



##########################################################################################92


@testset "test my_interpolation" begin
    @test_throws AssertionError GaPSE.my_interpolation(1.0, 1.0, 2.0, 2.0, 0.5)
    @test_throws AssertionError GaPSE.my_interpolation(1.0, 1.0, 2.0, 2.0, 2.5)

    @test isapprox(GaPSE.my_interpolation(1.0, 1.0, 2.0, 4.0, 1.0), 1.0; rtol=1e-4)
    @test isapprox(GaPSE.my_interpolation(1.0, 1.0, 2.0, 4.0, 1.25), 1.75; rtol=1e-4)
    @test isapprox(GaPSE.my_interpolation(1.0, 1.0, 2.0, 4.0, 2.0), 4.0; rtol=1e-4)
end

@testset "test EPLs" begin
    RTOL = 1e-2

    @testset "zeros" begin
        xs = 10 .^ range(1, 2, length=100)
        ys = [2.45e2 * x^1.564 for x in xs]
        p0_l, p0_r = [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]

        @test_throws AssertionError GaPSE.EPLs(xs[begin+1], ys, p0_l, p0_r)
        @test_throws AssertionError GaPSE.EPLs(xs, ys[end-1], p0_l, p0_r)
        @test_throws AssertionError GaPSE.EPLs(xs, ys, [1, 2, 3, 4], p0_r)
        @test_throws AssertionError GaPSE.EPLs(xs, ys, p0_l, [1, 2, 3, 4])
        @test_throws AssertionError GaPSE.EPLs(xs, ys, p0_l, p0_r; N_left=1)
        @test_throws AssertionError GaPSE.EPLs(xs, ys, p0_l, p0_r; N_right=1)
    end

    @testset "first" begin
        switch, si_1, si_2, b, a = 38, 1.3, 2.1, 6.1234e1, 1.123e2

        xs = 10 .^ range(1, 2, length=100)
        ys = [GaPSE.two_power_laws(x; switch=38, si_1=si_1, si_2=si_2, b=b, a=a)
            for x in xs]

        epls = GaPSE.EPLs(xs, ys, [1.0, 1.0, 0.0], [1.0, 1.0, 0.0];
            N_left=15, N_right=15)

        new_xs = 10 .^ range(0, 3, length=1000)
        new_ys = [epls(x) for x in new_xs]

        @test all([isapprox(
            GaPSE.two_power_laws(x; switch=switch, si_1=si_1, si_2=si_2, b=b, a=a),
            y; rtol=RTOL)
                    for (x, y) in zip(new_xs, new_ys)])
    end

end


##########################################################################################92


@testset "test polynomial" begin
    x, y = 2.3, 35.47
    c, b, a = 3.0, 2.0, 15.0
    @test isapprox(y, GaPSE.polynomial(x, c, b, a), rtol=1e-6)
end

@testset "test polynomial_from_data" begin
    RTOL = 2e-2

    @testset "zeros" begin
        c, b, a = 2.69, 3.45, 0.0
        xs = 1:0.1:10
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0]
        @test_throws AssertionError GaPSE.polynomial_from_data(xs[begin:end-1], ys, p0)
        @test_throws AssertionError GaPSE.polynomial_from_data(xs, ys[begin:end-1], p0)
        @test_throws AssertionError GaPSE.polynomial_from_data(xs, ys, [1.0, 1.0, 1.0, 1.0])
        @test_throws AssertionError GaPSE.polynomial_from_data(xs, ys, [1.0, 1.0], 0.0, 2.0)
        @test_throws AssertionError GaPSE.polynomial_from_data(xs, ys, [1.0, 1.0], 1.0, 111.0)
    end

    @testset "one param: first" begin
        c, b, a = 2.69, 0.0, 0.0
        xs = 1:0.1:10
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0]
        c_si, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "one param: second" begin
        c, b, a = 2.696, 0.0, 0.0
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0]
        c_si, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "two params: first" begin
        c, b, a = 2.69, 3.45, 0.0
        xs = 1:0.1:10
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0]
        c_si, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "two params: second" begin
        c, b, a = 6.65e-1, 1.4e12, 0.0
        xs = range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0]
        c_si, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "two params: third" begin
        c, b, a = 1.165e11, 3.12, 0.0
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0]
        c_si, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_si, rtol=RTOL)
        #@test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "two params: fourth" begin
        c, b, a = 6.65e-11, 1e-11, 0.0
        xs = 10 .^ range(-6, -4, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0]
        c_si, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "two params: fifth" begin
        c, b, a = 6.65e-1, 0.0, 0.0
        xs = 10 .^ range(-6, -4, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0]
        c_si, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_si, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL, atol=1e-13)
        @test isapprox(a, c_a, rtol=RTOL)
    end


    @testset "three params: first" begin
        c, b, a = -2.69, 3.45, 1e40
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]
        @test_throws AssertionError GaPSE.polynomial_from_data(xs, ys, p0)
    end

    @testset "three params: second" begin
        c, b, a = 2.65, 1.54, 2.34563
        xs = range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]
        c_c, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)
        @test isapprox(c, c_c, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "three params: third" begin
        c, b, a = 6.65e20, 2.14e-10, 0.0
        xs = range(2, 3, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]

        c_c, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0; err=RTOL)

        @test isapprox(c, c_c, rtol=RTOL)
        #@test isapprox(b, c_b, rtol=RTOL)
        #@test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "three params: fourth" begin
        c, b, a = 6.65e-12, 2.14e10, 0.0
        xs = range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]

        c_c, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)

        #@test isapprox(c, c_c, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        #@test isapprox(a, c_a, rtol=RTOL)
    end


    @testset "three params: fifth" begin
        c, b, a = 6.65, 1.2, 0.0
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]

        c_c, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)

        @test isapprox(c, c_c, rtol=RTOL)
        @test isapprox(b, c_b, rtol=1e-1)
        #@test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "three params: sixth" begin
        c, b, a = 6.65, 1.54e-10, 1e4
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]

        c_c, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)

        @test isapprox(c, c_c, rtol=RTOL)
        #@test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "three params: seventh" begin
        c, b, a = 0, 1.54e2, 1e5
        xs = 10 .^ range(4, 6, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]

        c_c, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)

        #@test isapprox(c, c_c, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end

    @testset "three params: eighth" begin
        c, b, a = -6.65e2, -1.2e3, -1e5
        xs = -10 .^ range(6, 4, length=100)
        ys = [a + b * x + c * x^2 for x in xs]
        p0 = [1.0, 1.0, 1.0]

        c_c, c_b, c_a = GaPSE.polynomial_from_data(xs, ys, p0)

        #@test isapprox(c, c_c, rtol=RTOL)
        @test isapprox(b, c_b, rtol=RTOL)
        @test isapprox(a, c_a, rtol=RTOL)
    end
end

