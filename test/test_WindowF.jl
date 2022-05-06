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

kwargs_F_hcub = Dict(
     :θ_max => π / 2.0, 
     :tolerance => 1e-8, 
     :rtol => 1e-2, 
     :atol => 1e-3,
)

kwargs_F_trapz = Dict(
     :θ_max => π / 2.0::Float64, 
     :tolerance => 1e-8::Float64, 
     :N => 1000::Int64, 
     :en => 1.0::Float64,
)

kwargs_print_map_F = Dict(
     :θ_max => π / 2.0, 
     :tolerance => 1e-8, 
     :rtol => 1e-2, 
     :atol => 1e-3,
     :pr => true,
)

@testset "test F_hcub" begin
     RTOL = 1e-2

     @test isapprox(GaPSE.F_hcub(0, 0; kwargs_F_hcub...)[1], 39.0406; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(1, 0; kwargs_F_hcub...)[1], 29.25801; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(2, 0; kwargs_F_hcub...)[1], 25.28027; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(3, 0; kwargs_F_hcub...)[1], 23.51367; rtol = RTOL)

     @test isapprox(GaPSE.F_hcub(0, -0.8; kwargs_F_hcub...)[1], 38.89266; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(1, -0.8; kwargs_F_hcub...)[1], 23.35162; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(2, -0.8; kwargs_F_hcub...)[1], 11.83636; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(3, -0.8; kwargs_F_hcub...)[1], 10.90119; rtol = RTOL)

     @test isapprox(GaPSE.F_hcub(0, 0.8; kwargs_F_hcub...)[1], 38.89261; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(1, 0.8; kwargs_F_hcub...)[1], 34.85789; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(2, 0.8; kwargs_F_hcub...)[1], 33.54063; rtol = RTOL)
     @test isapprox(GaPSE.F_hcub(3, 0.8; kwargs_F_hcub...)[1], 32.91128; rtol = RTOL)
end

@testset "test F_trapz" begin
     RTOL = 1e-4

     @test isapprox(GaPSE.F_trapz(0, 0; kwargs_F_trapz...), 39.40821; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(1, 0; kwargs_F_trapz...), 29.59887; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(2, 0; kwargs_F_trapz...), 25.55135; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(3, 0; kwargs_F_trapz...), 23.77376; rtol = RTOL)

     @test isapprox(GaPSE.F_trapz(0, -0.8; kwargs_F_trapz...), 39.41779; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(1, -0.8; kwargs_F_trapz...), 23.77100; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(2, -0.8; kwargs_F_trapz...), 13.87924; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(3, -0.8; kwargs_F_trapz...), 11.40667; rtol = RTOL)

     @test isapprox(GaPSE.F_trapz(0, 0.8; kwargs_F_trapz...), 39.41779; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(1, 0.8; kwargs_F_trapz...), 35.42117; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(2, 0.8; kwargs_F_trapz...), 34.04887; rtol = RTOL)
     @test isapprox(GaPSE.F_trapz(3, 0.8; kwargs_F_trapz...), 33.32257; rtol = RTOL)
end


@testset "test print_map_F first method" begin
     name = "datatest/F_first_method.txt"
     output = "F_first_output.txt"

     @testset "zeros" begin
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 0.25; x1 = -0.5)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 0.25; x1 = 1.0, x2 = 0.5)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 0.25; x2 = 11.0)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 0.25; μ1 = -1.5)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 0.25; μ1 = -0.9, μ2 = -0.95)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 0.25; μ2 = 1.5)

          @test_throws AssertionError GaPSE.print_map_F(output, 0.0, 0.25)
          @test_throws AssertionError GaPSE.print_map_F(output, -1.0, 0.25)
          @test_throws AssertionError GaPSE.print_map_F(output, 2.0, 0.25)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 0.0)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, -1.0)
          @test_throws AssertionError GaPSE.print_map_F(output, 0.25, 2.0)
     end

     GaPSE.print_map_F(output, 0.25, 0.25;
          trapz = true, x1 = 0, x2 = 3, μ1 = -1, μ2 = 1,
          Fmap_opts = kwargs_F_trapz)

     @testset "first" begin
          table_output_F = readdlm(output, comments = true)
          output_xs = convert(Vector{Float64}, table_output_F[:, 1])
          output_μs = convert(Vector{Float64}, table_output_F[:, 2])
          output_Fs = convert(Vector{Float64}, table_output_F[:, 3])

          table_F = readdlm(name, comments = true)
          xs = convert(Vector{Float64}, table_F[:, 1])
          μs = convert(Vector{Float64}, table_F[:, 2])
          Fs = convert(Vector{Float64}, table_F[:, 3])

          @test all([x1 ≈ x2 for (x1, x2) in zip(xs, output_xs)])
          @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
          @test all([F1 ≈ F2 for (F1, F2) in zip(Fs, output_Fs)])
     end

     @testset "second" begin
          table_output_F = GaPSE.WindowF(output)
          output_xs = table_output_F.xs
          output_μs = table_output_F.μs
          output_Fs = table_output_F.Fs

          table_F = GaPSE.WindowF(name)
          xs = table_F.xs
          μs = table_F.μs
          Fs = table_F.Fs

          @test all([x1 ≈ x2 for (x1, x2) in zip(xs, output_xs)])
          @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
          @test all([F1 ≈ F2 for (F1, F2) in zip(Fs, output_Fs)])
     end

     rm(output)
end


@testset "test print_map_F second method" begin
     name = "datatest/F_second_method.txt"
     output = "F_second_output.txt"

     calc_xs = [x for x in 0:0.25:3]
     calc_μs = vcat([-1.0, -0.98, -0.95], [μ for μ in -0.9:0.1:0.9], [0.95, 0.98, 1.0])

     @testset "zeros" begin
          @test_throws AssertionError GaPSE.print_map_F(output, [1.0 for i in 1:10], calc_μs)
          @test_throws AssertionError GaPSE.print_map_F(output, calc_xs, [0.5 for i in 1:10])
          @test_throws AssertionError GaPSE.print_map_F(output, [1.0, 2.0, 100.0], calc_μs)
          @test_throws AssertionError GaPSE.print_map_F(output, calc_xs, [-1.5, -0.99, 0.0, 0.99, 1.5])
          @test_throws AssertionError GaPSE.print_map_F(output, reverse(calc_xs), calc_μs)
          @test_throws AssertionError GaPSE.print_map_F(output, calc_xs, reverse(calc_μs))
     end

     GaPSE.print_map_F(output, calc_xs, calc_μs; 
          trapz = true, Fmap_opts = kwargs_F_trapz)

     @testset "first" begin
          table_output_F = readdlm(output, comments = true)
          output_xs = convert(Vector{Float64}, table_output_F[:, 1])
          output_μs = convert(Vector{Float64}, table_output_F[:, 2])
          output_Fs = convert(Vector{Float64}, table_output_F[:, 3])

          table_F = readdlm(name, comments = true)
          xs = convert(Vector{Float64}, table_F[:, 1])
          μs = convert(Vector{Float64}, table_F[:, 2])
          Fs = convert(Vector{Float64}, table_F[:, 3])

          @test all([x1 ≈ x2 for (x1, x2) in zip(xs, output_xs)])
          @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
          @test all([F1 ≈ F2 for (F1, F2) in zip(Fs, output_Fs)])
     end

     @testset "second" begin
          table_output_F = GaPSE.WindowF(output)
          output_xs = table_output_F.xs
          output_μs = table_output_F.μs
          output_Fs = table_output_F.Fs

          table_F = GaPSE.WindowF(name)
          xs = table_F.xs
          μs = table_F.μs
          Fs = table_F.Fs

          @test all([x1 ≈ x2 for (x1, x2) in zip(xs, output_xs)])
          @test all([μ1 ≈ μ2 for (μ1, μ2) in zip(μs, output_μs)])
          @test all([F1 ≈ F2 for (F1, F2) in zip(Fs, output_Fs)])
     end

     rm(output)
end


@testset "test WindowF: first convection" begin
     xs = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
     μs = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
     Fs = [0, 1, 2, 0, 2, 4, 0, 4, 8, 0, 8, 16]

     unique_xs = [0, 1, 2, 3]
     unique_μs = [-1, 0, 1]
     table_Fs = [0 1 2; 0 2 4; 0 4 8; 0 8 16]

     name = "test_WindowF_fc.txt"
     isfile(name) && rm(name)
     open(name, "w") do io
          println(io, "# line of comment")
          println(io, "# another one")
          for (x, μ, F) in zip(xs, μs, Fs)
               println(io, "$x \t $μ \t $F")
          end
     end

     F_fc = GaPSE.WindowF(name)

     @test size(F_fc.xs) == size(unique_xs)
     @test size(F_fc.μs) == size(unique_μs)
     @test size(F_fc.Fs) == size(table_Fs)
     @test all(F_fc.xs .== unique_xs)
     @test all(F_fc.μs .== unique_μs)
     @test all(F_fc.Fs .== table_Fs)

     rm(name)
end

@testset "test WindowF: second convection" begin
     xs = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
     μs = [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
     Fs = [0, 0, 0, 0, 1, 2, 4, 8, 2, 4, 8, 16]

     unique_xs = [0, 1, 2, 3]
     unique_μs = [-1, 0, 1]
     table_Fs = [0 1 2; 0 2 4; 0 4 8; 0 8 16]

     name = "test_WindowF_sc.txt"
     isfile(name) && rm(name)
     open(name, "w") do io
          println(io, "# line of comment")
          println(io, "# another one")
          for (x, μ, F) in zip(xs, μs, Fs)
               println(io, "$x \t $μ \t $F")
          end
     end

     F_sc = GaPSE.WindowF(name)

     @test size(F_sc.xs) == size(unique_xs)
     @test size(F_sc.μs) == size(unique_μs)
     @test size(F_sc.Fs) == size(table_Fs)
     @test all(F_sc.xs .== unique_xs)
     @test all(F_sc.μs .== unique_μs)
     @test all(F_sc.Fs .== table_Fs)

     rm(name)
end
