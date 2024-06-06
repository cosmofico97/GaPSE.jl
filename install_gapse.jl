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
Pkg.activate(normpath(@__DIR__))

let
     pkgs = [
          "IJulia", "LinearAlgebra",
          "DelimitedFiles", "Documenter", "Suppressor", "Test",
          "Printf", "NPZ", "ProgressMeter",

          "FFTW", "TwoFAST", 
          
          "ArbNumerics", "AssociatedLegendrePolynomials",
          "LegendrePolynomials", "SpecialFunctions",
          "WignerSymbols",

          "Dierckx", "GridInterpolations",

          "LsqFit",

          "QuadGK", "Trapz", "FastGaussQuadrature", "HCubature",
          
          "oneAPI",
     ]
     for pkg in pkgs
          if Base.find_package(pkg) === nothing
               Pkg.add(pkg)
          end
     end
end

Pkg.resolve()
Pkg.precompile()
include("src/GaPSE.jl")



