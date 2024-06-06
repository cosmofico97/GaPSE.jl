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

"""
    const f0 :: Float64

Linear growth rate at present time. Its value is equal to:
```math
    f_0 \\simeq 0.5126998572951
```
"""
const f0 = 5.126998572951e-01


"""
    const D0 :: Float64

Linear growth factor at present time. Its value is equal to:
```math
     D_0 = 1.0
```
"""
const D0 = 1.0


"""
    const ℋ0 :: Float64

Comoving Hubble constant at present time. Its value is, in natural system
(where the speed of light c=1): 
``\\mathcal{H}_0 \\simeq 3.335641\\times10^{-4} \\; h_0^{-1}\\mathrm{Mpc}``
"""
const ℋ0 = 3.3356409519815204e-4 # h_0/Mpc



##########################################################################################92


"""
    BackgroundData(
        z::Vector{Float64}
        conftime::Vector{Float64}
        comdist::Vector{Float64}
        angdist::Vector{Float64}
        lumdist::Vector{Float64}
        D::Vector{Float64}
        f::Vector{Float64}
        ℋ::Vector{Float64}
        ℋ_p::Vector{Float64})

Struct that contains all the relevant cosmological information for
future computations.
The data are stored with increasing distance values 
(so the first ones are associated to `z=0`).
It is internally used in `Cosmology.`

## Arguments

- `z::Vector{Float64}` : redshifts (adimensionals).

- `conftime::Vector{Float64}` : conformal times, measured in [Mpc/h].

- `comdist::Vector{Float64}` : comoving distances, measured in [Mpc/h].

- `angdist::Vector{Float64}` : angular diameter distances, measured in [Mpc/h].

- `lumdist::Vector{Float64}` : luminosity distances, measured in [Mpc/h].

- `D::Vector{Float64}` : linear growth factors, normalized to 1.0 at the present day (adimensional).

- `f::Vector{Float64}` : linear growth rates (adimensional).

- `ℋ::Vector{Float64}` : comoving Hubble parameters, measured in [h/Mpc].

- `ℋ_p::Vector{Float64}` : derivatives of the comoving Hubble parameter wrt the conformal time.
  It is here manually computed with the Dierckx function `derivative`.


## Constructors

`BackgroundData(file::String, z_spline_lim; names = NAMES_BACKGROUND, h = 0.7)`

- `file::string` : input file where the data are stored; it is expected that such file
  is a background output of the CLASS program (link: https://github.com/lesgourg/class_public)

- `z_spline_lim` : the maximum redhsift we are interested in our analysis. The constructor will
  store the data necessary for a study only in `0 < z < z_spline_lim`, for optimisation purposes.

- `names = NAMES_BACKGROUND` : the column names of the `file`. If the colum order change from
  the default one `NAMES_BACKGROUND`, you must set as input the vector of string with the correct
  one, with the SAME names. They are, with the default order:\n
  `$(string(NAMES_BACKGROUND .* " , "...))`

- `h = 0.7` : the adimensional hubble constant. By default, CLASS background data are measured with
  it numerically expressed (so distances are measured in `Mpc`, for example), while this code works
  with `h` in the unit of measure (so distances are measured in `Mpc/h`, for example).
  Change this value to `1.0` if the input data do not have this issue, or to your value of interest 
  (`0.67`, `0.5`, ...).

See also: [`CosmoParams`](@ref), [`Cosmology`](@ref)
"""
struct BackgroundData
    z::Vector{Float64}
    conftime::Vector{Float64}
    comdist::Vector{Float64}
    angdist::Vector{Float64}
    lumdist::Vector{Float64}
    D::Vector{Float64}
    f::Vector{Float64}
    ℋ::Vector{Float64}
    ℋ_p::Vector{Float64}

    function BackgroundData(file::String, z_spline_lim;
        names=NAMES_BACKGROUND, h=0.7)

        @assert 0.0 < h < 2.0 "h = $h is not a reasonable value, it should be 0 < h < 2 ."
        @assert 0 < z_spline_lim < 1e6 "z_spline_lim = $z_spline_lim is not a reasonable value, 0 < z_spline_lim < 1e6 is."
 
        I_redshift = findfirst(x -> x == "z", names)
        #I_comdist = findfirst(x -> x == "comov. dist.", names)

        data = readdlm(file, comments=true)

        #=
        N_z_MAX = findfirst(z -> z <= z_spline_lim, data[:, I_redshift]) - 1
        println("N_z_MAX = $N_z_MAX")
        com_dist_z_MAX = data[:, I_comdist][N_z_MAX]
        println("com_dist_z_MAX  = $com_dist_z_MAX ")
        N_3_com_dist_z_MAX = findfirst(s -> s <= 3.0 * com_dist_z_MAX, data[:, I_comdist]) - 1
        println("N_3_com_dist_z_MAX  = $N_3_com_dist_z_MAX ")

        data_dict = Dict([name => reverse(data[:, i][N_3_com_dist_z_MAX:end])
                        for (i, name) in enumerate(names)]...)
        =#

        N_z_MAX = findfirst(z -> z <= z_spline_lim, data[:, I_redshift]) - 1
        data_dict = Dict([name => reverse(data[:, i][N_z_MAX:end])
                        for (i, name) in enumerate(names)]...)

        com_H = data_dict["H [1/Mpc]"] ./ h ./ (1.0 .+ data_dict["z"])
        conf_time = data_dict["conf. time [Mpc]"] .* h
        spline_com_H = GaPSE.MySpline(reverse(conf_time), reverse(com_H); bc="error")
        com_H_p = [GaPSE.derivative(spline_com_H, t) for t in conf_time]

        new(
            data_dict["z"],
            conf_time,
            data_dict["comov. dist."] .* h,
            data_dict["ang.diam.dist."] .* h,
            data_dict["lum. dist."] .* h,
            data_dict["gr.fac. D"],
            data_dict["gr.fac. f"],
            com_H,
            com_H_p,
        )
    end
end







