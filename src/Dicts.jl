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
    const VALID_INTEGRATION_ALGORITHM = [:lobatto, :quad, :trap]

Valid integration lgorithm that can be used in order to perform an 
integration over the ``\\mu`` angle cosine.
"""
const VALID_INTEGRATION_ALGORITHM = [:lobatto, :quad, :trap]


"""
    const VALID_KWARGS_GNC = [
        :L, :use_windows, :alg, :obs,
        :N_trap, :N_lob, :atol_quad, :rtol_quad,
        :N_χs :N_χs_2, :pr, :suit_sampling]

Valid keyword argument names for the GNC function group.
"""
const VALID_KWARGS_GNC = [
    :L, :use_windows, :alg, :obs,
    :N_trap, :N_lob, :atol_quad, :rtol_quad,
    :N_χs, :N_χs_2, :pr, :suit_sampling, :backend
]


##################### PERTURBED LUMINOSITY DISTANCE #############################



"""
    const GR_EFFECTS_LD = [
        "auto_doppler", "auto_lensing",
        "auto_localgp", "auto_integratedgp", 
        
        "lensing_doppler", "doppler_lensing",
        "doppler_localgp", "localgp_doppler",
        "doppler_integratedgp", "integratedgp_doppler",
        "lensing_localgp", "localgp_lensing",
        "lensing_integratedgp", "integratedgp_lensing",
        "localgp_integratedgp", "integratedgp_localgp",
    ]

The names of the GR effects implemented.
Their order is associated with the one in `VEC_ξs_LD`, so be careful
to change it.

See also: [`VEC_ξs_LD`](@ref)
"""
const GR_EFFECTS_LD = [
    "auto_doppler", "auto_lensing",
    "auto_localgp", "auto_integratedgp", "lensing_doppler", "doppler_lensing",
    "doppler_localgp", "localgp_doppler",
    "doppler_integratedgp", "integratedgp_doppler",
    "lensing_localgp", "localgp_lensing",
    "lensing_integratedgp", "integratedgp_lensing",
    "localgp_integratedgp", "integratedgp_localgp",
];


"""
    const VEC_ξs_LD = [
        ξ_LD_Doppler, ξ_LD_Lensing, ξ_LD_LocalGP, ξ_LD_IntegratedGP, 
        ξ_LD_Lensing_Doppler, ξ_LD_Doppler_Lensing,
        ξ_LD_Doppler_LocalGP, ξ_LD_LocalGP_Doppler,
        ξ_LD_Doppler_IntegratedGP, ξ_LD_IntegratedGP_Doppler,
        ξ_LD_Lensing_LocalGP, ξ_LD_LocalGP_Lensing,
        ξ_LD_Lensing_IntegratedGP, ξ_LD_IntegratedGP_Lensing,
        ξ_LD_LocalGP_IntegratedGP, ξ_LD_IntegratedGP_LocalGP
    ]

The names of the GR effect TPCFs implemented.
Their order is associated with the one in `GR_EFFECTS_LD`, so be careful
to change it.

See also: [`GR_EFFECTS_LD`](@ref)
"""
const VEC_ξs_LD = [
    ξ_LD_Doppler, ξ_LD_Lensing, ξ_LD_LocalGP, ξ_LD_IntegratedGP,
    ξ_LD_Lensing_Doppler, ξ_LD_Doppler_Lensing,
    ξ_LD_Doppler_LocalGP, ξ_LD_LocalGP_Doppler,
    ξ_LD_Doppler_IntegratedGP, ξ_LD_IntegratedGP_Doppler,
    ξ_LD_Lensing_LocalGP, ξ_LD_LocalGP_Lensing,
    ξ_LD_Lensing_IntegratedGP, ξ_LD_IntegratedGP_Lensing,
    ξ_LD_LocalGP_IntegratedGP, ξ_LD_IntegratedGP_LocalGP
];



"""
    const DICT_GR_ξs_LD::Dict{String,Function}

For an input key string `effect` from `GR_EFFECTS_LD`, return the 
associated TPCF `DICT_GR_ξs_LD[effect]` from `VEC_ξs_LD`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_LD["auto_doppler"]
ξ_LD_Doppler
```

See also: [`GR_EFFECTS_LD`](@ref), [`VEC_ξs_LD`](@ref)
"""
const DICT_GR_ξs_LD = Dict([k => v for (k, v) in zip(GR_EFFECTS_LD, VEC_ξs_LD)]...)


"""
    const INDEX_GR_EFFECT_LD::Dict{String,Integer}

For an input key string `effect` from `GR_EFFECTS_LD`, return the 
associated index position in that vector.

## Example

```julia
julia> GaPSE.INDEX_GR_EFFECT_LD["auto_doppler"]
1
```

See also: [`GR_EFFECTS_LD`](@ref)
"""
const INDEX_GR_EFFECT_LD = Dict([name => i for (i, name) in enumerate(GR_EFFECTS_LD)]...)


"""
    const GR_EFFECT_INDEX_LD::Dict{Integer,String}

For an input index position `i` of `GR_EFFECTS_LD`, return the 
associated key string `effect`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_LD[1]
"auto_doppler"
```

See also: [`GR_EFFECTS_LD`](@ref)
"""
const GR_EFFECT_INDEX_LD = Dict([i => name for (i, name) in enumerate(GR_EFFECTS_LD)]...)


const GR_EFFECTS_chi_si_LD = [
    "lensing_doppler", "doppler_lensing",
    "doppler_integratedgp", "integratedgp_doppler",
    "lensing_localgp", "localgp_lensing",
    "localgp_integratedgp", "integratedgp_localgp",
];

const GR_EFFECTS_chi_di_LD = [
    "auto_lensing", "auto_integratedgp",
    "lensing_integratedgp", "integratedgp_lensing",
];

const GR_EFFECTS_chi_integrated_LD = vcat(GR_EFFECTS_chi_si_LD, GR_EFFECTS_chi_di_LD)

const chi_si_LD_kwargs = [:N_χs, :en]
const chi_di_LD_kwargs = [:N_χs_2, :en]
const chi_integrated_LD_kwargs = union(chi_si_LD_kwargs, chi_di_LD_kwargs)

function specif_kwargs_LD(name::String, kwargs)
    error = "$name is not a valid GR effect name for luminosity distance.\n" *
            "Valid GR effect names for luminosity distance are the following:\n" *
            string(GaPSE.GR_EFFECTS_LD .* " , "...)

    @assert (name ∈ GaPSE.GR_EFFECTS_LD) error

    if name ∈ GR_EFFECTS_chi_integrated_LD
        if name ∈ GR_EFFECTS_chi_si_LD
            return filter(p -> !(first(p) ∈ chi_di_LD_kwargs && first(p) ∉ chi_si_LD_kwargs), kwargs)
        else
            return filter(p -> !(first(p) ∈ chi_si_LD_kwargs && first(p) ∉ chi_di_LD_kwargs), kwargs)
        end
    else
        return filter(p -> (first(p) ∉ chi_integrated_LD_kwargs), kwargs)
    end
end


##################### GALAXY NUMBER COUNTS #############################




"""
    const GR_EFFECTS_GNC = [
        "auto_newton", "auto_doppler", "auto_lensing",
        "auto_localgp", "auto_integratedgp", 
        
        "newton_doppler", "doppler_newton",
        "newton_lensing", "lensing_newton",
        "newton_localgp", "localgp_newton",
        "newton_integratedgp", "integratedgp_newton",
        "lensing_doppler", "doppler_lensing",
        "doppler_localgp", "localgp_doppler",
        "doppler_integratedgp", "integratedgp_doppler",
        "lensing_localgp", "localgp_lensing",
        "lensing_integratedgp", "integratedgp_lensing",
        "localgp_integratedgp", "integratedgp_localgp",
    ]

The names of the GR effects implemented.
Their order is associated with the one in `VEC_ξs_GNC`, so be careful
to change it.

See also: [`VEC_ξs_GNC`](@ref)
"""
const GR_EFFECTS_GNC = [
    "auto_newton", "auto_doppler", "auto_lensing",
    "auto_localgp", "auto_integratedgp", "newton_doppler", "doppler_newton",
    "newton_lensing", "lensing_newton",
    "newton_localgp", "localgp_newton",
    "newton_integratedgp", "integratedgp_newton",
    "lensing_doppler", "doppler_lensing",
    "doppler_localgp", "localgp_doppler",
    "doppler_integratedgp", "integratedgp_doppler",
    "lensing_localgp", "localgp_lensing",
    "lensing_integratedgp", "integratedgp_lensing",
    "localgp_integratedgp", "integratedgp_localgp",
];


"""
    const VEC_ξs_GNC = [
        ξ_GNC_Newtonian, ξ_GNC_Doppler, ξ_GNC_Lensing,
        ξ_GNC_LocalGP, ξ_GNC_IntegratedGP, 
        
        ξ_GNC_Newtonian_Doppler, ξ_GNC_Doppler_Newtonian,
        ξ_GNC_Newtonian_Lensing, ξ_GNC_Lensing_Newtonian,
        ξ_GNC_Newtonian_LocalGP, ξ_GNC_LocalGP_Newtonian,
        ξ_GNC_Newtonian_IntegratedGP, ξ_GNC_IntegratedGP_Newtonian,
        ξ_GNC_Lensing_Doppler, ξ_GNC_Doppler_Lensing,
        ξ_GNC_Doppler_LocalGP, ξ_GNC_LocalGP_Doppler,
        ξ_GNC_Doppler_IntegratedGP, ξ_GNC_IntegratedGP_Doppler,
        ξ_GNC_Lensing_LocalGP, ξ_GNC_LocalGP_Lensing,
        ξ_GNC_Lensing_IntegratedGP, ξ_GNC_IntegratedGP_Lensing,
        ξ_GNC_LocalGP_IntegratedGP, ξ_GNC_IntegratedGP_LocalGP
    ]

The names of the GR effect TPCFs implemented.
Their order is associated with the one in `GR_EFFECTS_GNC`, so be careful
to change it.

See also: [`GR_EFFECTS_GNC`](@ref)
"""
const VEC_ξs_GNC = [
    ξ_GNC_Newtonian, ξ_GNC_Doppler, ξ_GNC_Lensing,
    ξ_GNC_LocalGP, ξ_GNC_IntegratedGP, ξ_GNC_Newtonian_Doppler, ξ_GNC_Doppler_Newtonian,
    ξ_GNC_Newtonian_Lensing, ξ_GNC_Lensing_Newtonian,
    ξ_GNC_Newtonian_LocalGP, ξ_GNC_LocalGP_Newtonian,
    ξ_GNC_Newtonian_IntegratedGP, ξ_GNC_IntegratedGP_Newtonian,
    ξ_GNC_Lensing_Doppler, ξ_GNC_Doppler_Lensing,
    ξ_GNC_Doppler_LocalGP, ξ_GNC_LocalGP_Doppler,
    ξ_GNC_Doppler_IntegratedGP, ξ_GNC_IntegratedGP_Doppler,
    ξ_GNC_Lensing_LocalGP, ξ_GNC_LocalGP_Lensing,
    ξ_GNC_Lensing_IntegratedGP, ξ_GNC_IntegratedGP_Lensing,
    ξ_GNC_LocalGP_IntegratedGP, ξ_GNC_IntegratedGP_LocalGP
];



"""
    const DICT_GR_ξs_GNC::Dict{String,Function}

For an input key string `effect` from `GR_EFFECTS_GNC`, return the 
associated TPCF `DICT_GR_ξs_GNC[effect]` from `VEC_ξs_GNC`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_GNC["auto_doppler"]
ξ_GNC_Doppler
```

See also: [`GR_EFFECTS_GNC`](@ref), [`VEC_ξs_GNC`](@ref)
"""
const DICT_GR_ξs_GNC = Dict([k => v for (k, v) in zip(GR_EFFECTS_GNC, VEC_ξs_GNC)]...)



"""
    const DICT_GR_ξs_GNC::Dict{Function,String}

For an input key TPCF function `func` from `VEC_ξs_GNC`, return the 
associated string `DICT_GR_ξs_GNC[func]` from `GR_EFFECTS_GNC`,

## Example

```julia
julia> GaPSE.DICT_GR_ξs_GNC[ξ_GNC_Doppler]
"auto_doppler"
```

See also: [`GR_EFFECTS_GNC`](@ref), [`VEC_ξs_GNC`](@ref)
"""
const DICT_ξs_GR_GNC = Dict([k => v for (k, v) in zip(VEC_ξs_GNC, GR_EFFECTS_GNC)]...)


"""
    const INDEX_GR_EFFECT_GNC::Dict{String,Integer}

For an input key string `effect` from `GR_EFFECTS_GNC`, return the 
associated index position in that vector.

## Example

```julia
julia> GaPSE.INDEX_GR_EFFECT_GNC["auto_doppler"]
1
```

See also: [`GR_EFFECTS_GNC`](@ref)
"""
const INDEX_GR_EFFECT_GNC = Dict([name => i for (i, name) in enumerate(GR_EFFECTS_GNC)]...)


"""
    const GR_EFFECT_INDEX_GNC::Dict{Integer,String}

For an input index position `i` of `GR_EFFECTS_GNC`, return the 
associated key string `effect`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_GNC[1]
"auto_doppler"
```

See also: [`GR_EFFECTS_GNC`](@ref)
"""
const GR_EFFECT_INDEX_GNC = Dict([i => name for (i, name) in enumerate(GR_EFFECTS_GNC)]...)


const GR_EFFECTS_chi_si_GNC = [
    "newton_lensing", "lensing_newton",
    "newton_integratedgp", "integratedgp_newton",
    "lensing_doppler", "doppler_lensing",
    "doppler_integratedgp", "integratedgp_doppler",
    "lensing_localgp", "localgp_lensing",
    "localgp_integratedgp", "integratedgp_localgp",
];

const GR_EFFECTS_chi_di_GNC = [
    "auto_lensing", "auto_integratedgp",
    "lensing_integratedgp", "integratedgp_lensing",
];

const GR_EFFECTS_chi_integrated_GNC = vcat(GR_EFFECTS_chi_si_GNC, GR_EFFECTS_chi_di_GNC)

const chi_si_GNC_kwargs = [:N_χs, :en, :suit_sampling, :backend]
const chi_di_GNC_kwargs = [:N_χs_2, :en, :suit_sampling, :backend]
const chi_integrated_GNC_kwargs = union(chi_si_GNC_kwargs, chi_di_GNC_kwargs)


function specif_kwargs_GNC(name::String, kwargs)
    error = "$name is not a valid GR effect name for galaxy number counts.\n" *
            "Valid GR effect names for galaxy number counts are the following:\n" *
            string(GaPSE.GR_EFFECTS_GNC .* " , "...)

    @assert (name ∈ GaPSE.GR_EFFECTS_GNC) error

    if name ∈ GR_EFFECTS_chi_integrated_GNC
        if name ∈ GR_EFFECTS_chi_si_GNC
            return filter(p -> !(first(p) ∈ chi_di_GNC_kwargs && first(p) ∉ chi_si_GNC_kwargs), kwargs)
        else
            return filter(p -> !(first(p) ∈ chi_si_GNC_kwargs && first(p) ∉ chi_di_GNC_kwargs), kwargs)
        end
    else
        return filter(p -> (first(p) ∉ chi_integrated_GNC_kwargs), kwargs)
    end
end

"""
    const EFFECTS_WITH_OBS_VEL = [
        "auto_doppler",
        "newton_doppler", "doppler_newton",
        "lensing_doppler", "doppler_lensing",
        "doppler_localgp", "localgp_doppler",
        "doppler_integratedgp", "integratedgp_doppler",
    ] 

Contains the names of the GNC effects that have observer terms derived from
a non-zero observer velocity.
"""
const EFFECTS_WITH_OBS_VEL = [
    "auto_doppler",
    "newton_doppler", "doppler_newton",
    "lensing_doppler", "doppler_lensing",
    "doppler_localgp", "localgp_doppler",
    "doppler_integratedgp", "integratedgp_doppler",
]


"""
    const VALID_OBS_VALUES = [:yes, :no, :noobsvel]

Contains the valid Symbols for the variable "obs", which refers to the GNC
terms related to the observer.
"""
const VALID_OBS_VALUES = [:yes, :no, :noobsvel]

##################### GALAXY NUMBER COUNTS x LUMINOSITY DISTANCE #############################




"""
    const GR_EFFECTS_GNCxLD = [
        "newton_doppler", 
        "newton_lensing", 
        "newton_localgp", 
        "newton_integratedgp",

        "doppler_doppler",
        "doppler_lensing",
        "doppler_localgp", 
        "doppler_integratedgp",

        "lensing_doppler",
        "lensing_lensing",
        "lensing_localgp",
        "lensing_integratedgp",
        
        "localgp_doppler",
        "localgp_lensing",
        "localgp_localgp",
        "localgp_integratedgp",

        "integratedgp_doppler",
        "integratedgp_lensing",
        "integratedgp_localgp",
        "integratedgp_integratedgp",
    ]

The names of the GR effects implemented.
Their order is associated with the one in `VEC_ξs_GNCxLD`, so be careful
to change it.

See also: [`VEC_ξs_GNCxLD`](@ref)
"""
const GR_EFFECTS_GNCxLD = [
    "newton_doppler",
    "newton_lensing",
    "newton_localgp",
    "newton_integratedgp", "doppler_doppler",
    "doppler_lensing",
    "doppler_localgp",
    "doppler_integratedgp", "lensing_doppler",
    "lensing_lensing",
    "lensing_localgp",
    "lensing_integratedgp", "localgp_doppler",
    "localgp_lensing",
    "localgp_localgp",
    "localgp_integratedgp", "integratedgp_doppler",
    "integratedgp_lensing",
    "integratedgp_localgp",
    "integratedgp_integratedgp",
];


"""
    const GR_EFFECTS_LDxGNC = [
        "doppler_newton", 
        "lensing_newton", 
        "localgp_newton", 
        "integratedgp_newton",

        "doppler_doppler",
        "lensing_doppler",
        "localgp_doppler", 
        "integratedgp_doppler",

        "doppler_lensing",
        "lensing_lensing",
        "localgp_lensing",
        "integratedgp_lensing",
        
        "doppler_localgp",
        "lensing_localgp",
        "localgp_localgp",
        "integratedgp_localgp",

        "doppler_integratedgp",
        "lensing_integratedgp",
        "localgp_integratedgp",
        "integratedgp_integratedgp",
    ]

The names of the GR effects implemented.
Their order is associated with the one in `VEC_ξs_LDxGNC`, so be careful
to change it.

See also: [`VEC_ξs_LDxGNC`](@ref)
"""
const GR_EFFECTS_LDxGNC = [
    "doppler_newton",
    "lensing_newton",
    "localgp_newton",
    "integratedgp_newton", "doppler_doppler",
    "lensing_doppler",
    "localgp_doppler",
    "integratedgp_doppler", "doppler_lensing",
    "lensing_lensing",
    "localgp_lensing",
    "integratedgp_lensing", "doppler_localgp",
    "lensing_localgp",
    "localgp_localgp",
    "integratedgp_localgp", "doppler_integratedgp",
    "lensing_integratedgp",
    "localgp_integratedgp",
    "integratedgp_integratedgp",
];


##########


"""
    const VEC_ξs_GNCxLD = [
        ξ_GNCxLD_Newtonian_Doppler,
        ξ_GNCxLD_Newtonian_Lensing, 
        ξ_GNCxLD_Newtonian_LocalGP, 
        ξ_GNCxLD_Newtonian_IntegratedGP, 

        ξ_GNCxLD_Doppler_Doppler,
        ξ_GNCxLD_Doppler_Lensing,
        ξ_GNCxLD_Doppler_LocalGP, 
        ξ_GNCxLD_Doppler_IntegratedGP, 
        
        ξ_GNCxLD_Lensing_Doppler,
        ξ_GNCxLD_Lensing_Lensing,
        ξ_GNCxLD_Lensing_LocalGP,
        ξ_GNCxLD_Lensing_IntegratedGP,

        ξ_GNCxLD_LocalGP_Doppler,
        ξ_GNCxLD_LocalGP_Lensing,
        ξ_GNCxLD_LocalGP_LocalGP,
        ξ_GNCxLD_LocalGP_IntegratedGP,

        ξ_GNCxLD_IntegratedGP_Doppler,
        ξ_GNCxLD_IntegratedGP_Lensing,
        ξ_GNCxLD_IntegratedGP_LocalGP,
        ξ_GNCxLD_IntegratedGP_IntegratedGP,
    ]

The names of the GR effect TPCFs implemented.
Their order is associated with the one in `GR_EFFECTS_GNCxLD`, so be careful
to change it.

See also: [`GR_EFFECTS_GNCxLD`](@ref)
"""
const VEC_ξs_GNCxLD = [
    ξ_GNCxLD_Newtonian_Doppler,
    ξ_GNCxLD_Newtonian_Lensing,
    ξ_GNCxLD_Newtonian_LocalGP,
    ξ_GNCxLD_Newtonian_IntegratedGP, ξ_GNCxLD_Doppler_Doppler,
    ξ_GNCxLD_Doppler_Lensing,
    ξ_GNCxLD_Doppler_LocalGP,
    ξ_GNCxLD_Doppler_IntegratedGP, ξ_GNCxLD_Lensing_Doppler,
    ξ_GNCxLD_Lensing_Lensing,
    ξ_GNCxLD_Lensing_LocalGP,
    ξ_GNCxLD_Lensing_IntegratedGP, ξ_GNCxLD_LocalGP_Doppler,
    ξ_GNCxLD_LocalGP_Lensing,
    ξ_GNCxLD_LocalGP_LocalGP,
    ξ_GNCxLD_LocalGP_IntegratedGP, ξ_GNCxLD_IntegratedGP_Doppler,
    ξ_GNCxLD_IntegratedGP_Lensing,
    ξ_GNCxLD_IntegratedGP_LocalGP,
    ξ_GNCxLD_IntegratedGP_IntegratedGP,
];


"""
    const VEC_ξs_LDxGNC = [
        ξ_LDxGNC_Doppler_Newtonian,
        ξ_LDxGNC_Lensing_Newtonian,
        ξ_LDxGNC_LocalGP_Newtonian,
        ξ_LDxGNC_IntegratedGP_Newtonian,

        ξ_LDxGNC_Doppler_Doppler,
        ξ_LDxGNC_Lensing_Doppler,
        ξ_LDxGNC_LocalGP_Doppler,
        ξ_LDxGNC_IntegratedGP_Doppler,

        ξ_LDxGNC_Doppler_Lensing,
        ξ_LDxGNC_Lensing_Lensing,
        ξ_LDxGNC_LocalGP_Lensing,
        ξ_LDxGNC_IntegratedGP_Lensing,

        ξ_LDxGNC_Doppler_LocalGP,
        ξ_LDxGNC_Lensing_LocalGP,
        ξ_LDxGNC_LocalGP_LocalGP,
        ξ_LDxGNC_IntegratedGP_LocalGP,

        ξ_LDxGNC_Doppler_IntegratedGP,
        ξ_LDxGNC_Lensing_IntegratedGP,
        ξ_LDxGNC_LocalGP_IntegratedGP,
        ξ_LDxGNC_IntegratedGP_IntegratedGP,
    ]

The names of the GR effect TPCFs implemented.
Their order is associated with the one in `GR_EFFECTS_LDxGNC`, so be careful
to change it.

See also: [`GR_EFFECTS_LDxGNC`](@ref)
"""
const VEC_ξs_LDxGNC = [
    ξ_LDxGNC_Doppler_Newtonian,
    ξ_LDxGNC_Lensing_Newtonian,
    ξ_LDxGNC_LocalGP_Newtonian,
    ξ_LDxGNC_IntegratedGP_Newtonian, ξ_LDxGNC_Doppler_Doppler,
    ξ_LDxGNC_Lensing_Doppler,
    ξ_LDxGNC_LocalGP_Doppler,
    ξ_LDxGNC_IntegratedGP_Doppler, ξ_LDxGNC_Doppler_Lensing,
    ξ_LDxGNC_Lensing_Lensing,
    ξ_LDxGNC_LocalGP_Lensing,
    ξ_LDxGNC_IntegratedGP_Lensing, ξ_LDxGNC_Doppler_LocalGP,
    ξ_LDxGNC_Lensing_LocalGP,
    ξ_LDxGNC_LocalGP_LocalGP,
    ξ_LDxGNC_IntegratedGP_LocalGP, ξ_LDxGNC_Doppler_IntegratedGP,
    ξ_LDxGNC_Lensing_IntegratedGP,
    ξ_LDxGNC_LocalGP_IntegratedGP,
    ξ_LDxGNC_IntegratedGP_IntegratedGP,
];


##########


"""
    const DICT_GR_ξs_GNCxLD::Dict{String,Function}

For an input key string `effect` from `GR_EFFECTS_GNCxLD`, return the 
associated TPCF `DICT_GR_ξs_GNCxLD[effect]` from `VEC_ξs_GNCxLD`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_GNCxLD["lensing_doppler"]
ξ_GNCxLD_Lensing_Doppler
```

See also: [`GR_EFFECTS_GNCxLD`](@ref), [`VEC_ξs_GNCxLD`](@ref)
"""
const DICT_GR_ξs_GNCxLD = Dict([k => v for (k, v) in zip(GR_EFFECTS_GNCxLD, VEC_ξs_GNCxLD)]...)


"""
    const DICT_GR_ξs_LDxGNC::Dict{String,Function}

For an input key string `effect` from `GR_EFFECTS_LDxGNC`, return the 
associated TPCF `DICT_GR_ξs_LDxGNC[effect]` from `VEC_ξs_v`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_LDxGNC["lensing_doppler"]
ξ_LDxGNC_Lensing_Doppler
```

See also: [`GR_EFFECTS_LDxGNC`](@ref), [`VEC_ξs_LDxGNC`](@ref)
"""
const DICT_GR_ξs_LDxGNC = Dict([k => v for (k, v) in zip(GR_EFFECTS_LDxGNC, VEC_ξs_LDxGNC)]...)


##########


"""
    const INDEX_GR_EFFECT_GNCxLD::Dict{String,Integer}

For an input key string `effect` from `GR_EFFECTS_GNCxLD`, return the 
associated index position in that vector.

## Example

```julia
julia> GaPSE.INDEX_GR_EFFECT_GNCxLD["newton_lensing"]
2
```

See also: [`GR_EFFECTS_GNCxLD`](@ref)
"""
const INDEX_GR_EFFECT_GNCxLD = Dict([name => i for (i, name) in enumerate(GR_EFFECTS_GNCxLD)]...)


"""
    const INDEX_GR_EFFECT_LDxGNC::Dict{String,Integer}

For an input key string `effect` from `GR_EFFECTS_LDxGNC`, return the 
associated index position in that vector.

## Example

```julia
julia> GaPSE.INDEX_GR_EFFECT_LDxGNC["lensing_newton"]
2
```

See also: [`GR_EFFECTS_LDxGNC`](@ref)
"""
const INDEX_GR_EFFECT_LDxGNC = Dict([name => i for (i, name) in enumerate(GR_EFFECTS_LDxGNC)]...)


##########


"""
    const GR_EFFECT_INDEX_GNCxLD::Dict{Integer,String}

For an input index position `i` of `GR_EFFECTS_GNCxLD`, return the 
associated key string `effect`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_GNCxLD[2]
"newton_doppler"
```

See also: [`GR_EFFECTS_GNCxLD`](@ref)
"""
const GR_EFFECT_INDEX_GNCxLD = Dict([i => name for (i, name) in enumerate(GR_EFFECTS_GNCxLD)]...)


"""
    const GR_EFFECT_INDEX_LDxGNC::Dict{Integer,String}

For an input index position `i` of `GR_EFFECTS_LDxGNC`, return the 
associated key string `effect`.

## Example

```julia
julia> GaPSE.DICT_GR_ξs_LDxGNC[2]
"doppler_newton"
```

See also: [`GR_EFFECTS_LDxGNC`](@ref)
"""
const GR_EFFECT_INDEX_LDxGNC = Dict([i => name for (i, name) in enumerate(GR_EFFECTS_LDxGNC)]...)


###########################################



const GR_EFFECTS_chi_si_GNCxLD = [
    "newton_lensing",
    "newton_integratedgp", "doppler_lensing",
    "doppler_integratedgp", "lensing_doppler",
    "lensing_localgp", "localgp_lensing",
    "localgp_integratedgp", "integratedgp_doppler",
    "integratedgp_localgp",
];

const GR_EFFECTS_chi_di_GNCxLD = [
    "lensing_lensing",
    "lensing_integratedgp",
    "integratedgp_lensing",
    "integratedgp_integratedgp",
];

const GR_EFFECTS_chi_integrated_GNCxLD = vcat(GR_EFFECTS_chi_si_GNCxLD, GR_EFFECTS_chi_di_GNCxLD)

const chi_si_GNCxLD_kwargs = [:N_χs, :en]
const chi_di_GNCxLD_kwargs = [:N_χs_2, :en]
const chi_integrated_GNCxLD_kwargs = union(chi_si_GNCxLD_kwargs, chi_di_GNCxLD_kwargs)

function specif_kwargs_GNCxLD(name::String, kwargs)
    error = "$name is not a valid GR effect name for GNC x LD.\n" *
            "Valid GR effect names for GNC x LD are the following:\n" *
            string(GaPSE.GR_EFFECTS_GNCxLD .* " , "...)

    @assert (name ∈ GaPSE.GR_EFFECTS_GNCxLD) error

    if name ∈ GR_EFFECTS_chi_integrated_GNCxLD
        if name ∈ GR_EFFECTS_chi_si_GNCxLD
            return filter(p -> !(first(p) ∈ chi_di_GNCxLD_kwargs && first(p) ∉ chi_si_GNCxLD_kwargs), kwargs)
        else
            return filter(p -> !(first(p) ∈ chi_si_GNCxLD_kwargs && first(p) ∉ chi_di_GNCxLD_kwargs), kwargs)
        end
    else
        return filter(p -> (first(p) ∉ chi_integrated_GNCxLD_kwargs), kwargs)
    end
end





const GR_EFFECTS_chi_si_LDxGNC = [
    "lensing_newton",
    "integratedgp_newton", "lensing_doppler",
    "integratedgp_doppler", "doppler_lensing",
    "localgp_lensing", "lensing_localgp",
    "integratedgp_localgp", "doppler_integratedgp",
    "localgp_integratedgp",
];

const GR_EFFECTS_chi_di_LDxGNC = [
    "lensing_lensing",
    "lensing_integratedgp",
    "integratedgp_lensing",
    "integratedgp_integratedgp",
];

const GR_EFFECTS_chi_integrated_LDxGNC = vcat(GR_EFFECTS_chi_si_LDxGNC, GR_EFFECTS_chi_di_LDxGNC)

const chi_si_LDxGNC_kwargs = [:N_χs, :en]
const chi_di_LDxGNC_kwargs = [:N_χs_2, :en]
const chi_integrated_LDxGNC_kwargs = union(chi_si_LDxGNC_kwargs, chi_di_LDxGNC_kwargs)

function specif_kwargs_LDxGNC(name::String, kwargs)
    error = "$name is not a valid GR effect name for LD x GNC.\n" *
            "Valid GR effect names for LD x GNC are the following:\n" *
            string(GaPSE.GR_EFFECTS_LDxGNC .* " , "...)

    @assert (name ∈ GaPSE.GR_EFFECTS_LDxGNC) error

    if name ∈ GR_EFFECTS_chi_integrated_LDxGNC
        if name ∈ GR_EFFECTS_chi_si_LDxGNC
            return filter(p -> !(first(p) ∈ chi_di_LDxGNC_kwargs && first(p) ∉ chi_si_LDxGNC_kwargs), kwargs)
        else
            return filter(p -> !(first(p) ∈ chi_si_LDxGNC_kwargs && first(p) ∉ chi_di_LDxGNC_kwargs), kwargs)
        end
    else
        return filter(p -> (first(p) ∉ chi_integrated_GNCxLD_kwargs), kwargs)
    end
end



##################### miscellanea #############################



const GR_EFFECTS_VECTOR = [
    GaPSE.GR_EFFECTS_LD,
    GaPSE.GR_EFFECTS_GNC,
    GaPSE.GR_EFFECTS_GNCxLD,
    GaPSE.GR_EFFECTS_LDxGNC
];

