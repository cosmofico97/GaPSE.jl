# GaPSE - a model for the Galaxy Power Spectrum Estimator

![julia-version](https://img.shields.io/badge/julia_version-v1.8-9558B2?style=flat&logo=julia) 
![package-version](https://img.shields.io/github/v/release/foglienimatteo/GaPSE.jl?include_prereleases)
![CI-build](https://img.shields.io/github/actions/workflow/status/foglienimatteo/GaPSE.jl/UnitTests.yml)
![size](https://img.shields.io/github/repo-size/foglienimatteo/GaPSE.jl) 
![license]( https://img.shields.io/github/license/foglienimatteo/GaPSE.jl)
[![codecov](https://codecov.io/gh/foglienimatteo/GaPSE.jl/branch/main/graph/badge.svg?token=67GIZ9RA8Y)](https://codecov.io/gh/foglienimatteo/GaPSE.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://foglienimatteo.github.io/GaPSE.jl/stable) 

GaPSE (Galaxy Power Spectrum Estimator) is a software for cosmological computations written in the [Julia Programming Language](https://julialang.org).

IMPORTANT NOTE: This is a work-in-progress project! As a consequence, currently in this pre-release:
- it is possible to compute the power spectrum/correlation function multipoles with L=1,2,3,... of the effects we'll show next, but 2 effects among the Galaxy Number Counts multipoles (Newton-Lensing and Lensing-Newton) converge very slowly, so their computation is not still 100% ready. However, the monopole (L=0) computations do not have any problem with `quad`, and even the GNC sum for higher order multipoles is not affected;
- The Power Spectrum computations with `:twofast` do not work properly, you should always prefer `:fftog`. However, due to the fact that with `:fftlog` you must specify manually the bias parameter, the Power Spectra of a whole group of terms creates FFT oscillations in the smallest ones. The leading ones and the sum are not however affected.  
- the code functions are well documented; check the github pages website https://foglienimatteo.github.io/GaPSE.jl/stable if you can't see correctly the analytical expressions written in the docstrings; 
- few people used this code, so bugs are behind the corner; do not hesitate to raise the finger to point out them (see in the [How to report bugs, suggest improvements and/or contribute](#how-to-report-bugs-suggest-improvements-andor-contribute) section below)!
- if you use this code, please read the [Using this code](##using-this-code) section below


## Table of Contents

- [GaPSE - a model for the Galaxy Power Spectrum Estimator](#gapse---a-model-for-the-galaxy-power-spectrum-estimator)
  - [Table of Contents](#table-of-contents)
  - [Brief description](#brief-description)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dependencies](#dependencies)
  - [How to report bugs, suggest improvements and/or contribute](#how-to-report-bugs-suggest-improvements-andor-contribute)
  - [Using this code](#using-this-code)
  - [Licence](#licence)
  - [References](#references)


## Brief description

Measurements of the clustering of galaxies in Fourier space, at low wavenumbers, offer a window into the early Universe via the possible presence of scale dependent bias generated by Primordial Non Gaussianities [[1]](#1) [[2]](#1).
On such large scales, a Newtonian treatment of density of density perturbations might not be sufficient to describe the measurements, and a fully relativistic calculation should be employed.

Given the matter Power Spectrum (PS) at redshift $z=0$ and the background quantities for the Universe considered (both read from [CLASS](https://github.com/lesgourg/class_public) outputs), this program can compute:

- all the 16 TPCFs arising from the Luminosity Distance (LD) perturbations (see Eq.(2.48) of [[4]](#1)) for an arbitrary multipole order.

- all the 25 TPCFs concerning the relativistic Galaxy Number Counts (GNC)  (see Eq.(2.52) of [[4]](#1)) for an arbitrary multipole order.

- all the 20 cross correlations between GNC and LD (and their 20 counterparts LD cross GNC) for an arbitrary multipole order.

- the PS multipoles of all of them (based on the Yamamoto estimator).

- the Doppler and matter TPCFs in the plane-parallel approximation.

All these calculations can be performed both with and without a survey window function. The code implements also a toy-survey with azymuthal symmetry.

This project, and the analytical expressions used for the TPCFs, are based on the article of Emanuele Castorina and Enea Di Dio [[3]](#1). 

## Installation

Currently, this package is not in the Julia package registries. Assuming that you have already installed a coompatible Julia version, the simplest way to install this software is then the following:

- in the terminal, go to the directory you want to install this package;
  
- clone this repository with Git
  ```bash
  $ git clone https://github.com/foglienimatteo/GaPSE.jl
  ```
  or manually download the source code from the url https://github.com/foglienimatteo/GaPSE.jl (Code > Download Zip)

- go inside the directory of GaPSE (`$ cd GaPSE.jl` in the shell) 

Inside the directory, there is a file called `install_gapse.jl`, which is a Julia script conceived for downloading and installing all the dependencies of GaPSE. You can run it by typing in the terminal:

```bash
     $ julia install_gapse.jl
```
If there are no error messages at the end of the installations, than GaPSE is corretly configured and you can start to use it!
  


NOTE: instead of using the `install_gapse.jl` script, you can also do the same in a more interactive way, if you prefer:

- open a Julia REPL session and activate the project; you can achieve that opening the REPL with 
  ```bash
  $ julia --activate=.
  ```
  or directy inside the REPL entering the Pkg mode (`]`) and running `activate .`

- enter the Pkg mode (if you haven't done in the previous step) typing `]` and run `instantiate`; this command will automatically detect and install all the package dependecies (listed in `Project.toml`)

- done! You can exit from the package mode (press the Backspace key on an empty line) and start to use GaPSE

## Usage

The only mandatory instrument to run the code is a Julia REPL with version ≥ 1.8. 
There are three ways in order to use this code:

- you can write whatever instruction inside the file `GaPSE-exe.jl` and then run in the command line
  ```bash
    $ julia GaPSE-exe.jl
  ```

- you can open a Julia REPL session, include the code with
  ```julia
     include("<path-to-GaPSE.jl-directory>/src/GaPSE.jl")
  ```
  and then use interactively the GaPSE functions

- you can run the same `include("<path-to-GaPSE.jl-directory>/src/GaPSE.jl")` command in a Jupyter Notebook, and use the code functions inside it. This is by far the most confortable way.

Some `.ipynb`s are already provided in the directory `ipynbs` :
- we encourage you to follow the `ipynbs/TUTORIAL.ipynb` file first. The basic structure of the code and the most important functions are there presented
- `ipynbs/Computations_b1p5-sb0-fevo0.ipynb` explains the analytical Primordial Non-Gaussianities model we use here, compute its contribution in the redshift bin $1.0 \leq z \leq 1.5$ and compare it with the GNC effects, all using our toy-model window function with angular opening $\theta_{\rm max} = \pi/2$
- `ipynbs/Generic_Window.ipynb` explains how to use GaPSE with a generic Window Function of your choice
- the `ipynbs/Computations_b1p5-sb0-fevo0.jl` Julia file its the translation into script of `ipynbs/Computations_b1p5-sb0-fevo0.ipynb`; you can easily run it from the command line with:
  ```bash
    $ julia Computations_b1p5-sb0-fevo0.jl
  ```

The code is well tested and documented: almost each struct/function has a docstring that you can easily access in Julia with `?<name-of-the-struct/function>`, and there is an acitive GitHub Pages website with the [latest stable documentation](https://foglienimatteo.github.io/GaPSE/stable).

## Dependencies

GaPSE.jl makes extensive use of the following packages:

- [TwoFAST](https://github.com/hsgg/TwoFAST.jl)[[5]](#1), [FFTLog](https://github.com/marcobonici/FFTLog.jl) and [FFTW](https://github.com/JuliaMath/FFTW.jl) in order to perform Fast Fourier Transforms on integrals containing Spherical Bessel functions $j_\ell(x)$
- [Dierckx](https://github.com/kbarbary/Dierckx.jl) and [GridInterpolations](https://github.com/sisl/GridInterpolations.jl) for 1D and 2D Splines respectively
- [LsqFit](https://github.com/JuliaNLSolvers/LsqFit.jl) for basic least-squares fitting
- [QuadGK](https://github.com/JuliaMath/QuadGK.jl), [Trapz](https://github.com/francescoalemanno/Trapz.jl) and [FastGaussQuadrature](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) for preforming 1D integrations, and [HCubature](https://github.com/JuliaMath/HCubature.jl) for the 2D ones
- [ArbNumerics](https://github.com/JeffreySarnoff/ArbNumerics.jl), [AssociatedLegendrePolynomials](https://github.com/jmert/AssociatedLegendrePolynomials.jl), [LegendrePolynomials](https://github.com/jishnub/LegendrePolynomials.jl) and [SpecialFunctions](https://github.com/JuliaMath/SpecialFunctions.jl) for mathematical function evaluations, especially for the Legendre Polinomials $\mathcal{L}_{\ell}(x)$ and the Gamma function $ \Gamma(x) $
- other native Julia packages: [DelimitedFiles](https://github.com/JuliaData/DelimitedFiles.jl), [Documenter](https://github.com/JuliaDocs/Documenter.jl), [IJulia](https://github.com/JuliaLang/IJulia.jl), [LinearAlgebra](https://github.com/JuliaLang/julia/tree/master/stdlib/LinearAlgebra), [NPZ](https://github.com/fhs/NPZ.jl), [Printf](https://github.com/JuliaLang/julia/tree/master/stdlib/Printf), [ProgressMeter](https://github.com/timholy/ProgressMeter.jl), [Suppressor](https://github.com/JuliaIO/Suppressor.jl), [Test](https://github.com/JuliaLang/julia/tree/master/stdlib/Test)


## How to report bugs, suggest improvements and/or contribute

As already mentioned above, this is a WIP project used mostly by the authors themselves, and so bugs are behind the corner. If you discover one of them, or if you would like to make a suggestion about a possible new feature that the code might implement, do not hesitate to contact the authors via email (<matteo.foglieni@lrz.de>) or fork the repository and open a pull request like follows:

- fork the project: on the top of the GaPSE.jl Github page, go to Fork > Create a new Fork
- download your forked repository from your GitHub profile
- create your branch: in the terminal, run `$ git checkout -b feature/<your-feature-name>`
- make the changes/improvements you want in that branch
- commit your changes in that branch: in the terminal, run `$ git commit -m 'added the feature <your-feature-name>'`
- push:  in the terminal, run `$ git push origin feature/<your-feature-name>`
- open a Pull Request for that branch


## Using this code

If you use GaPSE to compute the galaxy power spectrum/correlation function please refer to the two following papers:

- Castorina, Di Dio, _The observed galaxy power spectrum in General Relativity_ (2022), Journal of Cosmology and Astroparticle Physics, DOI: 10.1088/1475-7516/2022/01/061, url: https://doi.org/10.1088/1475-7516/2022/01/061

- Foglieni, Pantiri, Di Dio, Castorina,  _The large scale limit of the observed galaxy power spectrum_ (2023)

If you also use the code to compute the perturbations in the luminosity distance, please refer also to

- Pantiri, Foglieni, Di Dio, Castorina,  _The power spectrum of luminosity distance fluctuations in General Relativity_ (2023) [in preparation]

## Licence

This software is under the [GNU 3.0 General Public Licence](https://www.gnu.org/licenses/gpl-3.0.en.html). See the file [LICENCE.md](./LICENCE.md).

## References
<a id="1">[1]</a> 
Dalal, Doré et al., _Imprints of primordial non-Gaussianities on large-scale structure_ (2008), American Physical Society, DOI: 10.1103/PhysRevD.77.123514, 
url: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.77.123514

<a id="2">[2]</a> 
Slosar, Hirata et al., _Constraints on local primordial non-Gaussianity from large scale structure_ (2008), Journal of Cosmology and Astroparticle Physics, DOI: 10.1088/1475-7516/2008/08/031, url: https://doi.org/10.1088/1475-7516/2008/08/031

<a id="3">[3]</a> 
Castorina, Di Dio, _The observed galaxy power spectrum in General Relativity_ (2022), Journal of Cosmology and Astroparticle Physics, DOI: 10.1088/1475-7516/2022/01/061, url: https://doi.org/10.1088/1475-7516/2022/01/061

<a id="4">[4]</a> 
Di Dio, Montanari, _Curvature constraints from large scale structure_ (2016), Journal of Cosmology and Astroparticle Physics, DOI: 10.1088/1475-7516/2016/06/013, url: https://iopscience.iop.org/article/10.1088/1475-7516/2016/06/013

<a id="5">[5]</a>
Gebhardt, Jeong et al, _Fast and accurate computation of projected two-point functions_ (2018), American Physical Society, DOI: 10.1103/PhysRevD.97.023504, url: https://link.aps.org/doi/10.1103/PhysRevD.97.023504
