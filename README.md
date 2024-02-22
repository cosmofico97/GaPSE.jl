# GaPSE - a model for the Galaxy Power Spectrum Estimator

![julia-version](https://img.shields.io/badge/julia_version-v1.8-9558B2?style=flat&logo=julia) 
![package-version](https://img.shields.io/github/v/release/foglienimatteo/GaPSE.jl?include_prereleases)
![CI-build](https://img.shields.io/github/actions/workflow/status/foglienimatteo/GaPSE.jl/UnitTests.yml)
![size](https://img.shields.io/github/repo-size/foglienimatteo/GaPSE.jl) 
![license]( https://img.shields.io/github/license/foglienimatteo/GaPSE.jl)
[![codecov](https://codecov.io/gh/foglienimatteo/GaPSE.jl/branch/main/graph/badge.svg?token=67GIZ9RA8Y)](https://codecov.io/gh/foglienimatteo/GaPSE.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://foglienimatteo.github.io/GaPSE.jl/stable) 

GaPSE (Galaxy Power Spectrum Estimator) is a software for cosmological computations written in the [Julia Programming Language](https://julialang.org).

<<<<<<< HEAD
IMPORTANT NOTE: This is a work-in-progress project! As a consequence, currently in this pre-release:
- it is possible to compute the power spectrum/correlation function multipoles with L=1,2,3,... of the effects we'll show next, but 2 effects among the Galaxy Number Counts multipoles (Newton-Lensing and Lensing-Newton) converge very slowly, so their computation is not still 100% ready. However, the monopole (L=0) computations do not have any problem with `quad`, and even the GNC sum for higher order multipoles is not affected;
- The Power Spectrum computations with `:twofast` do not work properly, you should always prefer `:fftog`. However, due to the fact that with `:fftlog` you must specify manually the bias parameter, the Power Spectra of a whole group of terms creates FFT oscillations in the smallest ones. The leading ones and the sum are not however affected.  
- the code functions are well documented; check the github pages website https://foglienimatteo.github.io/GaPSE.jl/stable if you can't see correctly the analytical expressions written in the docstrings; 
- few people used this code, so bugs are behind the corner; do not hesitate to raise the finger to point out them (see in the [How to report bugs, suggest improvements and/or contribute](#how-to-report-bugs-suggest-improvements-andor-contribute) section below)!
- if you use this code, please read the [Using this code](##using-this-code) section below
=======
## Important Remarks

This is a work-in-progress project! As a consequence, currently in this pre-release:
- The monopole (L=0) computations of the Galaxy Number Counts (GNC) correlation functions with `quad` integration and their corresponding Power Spectra (PS) with `:fftlog` have been tested extensively. It is also possible to compute their correlation function multipoles $L=1,2,3,...$  with `:lobatto` integration. However, `:lobatto` still leads to small numerical oscillations, and two terms (Newton-Lensing and Lensing-Newton) converge very slowly (i.e. you must use a high `N_lob` number of points). These problems are nevertheless suppresed when you employ the azymuthally symmetric window function of this code (`use_windows=true`) and consider the power spectrum sum of all these effects;
- you cannot go further than $z \simeq 1.5$;
- The PS computations are by default made with the Julia package [FFTLog](https://github.com/marcobonici/FFTLog.jl). Due to the fact that with `:fftlog` you must specify manually the bias parameter, the Power Spectra computations of a whole group of terms create artificial FFT oscillations in some of them. However, the leading ones and the sum are not affected. The PS code structure gives also the option to implement other routines for this computation, and one with [TwoFAST](https://github.com/hsgg/TwoFAST.jl)[[5]](#1) is sketched (but not yet tested and reliable);
- the code functions are well documented; check the github pages website https://foglienimatteo.github.io/GaPSE.jl/stable if you cannot see correctly the analytical expressions written in the docstrings. The Two-Point Correlation Functions docstrings of the groups `LD`, `GNCxLD` and `LDxGNC` (see below for explanation) are still missing; 
- few people used this code, so bugs are behind the corner; do not hesitate to raise the finger to point them out (see in the [How to report bugs, suggest improvements and/or contribute](#how-to-report-bugs-suggest-improvements-andor-contribute) section below)!
- if you use this code, please read the [Using this code](##using-this-code) section below.
>>>>>>> main


## Table of Contents

- [GaPSE - a model for the Galaxy Power Spectrum Estimator](#gapse---a-model-for-the-galaxy-power-spectrum-estimator)
  - [Important Remarks](#important-remarks)
  - [Table of Contents](#table-of-contents)
  - [Brief description](#brief-description)
  - [Installation and Usage](#installation-and-usage)
    - [traditional way: Installation](#traditional-way-installation)
    - [traditional way: Usage](#traditional-way-usage)
    - [Docker container: Installation and Usage](#docker-container-installation-and-usage)
  - [Dependencies](#dependencies)
  - [How to report bugs, suggest improvements and/or contribute](#how-to-report-bugs-suggest-improvements-andor-contribute)
  - [Using this code](#using-this-code)
  - [Licence](#licence)
  - [References](#references)


## Brief description

Measurements of the clustering of galaxies in Fourier space, at low wavenumbers, offer a window into the early Universe via the possible presence of scale dependent bias generated by Primordial Non Gaussianities [[1]](#1) [[2]](#1).
On such large scales, a Newtonian treatment of density perturbations might not be sufficient to describe the measurements, and a fully relativistic calculation should be employed.

Given the matter Power Spectrum (PS) at redshift $z=0$ and the background quantities for the Universe considered (both read from [CLASS](https://github.com/lesgourg/class_public) outputs), this code can compute:

- all the 16 Two-Point Correlation Functions (TPCFs) arising from the Luminosity Distance (LD) perturbations (see Eq.(2.48) of [[4]](#1)) for an arbitrary multipole order;

- all the 25 TPCFs concerning the relativistic Galaxy Number Counts (GNC)  (see Eq.(2.52) of [[4]](#1)) for an arbitrary multipole order;

- all the 20 cross correlations between GNC and LD (and their 20 counterparts LD cross GNC) for an arbitrary multipole order;

- the PS multipoles of all of them (based on the Yamamoto estimator);

- the Doppler and matter TPCFs in the Plane-Parallel (PP) approximation;

All these calculations can be performed both with and without an azymuthally symmetric window function, that the code allows to compute for a given redshift bin and angular opening.
The multipole computations are tested for $0 \leq L \leq 4$.

This project, and the analytical expressions used for the TPCFs, are based on the paper by Emanuele Castorina and Enea Di Dio [[3]](#1). 

## Installation and Usage

<<<<<<< HEAD
Currently, this package is not in the Julia package registries. 
There are two main ways to install and use GaPSE on your local machine:
-  the traditional way: you clone this gitrepo locally and you install the librarires that GaPSE needs in a suited Julia enviroment; it requires a compatible Julia version ≥1.8;
-  using a Docker container (experimental): you pull and run the GaPSE container; it requires a [Docker](https://www.docker.com) installation.


### traditional way: Installation

Assuming that you have already installed a coompatible Julia version, the simplest way to install this software is then the following:

=======
Currently, this package is not in the Julia package registries. Assuming that you have already installed a compatible Julia version (1.8.x), the simplest way to install this software is then the following:

>>>>>>> main
- in the terminal, go to the directory where you want to install this package;
  
- clone this repository with Git
  ```bash
  $ git clone https://github.com/foglienimatteo/GaPSE.jl
  ```
  or manually download the source code from the url https://github.com/foglienimatteo/GaPSE.jl (Code > Download Zip);

- go inside the directory of GaPSE (`$ cd GaPSE.jl` in the shell).

Inside the directory, there is a file called `install_gapse.jl`, which is a Julia script conceived for downloading and installing all the dependencies of GaPSE. You can run it by typing in the terminal:

```bash
     $ julia install_gapse.jl
```

If there are no error messages at the end of the installation, then GaPSE is corretly configured and you can start to use it!
  
NOTE: the packages that this script will install are the ones strictly required for GaPSE. The ipynbs we provide need however some more, as `Plots`, `LaTeXStrings` and `PyPlot` (which in turn requires a python kernel with `Matplotlib` installed); in case you don't have them, run with a terminal in this directory:
```bash
$ pip3 install matplotlib
$ julia --activate=. --eval 'using Pkg; for p in ["Plots", "LaTeXStrings", "PyPlot"]; Pkg.add(p); end; Pkg.resolve()'
```

NOTE: instead of using the `install_gapse.jl` script, you can also do the same in a more interactive way, if you prefer:

- open a Julia REPL session and activate the project; you can achieve that opening the REPL with 
  ```bash
  $ julia --activate=.
  ```
  or directy inside the REPL entering the Pkg mode (`]`) and running `activate .`;

- enter the Pkg mode (if you haven't done in the previous step) typing `]` and run `instantiate`; this command will automatically detect and install all the package dependecies (listed in `Project.toml`);

- done! You can exit from the package mode (press the Backspace key on an empty line) and start to use GaPSE.


### traditional way: Usage

There are three ways in order to use this code:

- you can write whatever instruction inside the file `GaPSE-exe.jl` and then run in the command line
  ```bash
    $ julia GaPSE-exe.jl
  ```

- you can open a Julia REPL session, include the code with
  ```julia
     include("<path-to-GaPSE.jl-directory>/src/GaPSE.jl")
  ```
  and then use interactively the GaPSE functions;

- you can run the same `include("<path-to-GaPSE.jl-directory>/src/GaPSE.jl")` command in a Jupyter Notebook, and use the code functions inside it. This is by far the most confortable way.

Some `.ipynb`s are already provided in the directory `ipynbs` :
- we encourage you to follow the `ipynbs/TUTORIAL.ipynb` file first. The basic structure of the code and the most important functions are there presented;
- `ipynbs/Computations_b1p5-sb0-fevo0.ipynb` explains the analytical Primordial Non-Gaussianities model we use here, compute its contribution in the redshift bin $1.0 \leq z \leq 1.5$ and compare it with the GNC effects, all using our toy-model window function with angular opening $\theta_{\rm max} = \pi/2$;
- `ipynbs/Generic_Window.ipynb` explains how to use GaPSE with a generic Window Function of your choice;
- `ipynbs/eBOSS_Window.ipynb` apply GaPSE on a real case scenario: the eBOSS window function;
- the `ipynbs/Computations_b1p5-sb0-fevo0.jl` Julia file its the translation into script of `ipynbs/Computations_b1p5-sb0-fevo0.ipynb`; you can easily run it from the command line with:
  ```bash
    $ julia <scriptname.jl>
  ```
  The other scripts`ipynbs/PS_L01234_pt0.jl`, ..., `ipynbs/PS_L01234_pt0.jl` allow you to speed up the `ipynbs/PS_L01234.jl` computations, running them in different terminals.

The code is well tested and documented: almost each struct/function has a docstring that you can easily access in Julia with `?<name-of-the-struct/function>`, and there is an active GitHub Pages website with the [latest stable documentation](https://foglienimatteo.github.io/GaPSE/stable).

### Docker container: Installation and Usage

The `Dockerfile` we provide in this directory is the one we used to create the container image corresponding to this GaPSE version.

The images are saved in https://hub.docker.com/repository/docker/matteofoglieni/gapse/general and the tag is the same as the GaPSE version the container refers to + a latin letter (alphabetically orderer), to take into account different version of the Dockerfile which refer to the same GaPSE one.
The latest container name is then `gapse:0.8.0a`.

These containers have already installed all the Julia packages that GaPSE needs (i.e. the ones listed in `Project.toml`) + come others for the ipynbs (check the Dockerfile itself).

Supposing that you have already installed Docker, so as to use GaPSE as a container:
- download the image: 
  ```bash
  $ sudo docker pull matteofoglieni/gapse:0.8.0a
  ```
- choose a free port where to access the JupyterLab of the container; we will use `10000`;
- run the container with that port:
  ```bash
  $ sudo docker run -d -p 10000:8888 matteofoglieni/gapse:0.8.0a
  ```
- get the logs of the container and copy the Jupyter token (in the following output is `531vbeb08567581944e486d47e1tee15683757086205da68`):
  ```bash
  $ sudo docker logs $(sudo docker ps -ql)
  ...
  [I 2023-09-13 12:51:47.960 ServerApp] Jupyter Server 2.7.0 is running at:
  [I 2023-09-13 12:51:47.960 ServerApp] http://7b1ca9747263:8888/lab?token=531vbeb08567581944e486d47e1tee15683757086205da68
  [I 2023-09-13 12:51:47.960 ServerApp]     http://127.0.0.1:8888/lab?token=531vbeb08567581944e486d47e1tee15683757086205da68
  [I 2023-09-13 12:51:47.960 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
  [C 2023-09-13 12:51:47.962 ServerApp] 
      
      To access the server, open this file in a browser:
          file:///home/jovyan/.local/share/jupyter/runtime/jpserver-7-open.html
      Or copy and paste one of these URLs:
          http://7f1ca9847263:8888/lab?token=531vbeb08567581944e486d47e1tee15683757086205da68
          http://127.0.0.1:8888/lab?token=531vbeb08567581944e486d47e1tee15683757086205da68
  [I 2023-09-13 12:51:48.653 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server
  ...
  ```
- open a browser of your choice on your local machine and paste as URL  `http://127.0.0.1:10000/lab?token=531vbeb08567581944e486d47e1tee15683757086205da68` (essentially, you just have to substitute the port `8888` with the one you choose in the URL showed in the previous logs).
- now you should see a JupyterLab page containing the GaPSE files.

Now you can use GaPSE inside the Jupyter interface as normally!

Quick summary of Docker commands, in case you don't know them:
- `sudo docker ps [-a]` : list running containers (all with `-a`);
- `sudo docker logs <container-name/id>` : get the logs of a container;
- `sudo docker start/stop <container-name/id>` : start a stopped container/stop a running container;
- `sudo docker rm <container-id/name>` : delete a container;
- `sudo docker pull <image>` : download a container image;
- `sudo docker run <image>` : create a running container from an image;
- `sudo docker image list` : list all the local images;
- `sudo docker image rm <image>` : delete an image.



## Dependencies

GaPSE.jl makes extensive use of the following packages:

- [TwoFAST](https://github.com/hsgg/TwoFAST.jl)[[5]](#1), [FFTLog](https://github.com/marcobonici/FFTLog.jl) and [FFTW](https://github.com/JuliaMath/FFTW.jl) in order to perform Fast Fourier Transforms on integrals containing Spherical Bessel functions $j_\ell(x)$;
- [Dierckx](https://github.com/kbarbary/Dierckx.jl) and [GridInterpolations](https://github.com/sisl/GridInterpolations.jl) for 1D and 2D Splines respectively;
- [LsqFit](https://github.com/JuliaNLSolvers/LsqFit.jl) for basic least-squares fitting;
- [QuadGK](https://github.com/JuliaMath/QuadGK.jl), [Trapz](https://github.com/francescoalemanno/Trapz.jl) and [FastGaussQuadrature](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) for preforming 1D integrations, and [HCubature](https://github.com/JuliaMath/HCubature.jl) for the 2D ones;
- [ArbNumerics](https://github.com/JeffreySarnoff/ArbNumerics.jl), [AssociatedLegendrePolynomials](https://github.com/jmert/AssociatedLegendrePolynomials.jl), [LegendrePolynomials](https://github.com/jishnub/LegendrePolynomials.jl) and [SpecialFunctions](https://github.com/JuliaMath/SpecialFunctions.jl) for mathematical function evaluations, especially for the Legendre Polinomials $\mathcal{L}_{\ell}(x)$ and the Gamma function $\Gamma(x)$;
- other native Julia packages: [DelimitedFiles](https://github.com/JuliaData/DelimitedFiles.jl), [Documenter](https://github.com/JuliaDocs/Documenter.jl), [IJulia](https://github.com/JuliaLang/IJulia.jl), [LinearAlgebra](https://github.com/JuliaLang/julia/tree/master/stdlib/LinearAlgebra), [NPZ](https://github.com/fhs/NPZ.jl), [Printf](https://github.com/JuliaLang/julia/tree/master/stdlib/Printf), [ProgressMeter](https://github.com/timholy/ProgressMeter.jl), [Suppressor](https://github.com/JuliaIO/Suppressor.jl), [Test](https://github.com/JuliaLang/julia/tree/master/stdlib/Test).

Furthermore, the notebooks we provide in `ipynbs` use:
- [Plots](https://github.com/JuliaPlots/Plots.jl) for the pure julian plots;
- [LaTeXStrings](https://github.com/JuliaStrings/LaTeXStrings.jl) for the labels in LaTeX;
- [PyPlot](https://github.com/JuliaPy/PyPlot.jl) for the julian plots in the python style; this package is based on the [Matplotlib](https://matplotlib.org) Python package, and it requires it in order to run properly.


## How to report bugs, suggest improvements and/or contribute

As already mentioned above, this is a WIP project used mostly by the authors themselves, so bugs are behind the corner. If you discover one of them, or if you would like to make a suggestion about a possible new feature that the code might implement, do not hesitate to contact the authors via email (<matteo.foglieni@lrz.de>) or fork the repository and open a pull request as follows:

- fork the project: on the top of the GaPSE.jl Github page, go to Fork > Create a new Fork;
- download your forked repository from your GitHub profile;
- create your branch: in the terminal, run `$ git checkout -b feature/<your-feature-name>`;
- make the changes/improvements you want in that branch;
- commit your changes in that branch: in the terminal, run `$ git commit -m 'added the feature <your-feature-name>'`;
- push:  in the terminal, run `$ git push origin feature/<your-feature-name>`;
- open a Pull Request for that branch.


## Using this code

If you use GaPSE to compute the galaxy power spectrum/correlation function please refer to the two following papers:

- Castorina, Di Dio, _The observed galaxy power spectrum in General Relativity_ (2022), Journal of Cosmology and Astroparticle Physics, DOI: [10.1088/1475-7516/2022/01/061](https://doi.org/10.1088/1475-7516/2022/01/061) (arXiv [2106.08857](https://arxiv.org/abs/2106.08857))

- Foglieni, Pantiri, Di Dio, Castorina,  _The large scale limit of the observed galaxy power spectrum_ (2023), Physical Review Letters, DOI: [10.1103/PhysRevLett.131.111201](https://doi.org/10.1103/PhysRevLett.131.111201) (arXiv [2303.03142](https://arxiv.org/abs/2303.03142))

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
