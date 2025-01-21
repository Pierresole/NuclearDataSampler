# NuclearDataSampler

Welcome to **NuclearDataSampler**, a Python-based code aiming to randomly sample evaluated nuclear data files (ENDF) under well-defined, minimal assumptions. This project is part of an effort to explore and compare different approaches to uncertainty quantification (UQ) in nuclear data, including classic sensitivity-based methods and the more direct “Total Monte Carlo” (TMC) approach. 
This project was made possible by the efficient handling of nuclear data files using ENDFtk and its C++/Python bindings.

---

## :one: Dependencies and Installation

This project relies on **ENDFtk** for reading and writing ENDF files:

- [ENDFtk GitHub Repository](https://github.com/njoy/ENDFtk)

Before installing **NuclearDataSampler**, you should ensure that **ENDFtk** is installed and available in your environment. Please see the [ENDFtk repository](https://github.com/njoy/ENDFtk) for instructions on building or installing ENDFtk.

### Installation

You can install **NuclearDataSampler** via pip:

```bash
pip install NuclearDataSampler
```

If you want to actively develop or contribute to the project, clone this repository and install in “editable” mode:

```bash
git clone https://github.com/Pierresole/NuclearDataSampler.git
cd NuclearDataSampler
pip install -e .
```
This will let you edit the code locally and directly test your changes without reinstalling.

---

## :two: Progress Overview

| Perturbed Parameters         | Status            | Comment |
|---                           |           :---:   |---      |
| Thermal Parameters           | :white_check_mark:| LEAPR inputs|
| Resonance Parameters         |                   |         |
| - URR                        | :white_check_mark:|         |
| - MLBW                       |                   |         |
| - RM                         | :white_check_mark:|         |
| - RML                        | :white_check_mark:|         |
| Cross Sections (groupwise)   | :x:               | Interpolation(1) |

(1) To perturb a MF3 based on its MF33, it is necessary to create an easily interpolable XS. This has been done. What is left is to think of the code organization and the way of perturbing composed cross sections.

---

## :three: The Core Idea of NuclearDataSampler

1. **Input**: You provide an ENDF file that contains both the nominal (mean) parameter values and the associated covariance matrix. 
2. **Sampling**: We draw random samples from the **multivariate Gaussian distribution** parameterized by that mean vector and covariance matrix—no additional re-interpretation or modeling assumptions are introduced.
3. **Output**: Each random draw updates the relevant sections of the ENDF file, producing a new, consistent ENDF file that reflects one realization of the underlying uncertainties.

By construction, this approach works at the evaluated nuclear data level, avoiding additional data-format conversions or embedded nuclear reaction model assumptions. This makes it simple to compare with or feed into other downstream codes.

> **Note**: A mean vector and a covariance matrix uniquely define a (multivariate) Gaussian distribution. By specifying only these two ingredients, we are implicitly stating that uncertainties follow a normal distribution in parameter space. Any more complicated shape would require higher-order moments or parametric expansions, and at least something to verify our hypothesis or some guidance from evaluated distributions.

Many researchers in the field of UQ have acknowledged the effectiveness of using of LHS sampling, which we have applied in NDSampler and LEAPRSampler.

---

## :four: Context

### UQ in Particle Transport Physics

Uncertainty Quantification (UQ) in particle transport typically involves answering the question: “How do the uncertainties in nuclear cross sections, resonance parameters, and other fundamental data propagate to engineering or physics parameters of interest (e.g., reaction rates, keff in reactor calculations, neutron and neutrinos flux spectrum)?” 

Two common strategies to address this question are:
1. **Sensitivity-based approaches**: Perturb the inputs systematically using partial derivatives or adjoint solutions to infer the output uncertainties. These methods often rely on linear approximations and can have difficulties with strongly non-linear or resonance-dominated processes.
2. **Total Monte Carlo (TMC)**: Re-sample the nuclear data themselves many times (often from an assumed probability distribution, typically a multivariate Gaussian) and compute the transport problem to statistically evaluate the distribution of the output. This direct approach is computationally more intensive, but it does not rely on linearity assumptions or pre-computed sensitivities.

**NuclearDataSampler** focuses on the TMC idea: given an ENDF file and its uncertainty information (covariances), produce randomized ENDF files ready for downstream usage—without imposing any additional assumptions or re-interpretations.


## State of the Art and Motivation

Several tools exist for generating or processing perturbed nuclear data files, each with its own set of assumptions and workflows. For example:

- **TALYS** (1) 
  A comprehensive nuclear reaction model code that can generate cross sections, angular distributions, and other observables for many projectiles, targets, and reaction channels. TALYS includes various nuclear structure models (optical models, level densities, gamma strength functions, etc.). One typical usage is to sample model parameters (e.g., nuclear level density parameters, optical model potential parameters) multiple times, generate numerous “realizations” of cross sections, and compare these with experimental data at certain steps. This effectively yields an ensemble of cross-section evaluations.

- **SANDY** (2) 
  A Python package focused on sampling and analyzing nuclear data uncertainties. It can produce perturbed nuclear data in PENDF formats using the processing code NJOY.

- **FRENDY** (3) 
  A data processing system designed to read, process, and produce ACE-formatted data from evaluated nuclear data libraries. FRENDY also provides modules to handle uncertainties.

Despite these codes’ capabilities, many times a user just needs a straightforward way to:
1. Take an existing ENDF file that comes with mean values and a covariance matrix.
2. Sample new sets of evaluated data from a well-defined **multivariate Gaussian distribution** (mean vector + covariance matrix).
3. Output new ENDF files with minimal additional assumptions or format transformations.

That is exactly what **NuclearDataSampler** aims to do. 

What motivated this code is a simple but faithful treatment of resonance parameters (RP) uncertainty that are lumped into group cross sections (XS), especially in SANDY and FRENDY. A very foundational code to study the effect of lumping RP uncertainties into group XS uncertainties is **ENDSAM** (4) developed at JSI. 

ENDSAM is able to generate random files but was primarily developed to check whether the relative uncertainty of certain parameters is too high, and if so, verify if their covariance matrix is mathematically correct (log-normal transformation). 

The conclusions drawn from ENDSAM results and the discussions it triggered in the community (5) led to **NuclearDataSampler** to avoid backend interpretation. Namely if positive parameters are sampled negatively, any backend patch will not be verifiably correct without access to the statistical evaluated distribution. However, if the patch is "acceptable," it may be applied. Similarly, if the covariance matrix is not positive definite and the problematic eigenvalues are significantly negative, one should not clip them to a positive value.

> **Keep cool and call an evaluator**
 
Diagnosing and communicating such issues is essential for the evaluation of codes. Conversely, "screwdriver-ing" or "quick fixes" to make things work can obscure more serious underlying problems.

*References :link: :*

(1) Koning, A.J., Hilaire, S., and Goriely, S., NRG CEA ULB and IAEA [TALYS Repository](https://nds.iaea.org/talys/). 

(2) Fiorito, L., SCK-CEN, [SANDY Repository](https://github.com/luca-fiorito-11/sandy).

(3) Tada, K., JAEA, [FRENDY Repository](https://rpg.jaea.go.jp/main/en/program_frendy/).  

(4) Plevnik, L. and Žerovnik, G., JSI, *"Computer code ENDSAM for random sampling and validation of the resonance parameters covariance matrices of some major nuclear data libraries"*, Annals of Nuclear Energy, 2016.  
DOI: [10.1016/j.anucene.2016.04.026](https://doi.org/10.1016/j.anucene.2016.04.026)

(5) Taavitsainen, A. and Vanhanen, R., *"On the maximum entropy distributions of inherently positive nuclear data"*, 2017, Aalto University School of Science, Finland. DOI: [10.1016/j.nima.2016.11.061](https://doi.org/10.1016/j.nima.2016.11.061)

---

## The (Multivariate) Gaussian Assumption

When we talk about a **mean vector** $\mu$ and a **covariance matrix** $\Sigma$, we are specifying a **multivariate Gaussian (normal) distribution**:


$p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det(\Sigma)}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu)\right).$

- $\mu \in \mathbb{R}^n$ is the vector of means for the $n$ parameters.
- $\Sigma \in \mathbb{R}^{n \times n}$ is the covariance matrix describing pairwise correlations between parameters.

By definition, only specifying $\mu$ and $\Sigma$ means that the distribution is **exactly Gaussian**. Any higher-order "shape" information (e.g., skewness, kurtosis, etc.) is zero for a perfect Gaussian. 

If the physical reality demands more complex distributions, we must add more parameters or move beyond the Gaussian assumption. **NuclearDataSampler** is developed to be flexible enough to be used with advanced relationship models like Copulas. Sampling from more complicated dependencies and laws is straightforward in Python, but the first move should come from evaluation codes, which should be able to communicate the parameters' distributions.

---

## :five: Contributing :construction_worker:
Contributions are welcome—whether it’s adding new features, fixing bugs, or improving documentation. 

There is still work to be done to address the six types of uncertainties present in nuclear data files: neutron multiplicities, resonance parameters, multigroup cross sections, angular distributions, energy distributions, and fission spectra.

Feel free to explore and adapt the TMC approach with this simple sampling tool. We hope it simplifies your workflows and fosters further comparisons with other uncertainty quantification methods!

&copy; 2025 NuclearDataSampler Contributors