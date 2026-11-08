# WaveSim — Circular Drum Membrane Simulator

A Python-based scientific computing project that numerically simulates wave propagation and vibration on a circular drum membrane using the two-dimensional damped wave equation.

## Overview

WaveSim is a numerical simulation of a vibrating circular membrane. It uses finite-difference methods to approximate the propagation, reflection, and damping of mechanical waves across a two-dimensional domain.

The project combines numerical computation, scientific diagnostics, automated testing, and data visualization to study the dynamic behaviour of a circular membrane.

## Features

- Numerical solution of the 2D damped wave equation.
- Circular computational domain with fixed boundary conditions.
- Gaussian initial displacement.
- Explicit finite-difference time-stepping scheme.
- CFL stability condition validation.
- Configurable wave speed and damping coefficient.
- Three-dimensional animated membrane visualization.
- Two-dimensional displacement heatmap.
- Displacement and RMS analysis.
- Kinetic, potential, and approximate total energy diagnostics.
- Grid-refinement experiment.
- Automated numerical and software tests.

## Mathematical Model

The membrane vibration is governed by the two-dimensional damped wave equation:

```math
\frac{\partial^2 u}{\partial t^2}
+ 2\gamma\frac{\partial u}{\partial t}
= c^2\nabla^2u
```

Where:

- $u(x,y,t)$: Membrane displacement.
- $c$: Wave propagation speed.
- $\gamma$: Damping coefficient.
- $t$: Time.
- $\nabla^2$: Two-dimensional Laplacian operator.

### Computational Domain

The membrane occupies a circular region:

```math
x^2+y^2\leq R^2
```

Where $R$ is the membrane radius.

### Boundary Condition

The membrane has fixed boundaries:

```math
u(x,y,t)=0
```

At the computational boundary.

### Initial Conditions

The initial displacement is modelled using a Gaussian disturbance:

```math
u(x,y,0)=A\exp\left(
-\frac{(x-x_0)^2+(y-y_0)^2}{2\sigma^2}
\right)
```

Where:

- $A$: Initial disturbance amplitude.
- $x_0,y_0$: Disturbance location.
- $\sigma$: Gaussian spread.

The initial velocity is zero:

```math
\frac{\partial u}{\partial t}(x,y,0)=0
```

## Numerical Method

WaveSim uses a Cartesian computational grid and an explicit finite-difference scheme.

The spatial Laplacian is approximated using the five-point stencil:

```math
\nabla_h^2u_{i,j}=
\frac{u_{i+1,j}-2u_{i,j}+u_{i-1,j}}{\Delta x^2}
+
\frac{u_{i,j+1}-2u_{i,j}+u_{i,j-1}}{\Delta y^2}
```

The damped wave equation is discretized using finite differences in space and time.

The resulting time-stepping recurrence is:

```math
u_{i,j}^{n+1}=
\frac{
2u_{i,j}^{n}
-(1-\gamma\Delta t)u_{i,j}^{n-1}
+c^2\Delta t^2\nabla_h^2u_{i,j}^{n}
}{
1+\gamma\Delta t
}
```

The implementation uses the interior computational mask to update active points while maintaining zero displacement at fixed boundary points.

### CFL Stability Condition

For the explicit two-dimensional wave scheme, the time step is selected to satisfy:

```math
c\Delta t
\sqrt{
\frac{1}{\Delta x^2}
+
\frac{1}{\Delta y^2}
}
\leq 1
```

A configurable safety factor is used to select a time step below the standard CFL limit.

## Project Structure

```text
wavesim/
│
├── main.py
├── config.py
├── requirements.txt
├── .gitignore
│
├── simulation/
│   ├── __init__.py
│   ├── grid.py
│   ├── initial_conditions.py
│   ├── solver.py
│   └── diagnostics.py
│
├── visualization/
│   ├── __init__.py
│   ├── animation.py
│   ├── heatmap.py
│   └── plots.py
│
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_convergence.py
    ├── test_diagnostics.py
    ├── test_grid.py
    ├── test_initial_conditions.py
    ├── test_solver.py
    └── test_visualization.py
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Simulation execution and integration |
| `config.py` | Simulation parameters and validation |
| `simulation/grid.py` | Cartesian grid and circular domain generation |
| `simulation/initial_conditions.py` | Gaussian initial displacement |
| `simulation/solver.py` | Finite-difference wave equation solver |
| `simulation/diagnostics.py` | Displacement and energy measurements |
| `visualization/animation.py` | 3D membrane animation |
| `visualization/heatmap.py` | 2D displacement heatmap |
| `visualization/plots.py` | Displacement and energy plots |
| `tests/` | Automated numerical and software validation |

## Technologies Used

- **Python** — Core programming language.
- **NumPy** — Numerical computation and array operations.
- **Matplotlib** — Scientific visualization, plotting, and animation.
- **unittest** — Automated testing.

## Installation

### Prerequisites

- Python 3.14.7 or a compatible Python environment.
- Git.
- pip.

### 1. Clone the repository

```bash
git clone https://github.com/AnuSaha545/wavesim-2d-membrane-simulator.git
cd wavesim-2d-membrane-simulator
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Simulation

Execute the main application:

```bash
python main.py
```

The application initializes the computational domain, generates the initial disturbance, calculates a suitable time step, and executes the simulation.

It performs numerical validation and generates the displacement analysis, energy analysis, and membrane visualization.

## Configuration

Simulation parameters can be modified in `config.py`.

| Parameter | Description | Default |
|---|---|---:|
| `NX` | Number of grid points along X | 101 |
| `NY` | Number of grid points along Y | 101 |
| `DOMAIN_LENGTH` | Computational domain length | 2.0 |
| `MEMBRANE_RADIUS` | Circular membrane radius | 1.0 |
| `WAVE_SPEED` | Wave propagation speed | 1.0 |
| `DAMPING` | Damping coefficient | 0.35 |
| `AMPLITUDE` | Initial displacement amplitude | 0.5 |
| `STRIKE_X` | Initial disturbance X-coordinate | 0.0 |
| `STRIKE_Y` | Initial disturbance Y-coordinate | 0.0 |
| `SIGMA` | Gaussian disturbance spread | 0.15 |
| `CFL_SAFETY` | CFL safety factor | 0.9 |
| `SIMULATION_DURATION` | Total simulation duration | 5.0 |

Changing these parameters allows different membrane responses to be investigated.

## Diagnostics and Visualization

WaveSim provides several outputs for analysing membrane behaviour.

### 1. Three-Dimensional Membrane Animation

Displays the membrane displacement as a three-dimensional surface evolving over time.

### 2. Two-Dimensional Displacement Heatmap

Represents positive and negative displacement across the circular membrane using a colour map.

### 3. Displacement Analysis

Plots maximum amplitude and RMS displacement against simulation time.

### 4. Mechanical Energy Analysis

Plots kinetic energy, potential energy, and approximate total mechanical energy over time.

The energy diagnostics are reported in relative model units rather than joules.

## Numerical Validation

The implementation includes automated tests and numerical experiments to evaluate solver behaviour.

### Solver Testing

The solver test suite checks:

- Zero-displacement stability.
- Wave propagation.
- Fixed boundary conditions.
- Circular domain enforcement.
- Finite numerical values.
- Damping behaviour.
- CFL violation rejection.
- Long-duration numerical stability.
- Invalid grid rejection.

### Grid-Refinement Experiment

The numerical solution is evaluated at different grid resolutions:

- 51 × 51
- 101 × 101
- 201 × 201

The RMS difference between solutions at successive resolutions is calculated over comparable membrane points.

Observed results:

| Grid comparison | RMS difference |
|---|---:|
| 51 vs 101 | 0.00358822 |
| 101 vs 201 | 0.00183280 |

The measured difference decreases with grid refinement. This is encouraging numerical evidence, but does not establish a formal convergence order.

### Main Simulation Results

Configuration:

| Parameter | Value |
|---|---:|
| Grid resolution | 101 × 101 |
| Membrane radius | 1.0 |
| Wave speed | 1.0 |
| Damping coefficient | 0.35 |
| Simulation duration | 5.0 s |
| Time step | 0.012723 s |
| Number of time steps | 393 |
| CFL number | 0.8996 |

Validation results:

| Validation | Result |
|---|---|
| CFL stability condition | Passed |
| Fixed boundary conditions | Passed |
| Circular domain conditions | Passed |
| Finite displacement values | Passed |
| Final displacement validation | Passed |

Energy diagnostics:

| Measurement | Result |
|---|---:|
| Initial approximate energy | 0.39043206 |
| Final approximate energy | 0.01148443 |
| Energy change | -97.06% |

The reduction in approximate mechanical energy is consistent with the damping specified in the governing equation.

## Running the Tests

Run the solver tests:

```bash
python -m tests.test_solver
```

Run the grid-refinement experiment:

```bash
python -m tests.test_convergence
```

Run the complete test suite using Python's unittest discovery:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Numerical Limitations

The current implementation has the following limitations:

1. **Cartesian boundary approximation:** The circular membrane is represented using a Cartesian grid and domain masks. Consequently, the circular boundary is approximated rather than represented using an exact curved-boundary numerical method.

2. **Spatial discretization:** The five-point finite-difference stencil introduces spatial discretization error.

3. **Time discretization:** The explicit time-stepping method introduces temporal discretization error.

4. **Energy diagnostics:** The reported energy is an approximate discrete diagnostic in relative model units, not a direct measurement of physical energy in joules.

5. **Convergence assessment:** The grid-refinement experiment demonstrates decreasing measured differences but does not independently establish a formal convergence order.

These limitations should be considered when interpreting simulation results.

## Future Research Directions

Potential extensions include:

- Higher-order spatial discretization.
- More accurate curved-boundary numerical methods.
- Analytical modal solutions for circular membranes.
- Quantitative comparison with theoretical vibration modes.
- Improved energy-stability analysis.
- Parameter sensitivity studies.
- Performance optimization for higher-resolution simulations.

These are potential extensions and are not claimed as implemented features.

## Author

**Anu Saha**

GitHub: [AnuSaha545](https://github.com/AnuSaha545)

## License

A license has not yet been specified for this repository. Add an appropriate license before permitting reuse or redistribution of the project.
