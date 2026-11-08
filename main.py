
import numpy as np
import matplotlib.pyplot as plt

from config import (
    NX, NY, DOMAIN_LENGTH, MEMBRANE_RADIUS,
    WAVE_SPEED, DAMPING, AMPLITUDE,
    STRIKE_X, STRIKE_Y, SIGMA, CFL_SAFETY,
    SIMULATION_DURATION, ANIMATION_INTERVAL,
    validate_config
)

from simulation.grid import create_grid
from simulation.initial_conditions import create_initial_displacement
from simulation.solver import (
    WaveSolver,
    calculate_cfl_number,
    calculate_time_step,
)
from simulation.diagnostics import calculate_diagnostics

from visualization.animation import animate_membrane
from visualization.plots import plot_amplitude, plot_energy
from visualization.heatmap import plot_heatmap


def main():

    # Step 1: Validate configuration
    validate_config()

    # Step 2: Create circular computational grid
    X, Y, dx, dy, domain_mask, interior_mask = create_grid(
        NX, NY, DOMAIN_LENGTH, MEMBRANE_RADIUS
    )

    # Step 3: Generate initial displacement
    initial_displacement = create_initial_displacement(
        X, Y, AMPLITUDE, STRIKE_X, STRIKE_Y,
        SIGMA, interior_mask
    )

    # Step 4: Calculate stable time step
    dt, steps = calculate_time_step(
        dx,
        dy,
        SIMULATION_DURATION,
        WAVE_SPEED,
        CFL_SAFETY,
    )

    cfl_number = calculate_cfl_number(dx, dy, dt, WAVE_SPEED)

    # Step 5: Display simulation parameters
    print("\n========== WAVESIM ==========")
    print("Circular Drum Membrane Simulator")
    print("==============================")
    print(f"Grid resolution : {NX} x {NY}")
    print(f"Membrane radius : {MEMBRANE_RADIUS}")
    print(f"Spatial step dx : {dx:.5f}")
    print(f"Spatial step dy : {dy:.5f}")
    print(f"Time step       : {dt:.6f}")
    print(f"Wave speed      : {WAVE_SPEED}")
    print(f"Damping         : {DAMPING}")
    print(f"Duration        : {SIMULATION_DURATION}s")
    print(f"Active points   : {np.count_nonzero(interior_mask)}")
    print(f"CFL number      : {cfl_number:.4f}")
    print(f"Time steps      : {steps}")
    print("==============================")

    # Step 6: Validate numerical stability
    print("\n--- Numerical Validation ---")

    if cfl_number <= 1.0:
        print("PASS: CFL stability condition")
    else:
        raise ValueError("FAIL: Unstable time step!")

    # Step 7: Initialize solver
    solver = WaveSolver(
        initial_displacement,
        dx, dy, dt,
        WAVE_SPEED, DAMPING,
        domain_mask, interior_mask
    )

    # Step 8: Validate boundary conditions
    boundary_mask = domain_mask & ~interior_mask

    boundary_error = np.max(np.abs(solver.u[boundary_mask]))
    outside_error = np.max(np.abs(solver.u[~domain_mask]))

    print(f"Maximum boundary displacement: {boundary_error:.8f}")
    print(f"Maximum outside displacement: {outside_error:.8f}")

    if boundary_error < 1e-12 and outside_error < 1e-12:
        print("PASS: Fixed boundary conditions")
        print("PASS: Circular domain conditions")
    else:
        raise ValueError("FAIL: Boundary validation")

    # Step 9: Validate initial numerical values
    if np.isfinite(solver.u).all():
        print("PASS: Finite displacement values")
    else:
        raise ValueError("FAIL: Invalid numerical values")

    print("==============================")

    # Step 10: Collect scientific measurements
    history = {
        "max_amplitude": [],
        "rms_displacement": [],
        "kinetic_energy": [],
        "potential_energy": [],
        "total_energy": []
    }

    time_values = []

    for step in range(steps + 1):

        measurements = calculate_diagnostics(solver)

        time_values.append(solver.time)

        for key in history:
            history[key].append(measurements[key])

        if step < steps:
            solver.step()

    # Step 11: Validate final simulation state
    if not np.isfinite(solver.u).all():
        raise FloatingPointError(
            "Simulation produced invalid displacement values."
        )

    if not np.all(solver.u[~interior_mask] == 0.0):
        raise RuntimeError(
            "Fixed boundary conditions were violated."
        )
    if not np.isclose(solver.time, SIMULATION_DURATION):
        raise RuntimeError("Simulation did not reach the configured duration.")

    print("\n--- Final Simulation Validation ---")
    print("PASS: Finite final displacement")
    print("PASS: Fixed boundary conditions")

    # Step 12: Energy summary
    initial_energy = history["total_energy"][0]
    final_energy = history["total_energy"][-1]

    print("\n--- Energy Diagnostics ---")
    print(f"Initial approximate energy : {initial_energy:.8f}")
    print(f"Final approximate energy   : {final_energy:.8f}")
    print("Energy is reported in relative model units, not joules.")

    if initial_energy > 0:
        energy_change = (
            (final_energy - initial_energy)
            / initial_energy
        ) * 100

        print(f"Energy change        : {energy_change:.2f}%")
    else:
        print("Energy change        : Not defined (zero initial energy)")

    # Step 13: Generate scientific plots
    plot_amplitude(time_values, history)
    plot_energy(time_values, history)
    plot_heatmap(X, Y, solver.u, domain_mask)

    # Step 14: Reinitialize solver for animation
    solver = WaveSolver(
        initial_displacement,
        dx, dy, dt,
        WAVE_SPEED, DAMPING,
        domain_mask, interior_mask
    )

    # Step 15: Create membrane animation
    if plt.get_backend().lower() == "agg":
        print(
            "Non-interactive Agg backend: plot windows and animation "
            "were not displayed."
        )
    else:
        animation = animate_membrane(
            X,
            Y,
            solver,
            SIMULATION_DURATION,
            dt,
            ANIMATION_INTERVAL,
        )
        if animation.event_source is None:
            raise RuntimeError("Animation could not initialize its frame timer.")
        plt.show()


if __name__ == "__main__":
    main()
