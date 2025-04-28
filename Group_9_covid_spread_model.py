import numpy as np
import random

#we looked up how to present the graphs in a logical way, and found that we can import modules from matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#Global variable state declaration
EMPTY = 0 #black - uninhabitable
INFECTED_SYMPTOMATIC = 1  # Red - Active infection with symptoms
INFECTED_ASYMPTOMATIC = 2  # Orange - Active infection without symptoms
NOT_INFECTED = 3  # Green - Susceptible population
IMMUNE_LIMITED = 4  # Blue - Temporary immunity
IMMUNE_FOREVER = 5  # White - Permanent immunity

#Function for sim and generating graphs
def covid_simulation(days_to_run):

    # Seed for replicate trials
    random.seed(42)
    np.random.seed(42)

    # Create a 50x50 array
    grid = np.full((50, 50), NOT_INFECTED, dtype=int)

    # Set a fixed number of empty cells (20% of the grid)
    empty_count = int(0.2 * grid.size)

    # Randomly select positions for empty cells
    all_positions = [(i, j) for i in range(50) for j in range(50)]
    random.shuffle(all_positions)
    empty_positions = all_positions[:empty_count]

    # Set empty cells
    for i, j in empty_positions:
        grid[i, j] = EMPTY

    # Randomly select a position for the first COVID patient (not in an empty cell)
    non_empty_positions = [pos for pos in all_positions if pos not in empty_positions]
    first_patient_position = random.choice(non_empty_positions)
    print(f"First patient position: {first_patient_position}")

    # Set only one infected person at the start
    grid[first_patient_position] = INFECTED_SYMPTOMATIC

    # Randomly assign immune people (5% immune-forever)
    for i in range(50):
        for j in range(50):
            if grid[i, j] == NOT_INFECTED and random.random() < 0.05:
                grid[i, j] = IMMUNE_FOREVER

    # Initializing the arrays that track the individual people
    infection_duration = np.zeros_like(grid)  # Track infection duration
    immune_duration = np.zeros_like(grid)  # Track immune-limited duration

    # Set infection duration for first patient
    infection_duration[first_patient_position] = 1

    # Store grid conditions
    grid_history = [grid.copy()]

    # Run simulation for specified number of days
    for _ in range(days_to_run):
        # Move individuals with infection bias while preserving empty cells
        grid = move_individuals_with_infection_repelling_bias(grid)

        # Spread infection
        grid, infection_duration = spread_infection(grid, infection_duration)

        # Update immune states
        grid, infection_duration, immune_duration = update_immune_states(
            grid, infection_duration, immune_duration
        )

        # Store current state
        grid_history.append(grid.copy())

    # Create 6 equidistant snapshots
    snapshot_indices = [int(i * days_to_run / 5) for i in range(6)]
    snapshots = [grid_history[idx] for idx in snapshot_indices]

    # Visualize the snapshots with clear time labels
    visualize_snapshots(snapshots, snapshot_indices)

    return snapshots

# movement function to move away from infectious individuals
def move_individuals_with_infection_repelling_bias(grid):
    rows, cols = grid.shape
    new_grid = grid.copy()

    # Identify infected cells (same as original)
    infected_positions = [
        (i, j) for i in range(rows) for j in range(cols)
        if grid[i, j] in [INFECTED_SYMPTOMATIC, INFECTED_ASYMPTOMATIC]
    ]

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != EMPTY:
                possible_moves = []
                farther_moves = []

                # seeking proximity with new_dist < curr_dist
                # avoid priomximity with new_dist > curr_dist
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and new_grid[ni, nj] == EMPTY:
                            possible_moves.append((ni, nj))

                            # Changed distance comparison
                            for inf_i, inf_j in infected_positions:
                                curr_dist = abs(i - inf_i) + abs(j - inf_j)
                                new_dist = abs(ni - inf_i) + abs(nj - inf_j)
                                if new_dist > curr_dist:
                                    farther_moves.append((ni, nj))
                                    break

                if possible_moves:
                    # 25% chance to choose farther moves (arbitrary number to make the movement path logical)
                    if farther_moves and random.random() < 0.25:
                        new_i, new_j = random.choice(farther_moves)
                    else:
                        new_i, new_j = random.choice(possible_moves)

                    new_grid[new_i, new_j] = grid[i, j]
                    new_grid[i, j] = EMPTY
    return new_grid


def spread_infection(grid, infection_duration):
    #infection spread based on Moore radii
    new_grid = grid.copy()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            # Check if person is infected and can spread (first 8 days)
            if grid[i, j] in [INFECTED_SYMPTOMATIC, INFECTED_ASYMPTOMATIC] and infection_duration[i, j] <= 8:
                # Check Moore neighborhood radius 1
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if grid[ni, nj] == NOT_INFECTED:
                                if random.random() < 0.5:  # 50% chance of infection
                                    # 10% chance of being asymptomatic
                                    if random.random() < 0.1:
                                        new_grid[ni, nj] = INFECTED_ASYMPTOMATIC
                                    else:
                                        new_grid[ni, nj] = INFECTED_SYMPTOMATIC
                                    infection_duration[ni, nj] = 1  # Start counting infection days

                # Check Moore neighborhood radius 2
                for di in [-2, -1, 0, 1, 2]:
                    for dj in [-2, -1, 0, 1, 2]:
                        if abs(di) == 2 or abs(dj) == 2:  # Ensure radius 2
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if grid[ni, nj] == NOT_INFECTED:
                                    if random.random() < 0.25:  # 25% chance of infection
                                        # 10% chance of being asymptomatic
                                        if random.random() < 0.1:
                                            new_grid[ni, nj] = INFECTED_ASYMPTOMATIC
                                        else:
                                            new_grid[ni, nj] = INFECTED_SYMPTOMATIC
                                        infection_duration[ni, nj] = 1  # Start counting infection days

    return new_grid, infection_duration

def update_immune_states(grid, infection_duration, immune_duration):
    #editing immune states based on infection duration
    new_grid = grid.copy()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] in [INFECTED_SYMPTOMATIC, INFECTED_ASYMPTOMATIC]:
                # Increment infection duration
                infection_duration[i, j] += 1

                # Check if infection duration exceeds 14 days
                if infection_duration[i, j] >= 14:
                    # Always transition to immune-limited state (no new immune-forever, as this is our way of survival of the fittest)
                    new_grid[i, j] = IMMUNE_LIMITED

                    # Randomly assign immune-limited duration (3-6 months) --> mimics real-life scenarios
                    immune_months = random.randint(3, 6)
                    immune_duration[i, j] = immune_months * 30  # Convert months to days

                    # Reset infection duration
                    infection_duration[i, j] = 0

            elif grid[i, j] == IMMUNE_LIMITED:
                # Decrease immune-limited duration
                immune_duration[i, j] -= 1
                if immune_duration[i, j] <= 0:
                    # Transition to not infected state
                    new_grid[i, j] = NOT_INFECTED

    return new_grid, infection_duration, immune_duration


def visualize_snapshots(snapshots, day_indices):
    # Create custom colormap with logical colors
    colors = [(0, 0, 0),  # black (empty)
              (0.9, 0.1, 0.1),  # bright red (infected symptomatic)
              (1.0, 0.6, 0.0),  # bright orange (infected asymptomatic)
              (0.0, 0.8, 0.2),  # bright green (not infected)
              (0.2, 0.4, 0.8),  # bright blue (immune limited)
              (1, 1, 1)]  # white (immune forever)

    custom_cmap = LinearSegmentedColormap.from_list('covid_cmap', colors, N=6)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    state_labels = [
        'Empty',
        'Infected (Symptomatic)',
        'Infected (Asymptomatic)',
        'Not Infected',
        'Immune (Limited)',
        'Immune (Forever)'
    ]
    state_colors = ['black', 'red', 'orange', 'green', 'blue', 'white']

    for i, (ax, snapshot, day) in enumerate(zip(axes, snapshots, day_indices)):
        im = ax.imshow(snapshot, cmap=custom_cmap, vmin=0, vmax=5)
        ax.set_title(f'Day {day}', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Count each state
        counts = [np.count_nonzero(snapshot == s) for s in range(6)]
        # Formatting text for display
        legend_text = "\n".join(
            [f"{label}: {count}" for label, count in zip(state_labels, counts)]
        )
        # Place the text box to the right of the plot
        ax.text(
            1.05, 0.5, legend_text,
            transform=ax.transAxes,
            fontsize=12,
            va='center', ha='left',
            bbox=dict(boxstyle="round,pad=0.5", fc="w", ec="k", alpha=0.8)
        )

    plt.suptitle('COVID-19 Spread Simulation Over Time', fontsize=20, y=0.98)

    # Creating custom legend tags using object-based programming
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in state_colors]
    fig.legend(
        legend_patches, state_labels, loc='lower center',
        ncol=6, bbox_to_anchor=(0.5, 0.01), fontsize=14,
        frameon=True, fancybox=True, shadow=True
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.subplots_adjust(wspace=0.35, hspace=0.25)
    plt.show()

# Run the simulation for x number of days
#covid_simulation(72)

days = int(input('How many days would you like to run this simulation for? '))
covid_simulation(days)