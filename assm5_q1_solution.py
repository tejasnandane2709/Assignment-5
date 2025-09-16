"""
Assignment-5, Problem-1
Monty Hall simulation 
Broken into six tasks:
  Task 1: implement single-trial logic (single_trial)
  Task 2: batched simulator (simulate_monty)
  Task 3: running / cumulative probability 
  Task 4: plot monty_CI (confidence interval) 
  Task 5: compare to analytic and never-switch strategy (compare_strategies_plot)

To run: save as monty_hall.py and run with python3 (requires numpy, scipy, matplotlib).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# --------------------------
# TASK 1: Single-trial logic
# --------------------------
def single_trial(rng, verbose=True):
    """
    Perform a single Monty Hall trial using RNG instance (scipy-compatible).
    Returns tuple (car_pos, initial_pick, host_open, final_choice_if_switch, win_if_switch)
    Doors coded 0,1,2.
    """
    car = st.randint.rvs(0, 3, random_state=rng)         # car position {0,1,2}
    pick = st.randint.rvs(0, 3, random_state=rng)        # player's initial pick
    
    # host choice:
    if pick != car:
        # host must open the only door that is neither pick nor car
        host_open = ({0, 1, 2} - {pick, car}).pop()
    else:
        # pick == car -> host randomly opens one of the two goats
        other_doors = list({0, 1, 2} - {pick})
        choice_idx = st.randint(0, 2).rvs(random_state=rng)
        host_open = other_doors[int(choice_idx)]
    remaining = ({0, 1, 2} - {pick, host_open}).pop()
    final_choice_if_switch = remaining
    win_if_switch = int(final_choice_if_switch == car)
    win_if_not_switch = int(pick == car)

    if verbose:
        print(f"Car is behind door {car+1}, "
              f"you picked door {pick+1}, "
              f"host opened door {host_open+1}, "
              f"you switched to door {final_choice_if_switch+1} ---> "
              f"{'WIN' if win_if_switch else 'LOSE'}")
        
    return win_if_switch, win_if_not_switch

# --------------------------
# TASK 2 & 3: Build a batch simulator and plot the evolution
# --------------------------
def simulate_monty(N=200, switch=True, seed=None, createPlot=True):
    """
    Simulate N Monty Hall trials by reusing single_trial().
    Returns:
      p_k: array length N of cumulative win probability after each trial
      wins: array length N of 0/1 wins per trial
    Implementation notes:
      - vectorized generation is avoided for ease; interested students, please look into vectorization
    """
    # Plot the evolution of the analysis
    if createPlot:
        plt.ion()
        plt.figure()
        plt.title("Evolution of the Monty Hall Sampling")
        plt.grid(True)
        plt.ylim(0, 100)
        plt.autoscale(True)
        plt.ylabel('Winning Chance (%)')
        plt.xlabel('Number of samples')

    # Create a random number generator with the given seed (for reproducibility)
    rng = np.random.default_rng(seed)

    # Allocate an array to store win/loss outcomes (1 = win, 0 = loss)
    wins = np.empty(N, dtype=int)

    # Repeat N independent Monty Hall trials
    for i in range(N):
        win_if_switch, win_if_not_switch = single_trial(rng, verbose=False) # Run one trial with verbose False
        wins[i] = win_if_switch if switch else win_if_not_switch # assign outcome depending on the strategy = switch

    # Compute cumulative wins (running total of successes)
    cum_wins = np.cumsum(wins)

    # Compute running probability of winning after each trial
    p_k = cum_wins / (np.arange(1, N + 1))*100

    # Add point to the plot
    if createPlot:
        previousChance = p_k[0]
        plt.plot([1, N], [1/3*100, 1/3*100], 'b-', linewidth=2.0, label='No switch (33.3%)')
        plt.plot([1, N], [2/3*100, 2/3*100], 'r-', linewidth=2.0, label='Switch (66.7%)')

        for i in range(N):
            plt.plot([i, i+1], [previousChance, p_k[i]], 'k-', linewidth=1.0)
            plt.pause(0.00000001)
            previousChance = p_k[i]
        plt.legend()
        plt.ioff()          # turn off interactive mode
        plt.show(block=False)

    # Return both the running probabilities and the raw win/loss record
    return p_k, wins

# --------------------------
# TASK 4: Confidence interval plot
# --------------------------
def plot_monty_CI(N=500, switch=True, seed=None, alpha=0.05):
    """
    Simulate N trials, compute cumulative probability, and plot CI.
    CI is computed as normal approx: p_k ± z*sqrt(p*(1-p)/n)
    """

    # Run the simulation
    p_k, wins = simulate_monty(N, switch=switch, seed=seed, createPlot=False)
    
    # Convert p_k to fraction (0-1)
    p_frac = p_k / 100
    
    # Compute standard error and CI
    z = st.norm.ppf(1 - alpha/2)
    se = np.sqrt(p_frac * (1 - p_frac) / (np.arange(1, N+1)))
    lower = p_frac - z * se
    upper = p_frac + z * se

    # Plot
    plt.figure()
    plt.plot([1, N], [1/3*100, 1/3*100], 'b-', linewidth=2.0, label='No switch (33.3%)')
    plt.plot([1, N], [2/3*100, 2/3*100], 'r-', linewidth=2.0, label='Switch (66.7%)')
    plt.plot(np.arange(1, N+1), p_frac*100, 'k-', label='Estimated win %')
    plt.fill_between(np.arange(1, N+1), np.clip(lower*100, 0, 100), np.clip(upper*100, 0, 100), color='gray', alpha=0.3, label=f'{100*(1-alpha):.0f}% CI')
    
    plt.title("Monty Hall Winning Probability with Confidence Interval")
    plt.xlabel("Number of trials")
    plt.ylabel("Winning Chance (%)")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend()
    plt.show(block=False)


# --------------------------
# TASK 5: Side-by-side comparison of two strategies (switch or not) using the same randomization trials
# --------------------------
def compare_strategies_plot(N=500, seed=None):
    _, wins_switch_true = simulate_monty(N, switch=True, seed=seed, createPlot=False)
    _, wins_switch_false = simulate_monty(N, switch=False, seed=seed, createPlot=False)

    p_switch = np.cumsum(wins_switch_true) / np.arange(1, N+1)
    p_no_switch = np.cumsum(wins_switch_false) / np.arange(1, N+1)

    plt.figure()
    plt.plot(p_switch*100, 'r-', label='Switch strategy')
    plt.plot(p_no_switch*100, 'b-', label='No-switch strategy')
    plt.plot([1, N], [1/3*100, 1/3*100], 'b--', linewidth=2.0, label='Switch - theoretical (66.7%)')
    plt.plot([1, N], [2/3*100, 2/3*100], 'r--', linewidth=2.0, label='No switch - theoretical (33.3%)')
    plt.title("Switch vs. No-switch strategies")
    plt.xlabel("Number of trials")
    plt.ylabel("Winning Chance (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.show(block=False)


# --------------------------
# Main function
# --------------------------
if __name__ == "__main__":
    # Example parameters
    N = 200
    seed = 20250917
    rng = np.random.default_rng(seed)
    
    # Task 1: single trial
    # for i in range(5):
    #     print(f"Trial {i+1}:")
    #     single_trial(rng)

    # Task 2 & 3: simulate and get cumulative p_k
    p_switch, wins_switch = simulate_monty(N, switch=True, seed=seed, createPlot=True)

    # Task 4: plot CI
    # plot_monty_CI(N=200, switch=True, seed=None, alpha=0.05)
    # plot_monty_CI(N=200, switch=False, seed=None, alpha=0.05)

    # Task 5: side-by-side comparison of switch-or-not
    # compare_strategies_plot()

    # display any open matplotlib figures
    import matplotlib.pyplot as plt
    plt.show()

    pass
