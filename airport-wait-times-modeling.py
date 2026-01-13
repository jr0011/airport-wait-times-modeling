import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import expon, norm
import seaborn as sns

# Check if the file exists
url = "airport.csv"
if not os.path.isfile(url):
    raise FileNotFoundError(f"Could not find file: {url}")

# Read the file into a pandas dataframe
airport_df = pd.read_csv(url)


# Print only the rows where the month is September

# Filter for rows where Month is 'September' (case-insensitive) and Year is 2015-2019
september_df = airport_df[
        (airport_df['Month'].str.lower() == 'september') &
        (airport_df['Year'].between(2015, 2019))
    ]

# Calculate the average number of montly passengers
average_monthly_passengers = september_df['Total Passengers'].mean()

# Convert this to an hourly arrival rate assuming: 30-day month, 16-hour operational days
hourly_arrival_rate = average_monthly_passengers / (30 * 16)

per_lane_arrival_rate = hourly_arrival_rate / 50

arrival_rate_per_lane_per_minute = per_lane_arrival_rate / 60

def arrivals(env, arrival_rate: float, passengers: int, passenger_lane, arrival_data):
    """Simpy process that Simulates the arrival of passengers to security screening"""

    count = 0

    mean_interarrival_time = 1 / arrival_rate

    while count < passengers:

        number_of_events = np.random.poisson(lam = arrival_rate)

        arrival_time = np.random.exponential(scale = mean_interarrival_time)

        # print(f"arrival time: {arrival_time * 60}")
        
        yield env.timeout(arrival_time * 60)

        simulated_inter_arrivals.append(arrival_time * 60)
        simulated_number_of_events.append(number_of_events)

        arrival_data.append(env.now)

        passenger_records[count] = {'arrival': env.now}

        passenger_lane.put((count, arrival_time)) # Add Passenger To The Queue

        print(f"{env.now:6.1f} Seconds: Passenger {count} Arrived Now!")

        count += 1

def security(env, service_time_params: list, servers, passenger_lane):

    #processed_count = 0

    while True:

        passenger_id, arrival_time = yield passenger_lane.get()

        with servers.request() as req:

            yield req # Locks The Server Resource

            print(f"{env.now:6.1f} Seconds: Passenger {passenger_id} Going Through Security Check!")

            start_time = env.now

            passenger_records[passenger_id]['start'] = start_time

            service_time = np.random.normal(service_time_params[0], service_time_params[1])

            simulated_service_time.append(service_time*60)

            while service_time < 0:
                service_time = np.random.normal(service_time_params[0], service_time_params[1])

            yield env.timeout(service_time * 60)

            finish_time = env.now

            passenger_records[passenger_id]["finish"] = finish_time

            print(f"{env.now:6.1f} Seconds: Passenger {passenger_id} Done With Security Check!")

def customer_journey(passenger_information: dict):

    passenger_ids = sorted(passenger_information.keys())

    waiting_times =[]
    service_times = []
    total_times = []

    # Lists for absolute times
    arrival_times_sec = []
    start_times_sec = []

    for p_id in passenger_ids:
        record = passenger_records[p_id]
        if 'start' in record and 'finish' in record:
            waiting_time = record['start'] - record['arrival']
            service_time = record['finish'] - record['start']
            total_time = record['finish'] - record['arrival']
            
            waiting_times.append(waiting_time)
            service_times.append(service_time)
            total_times.append(total_time)

            # Append Absolute Times
            arrival_times_sec.append(record['arrival'])
            start_times_sec.append(record['start'])

    return waiting_times, service_times, total_times, arrival_times_sec, start_times_sec

# Calculate the test lambda using target p and expected service
def test_lambda(p_target = 0.85, expected_service = 1):
    lambda_test = p_target / expected_service
    p_test = lambda_test * expected_service

    print(f"Lamda_test = {lambda_test}")
    print(f"p_test = {p_test}")
    print(f"Is p_test less then 1? {p_test < 1}")

    return lambda_test, p_test

# Calculate Expected waiting time using the Pollaczek-Khintchine (P-K) formula for M/G/1:
def theoretical_wait(lambda_test, expected_service, sigma_service, p_test):
    service_time = expected_service**2 + sigma_service**2

    wait_time = (lambda_test * service_time) / (2 *(1 - p_test))

    return wait_time * 60

# Validate the simulation using expected waiting time
def validation_t_test(sample_mean, sample_sigma, expected_mean, number_samples=200):

    t = (sample_mean - expected_mean) / (sample_sigma / np.sqrt(number_samples))

    t_sig = 1.685
    print("t_{0,975, 39} = 1.685")

    print(f"Can we reject H0: {t > t_sig}")

    return t


def run_multiple_simulations(
    num_simulations: int = 200,
    passengers: int = 2000,
    simulation_time: float = 1000000,
    arrival_lambda: float = 0.85,
    service_time_params: list | None = None,
    warmup_n: int = 200,
):
    """Run multiple independent simulations and return a list of average waiting times."""

    if service_time_params is None:
        service_time_params = [1, 0.25]

    avg_waiting_times = []


    all_sim_inter_arrivals = []
    all_sim_service_time = []
    
    waiting_run1, service_run1, total_run1 = [], [], []
    arrival_sec_run1, start_sec_run1, passenger_records_run1 = [], [], {}

    for r in range(num_simulations):
        env = simpy.Environment()
        servers = simpy.Resource(env, capacity=1)
        passenger_lane = simpy.Store(env, capacity=1)

        global passenger_records
        global simulated_inter_arrivals
        global simulated_number_of_events
        global simulated_service_time

        passenger_records = {}
        simulated_inter_arrivals = []
        simulated_number_of_events = []
        simulated_service_time = []

        all_arrival_times = []

        # Start Simpy processes for this replication
        env.process(
            arrivals(
                env,
                arrival_rate=arrival_lambda,
                passengers=passengers,
                passenger_lane=passenger_lane,
                arrival_data=all_arrival_times,
            )
        )
        env.process(
            security(
                env,
                service_time_params=service_time_params,
                servers=servers,
                passenger_lane=passenger_lane,
            )
        )

        # Run the simulation
        env.run(until=simulation_time)

        waiting, service, total, arrival_sec, start_sec = customer_journey(passenger_records)

        all_sim_inter_arrivals.extend(simulated_inter_arrivals)
        all_sim_service_time.extend(simulated_service_time)

        if r == 0:
            waiting_run1 = waiting
            service_run1 = service
            total_run1 = total
            arrival_sec_run1 = arrival_sec
            start_sec_run1 = start_sec
            passenger_records_run1 = passenger_records

        # apply warm-up by discarding the first warmup_n customers
        waiting = waiting[warmup_n:]
        service = service[warmup_n:]
        total = total[warmup_n:]

        if len(waiting) > 0:
            avg_waiting_times.append(np.mean(waiting))
        else:
            avg_waiting_times.append(np.nan)

    return (
        avg_waiting_times, 
        all_sim_inter_arrivals, 
        all_sim_service_time, 
        waiting_run1, 
        service_run1, 
        total_run1, 
        arrival_sec_run1, 
        start_sec_run1,
        passenger_records_run1
    ) 

# Validate the simulation using expected waiting time
def validation(sample_mean, sample_sigma, expected_mean, number_samples=40):

    t = (sample_mean - expected_mean) / (sample_sigma / np.sqrt(number_samples))

    t_sig = 1.685
    print("t_{0,975, 39} = 1.685")

    print(f"Can we reject H0: {abs(t) > t_sig}")

    return t


def validation_plots(avg_waiting_times, theoretical_wait, alpha=0.05):
    """
    Create plots to visually compare the simulated average waiting times
    to the theoretical waiting time from the P-K formula.

    Plots:
    1. Histogram of replication means
    2. Mean ± 95% CI with theoretical W_q
    3. Trace plot of replication means
    4. Bar plot: theoretical vs simulated W_q (with CI on simulation)
    5. Optional: per-replication t-values vs critical region
    """
    avg_waiting_times = np.array(avg_waiting_times, dtype=float)
    R = len(avg_waiting_times)

    # Sample statistics and t-test ingredients (based on replication means)
    sample_mean = np.mean(avg_waiting_times)
    sample_std = np.std(avg_waiting_times, ddof=1)
    t_crit = 1.685  # given t_{0.975,39}
    se = sample_std / np.sqrt(R)
    t_stat = (sample_mean - theoretical_wait) / se

    # Figure 1: Plot 1 (histogram) + Plot 3 (trace plot)
    plt.figure(figsize=(10, 8))

    # Plot 1 — Histogram of per-replication waiting times
    plt.subplot(2, 1, 1)
    plt.hist(avg_waiting_times, bins=40, edgecolor="black", alpha=0.7)
    plt.axvline(theoretical_wait, color="red", linestyle="--", label="Theoretical $W_q$")
    plt.xlabel("Average waiting time per replication (seconds)")
    plt.ylabel("Frequency")
    plt.title("Plot 1: Distribution of replication means vs theory")
    plt.legend()
    plt.savefig("histogram1.png", dpi=300)

    # Plot 3 — Trace plot of replication means
    plt.subplot(2, 1, 2)
    plt.plot(range(1, R + 1), avg_waiting_times, marker="o", markersize=4)
    plt.axhline(theoretical_wait, color="red", linestyle="--", label="Theoretical $W_q$")
    plt.xlabel("Replication index")
    plt.ylabel("Mean waiting time per replication (seconds)")
    plt.title("Plot 3: Trace of replication means")
    plt.legend()
    plt.savefig("traceplot.png", dpi=300)


    plt.tight_layout()

    # Figure 2: Convergence of running mean to theoretical value
    running_mean = np.cumsum(avg_waiting_times) / np.arange(1, R + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, R + 1), running_mean, marker="o", markersize=3, label="Running mean of $W_q^{(r)}$")
    plt.axhline(theoretical_wait, color="red", linestyle="--", label="Theoretical $W_q$")
    plt.xlabel("Number of replications")
    plt.ylabel("Running mean waiting time (seconds)")
    plt.title("Convergence of simulated mean to theoretical $W_q$")
    plt.legend()
    plt.savefig("convergence_running_mean.png", dpi=300)

    # Figure 3: Plot 2 — Mean ± 95% Confidence Interval with theoretical line (horizontal CI)
    ci_lower = sample_mean - t_crit * se
    ci_upper = sample_mean + t_crit * se

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x=[sample_mean],
        y=[0],
        xerr=[[sample_mean - ci_lower], [ci_upper - sample_mean]],
        fmt="o",
        capsize=5,
        label="Simulated mean ± 95% CI",
    )
    plt.axvline(theoretical_wait, color="red", linestyle="--", label="Theoretical $W_q$")
    plt.ylim(-1, 1)
    plt.yticks([])
    plt.xlabel("Waiting time (seconds)")
    plt.title("Plot 2: Simulated mean with 95% CI vs theoretical $W_q$")
    plt.legend(loc="best")
    plt.savefig("ci_95.png", dpi=300)

    # Figure 4: Plot 4 — Comparison bar plot
    plt.figure(figsize=(6, 4))
    labels = ["Theoretical $W_q$", "Simulated $W_q$"]
    means = [theoretical_wait, sample_mean]
    x_pos = np.arange(len(labels))

    # Error bar only on simulated bar (95% CI)
    yerr = [0, ci_upper - sample_mean]

    plt.bar(x_pos, means, yerr=yerr, capsize=5, color=["red", "C0"])
    plt.xticks(x_pos, labels)
    plt.ylabel("Waiting time (seconds)")
    plt.title("Plot 4: Theoretical vs simulated waiting time")
    plt.savefig("error_bar.png", dpi=300)

    # Figure 5: Plot 5 — Per-replication t-statistics
    # Use pooled standard error (se) for all replications
    t_values = (avg_waiting_times - theoretical_wait) / se

    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(t_crit, color="red", linestyle="--", label=r"$\pm t_{0.975,39}$")
    plt.axhline(-t_crit, color="red", linestyle="--")
    plt.plot(range(1, R + 1), t_values, marker="o", linestyle="-", label="Per-replication t-value")
    plt.xlabel("Replication index")
    plt.ylabel("t-value")
    plt.title("Plot 5 : Per-replication t-values vs critical region")
    plt.legend(loc="best")
    plt.savefig("t_stats.png", dpi=300)

    # Render all created figures
    plt.show()

def simulation_plots(
    waiting_times, 
    service_times, 
    total_times, 
    arrival_times_sec, 
    start_times_sec,  
    passenger_information: dict, 
    simulated_inter_arrivals: list, 
    simulated_service_times:list,
):
    
    plt.figure(figsize=(10, 6))

    # Calculate means from the POOLED data
    mean_interarrival = np.mean(simulated_inter_arrivals)
    mean_service = np.mean(simulated_service_times)
    
    # Determine max X-axis range for plotting
    x_max = max(max(simulated_inter_arrivals) if simulated_inter_arrivals else 1, 
                max(simulated_service_times) if simulated_service_times else 1)
    x = np.linspace(0, x_max, 1000)

    # Plot Histograms (using 50 bins for better clarity with large pooled data)
    plt.hist(
        simulated_inter_arrivals, 
        bins=50, 
        density=True, 
        histtype='stepfilled', 
        label='Simulated Inter-Arrivals', 
        color='C0',
        alpha=0.7,
        edgecolor="black",
    )
    plt.hist(
        simulated_service_times, 
        bins=50, 
        density=True, 
        histtype='stepfilled', 
        label='Simulated Service Times', 
        alpha=0.7, 
        color='red',
        edgecolor="black"
    )

    # 1. Theoretical Exponential PDF (Arrivals)
    plt.plot(
        x, 
        expon.pdf(x, scale=mean_interarrival), 
        'b-', 
        linewidth=2, 
        label=f"Exp. PDF ($\mu={mean_interarrival:.2f}$)"
    )
    
    # 2. Theoretical Normal PDF (Service: $\mu=60$ sec, $\sigma=15$ sec)
    expected_mean_service = 60.0 
    expected_std_service = 15.0
    plt.plot(
        x, 
        norm.pdf(x, loc=expected_mean_service, scale=expected_std_service), 
        'r-', 
        linewidth=2, 
        label=f"Normal PDF ($\mu={expected_mean_service:.0f}, \sigma={expected_std_service:.0f}$)"
    )

    # Mean Lines
    plt.axvline(x=mean_interarrival, color='b', linestyle='--', linewidth=1.2, label=f'Sim. Mean Arrival Time ({mean_interarrival:.2f})')
    plt.axvline(x=mean_service, color='r', linestyle='--', linewidth=1.2, label=f'Sim. Mean Service Time ({mean_service:.2f})')

    plt.title(f" Inter-Arrival and Service Time Distributions (All Simulations)")
    plt.xlabel("Time Between Events (Seconds)")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig("mean_lines.png", dpi=300)
    plt.show()

    
    num_customers = len(waiting_times)
    
    START_INDEX = 0       # Start from the 101st customer (index 100)
    N_TO_PLOT = 100

    # Set the range of customers to plot for readability (e.g., first 100)
    END_INDEX = min(START_INDEX + N_TO_PLOT, num_customers)
    
    # Slice all arrays (durations and absolute times) to the same length
    p_indices_subset = list(range(START_INDEX, END_INDEX))
    waiting_subset = waiting_times[START_INDEX:END_INDEX]
    service_subset = service_times[START_INDEX:END_INDEX]
    arrival_sec_subset = arrival_times_sec[START_INDEX:END_INDEX]
    start_sec_subset = start_times_sec[START_INDEX:END_INDEX]

    # Check if there's data to plot
    if not p_indices_subset:
        print("Not enough data to create Gantt chart after specified START_INDEX.")
        return

    # Calculate height dynamically
    customers_to_plot_count = len(waiting_subset)
    BASE_HEIGHT_PER_CUSTOMER = 0.3
    plot_height = max(8, customers_to_plot_count * BASE_HEIGHT_PER_CUSTOMER)

    fig, ax = plt.subplots(figsize=(16, plot_height))

    # 1. Waiting Bar: Width is Duration, Offset (left) is ABSOLUTE ARRIVAL TIME
    ax.barh(p_indices_subset, waiting_subset, left=arrival_sec_subset, color='C0', label='Waiting Time', height=0.8)
    
    # 2. Service Bar: Width is Duration, Offset (left) is ABSOLUTE START TIME
    ax.barh(p_indices_subset, service_subset, left=start_sec_subset,color='red', label='Service Time', height=0.8)
    
    # Dynamic Y-Tick/Label Interval
    max_ticks = 20
    tick_interval = int(np.ceil(N_TO_PLOT / max_ticks)) if N_TO_PLOT > max_ticks else 1 

    # Since passenger_ids keys are numbers, use them for labels
    passenger_ids = sorted(passenger_information.keys())
    
    ax.set_yticks(p_indices_subset[::tick_interval])
    ax.set_yticklabels([f"Cust {i + 1}" for i in p_indices_subset[::tick_interval]], fontsize=8)

    if N_TO_PLOT > 0 and len(start_sec_subset) > 0:
        x_max_limit = start_sec_subset[-1] + service_subset[-1] + 100 # Last finish time + buffer
        ax.set_xlim(0, x_max_limit) 

    ax.set_title(f"Customer Timeline (Gantt Chart) - Customers {START_INDEX+1} to {END_INDEX} (Sim. Run 1)", fontsize=16)
    ax.set_xlabel("Absolute Simulation Time (Seconds)", fontsize=12)
    ax.set_ylabel("Customer ID", fontsize=12)
    ax.grid(axis='x', linestyle='-', alpha=0.5)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("chart1.png", dpi=300)
    plt.show()


def run_one_replication(n_passengers, arrival_lambda, servers_count, sigma_service, return_details=False):
    """Runs a single replication of the simulation."""
    global passenger_records, simulated_inter_arrivals, simulated_number_of_events, simulated_service_time
    
    passenger_records = {}
    simulated_inter_arrivals = []
    simulated_number_of_events = []
    simulated_service_time = []

    env = simpy.Environment()
    servers = simpy.Resource(env, capacity=servers_count)
    lane = simpy.Store(env)
    
    all_arrival_times = []

    env.process(arrivals(env, arrival_lambda, n_passengers, lane, all_arrival_times))

    for i in range(servers_count):
        env.process(security(env, [1, sigma_service], servers, lane))

    env.run()
    
    waiting, service, total, arrival_times, start_times = customer_journey(passenger_records)
    if len(waiting) == 0:
        if return_details:
            return 0, None
        return 0
    mean_waiting = np.mean(waiting) / 60  # minutes
    
    if return_details:
        details = {
            'waiting_times': np.array(waiting),
            'service_times': np.array(service),
            'total_times': np.array(total),
            'arrival_times_sec': np.array(arrival_times),
            'start_times_sec': np.array(start_times),
            'passenger_information': passenger_records,
            'simulated_inter_arrivals': simulated_inter_arrivals.copy(),
            'simulated_service_times': simulated_service_time.copy()
        }
        return mean_waiting, details
    
    return mean_waiting

def run_R_replications(R, n_passengers, arrival_lambda, scenario, capture_first=False):
    """Run R replications for a given scenario."""
    results = []
    first_run_details = None

    for r in range(R):
        np.random.seed(r)
        
        # Capture details on first run if requested
        if r == 0 and capture_first:
            mean_wait, first_run_details = run_one_replication(
                n_passengers,
                arrival_lambda,
                servers_count=scenario["servers"],
                sigma_service=scenario["sigma_service"],
                return_details=True
            )
            results.append(mean_wait)
        else:
            results.append(
                run_one_replication(
                    n_passengers,
                    arrival_lambda,
                    servers_count=scenario["servers"],
                    sigma_service=scenario["sigma_service"]
                )
            )
    
    if capture_first:
        return np.array(results), first_run_details
    return np.array(results)

def welch_t_test(baseline, option, alpha=0.05):
    """Welch t-test between baseline and an option."""
    m1, m2 = baseline.mean(), option.mean()
    v1, v2 = baseline.var(ddof=1), option.var(ddof=1)
    n1, n2 = len(baseline), len(option)
    
    diff = m1 - m2
    se = (v1/n1 + v2/n2)**0.5
    t_stat = diff / se

    df = (v1/n1 + v2/n2)**2 / ((v1**2)/(n1**2*(n1-1)) + (v2**2)/(n2**2*(n2-1)))
    from scipy.stats import t
    p_value = 2 * t.sf(np.abs(t_stat), df)

    return t_stat, p_value, diff, p_value < alpha

def compare_scenarios_table(baseline, optionA, optionB, alpha=0.05):
    records = []
    for name, option in [("Option A", optionA), ("Option B", optionB)]:
        t_stat, p, diff, sig = welch_t_test(baseline, option, alpha)
        records.append({
            "Scenario": name,
            "Mean Baseline (min)": baseline.mean(),
            f"Mean {name} (min)": option.mean(),
            "Difference (Baseline - Option)": diff,
            "t-stat": t_stat,
            "p-value": p,
            "Significant": sig
        })
    return pd.DataFrame(records)

def plot_boxplot_with_mean(baseline, optionA, optionB):
    data = [baseline, optionA, optionB]
    labels = ["Baseline", "Option A", "Option B"]
    plt.figure(figsize=(8,6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2))
    means = [d.mean() for d in data]
    plt.scatter([1,2,3], means, color='darkred', zorder=5, label='Mean')
    plt.ylabel("Average Waiting Time (seconds)")
    plt.title("Scenario Comparison: Waiting Times")
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("boxplot_means.png", dpi=300)
    plt.show()

def plot_violin_and_ci(baseline, optionA, optionB):
    """
    Create violin plot and confidence interval plot for scenario comparison.
    
    Parameters:
    -----------
    baseline : array-like
        Waiting times for baseline scenario
    optionA : array-like
        Waiting times for option A scenario
    optionB : array-like
        Waiting times for option B scenario
    """
    data = [baseline, optionA, optionB]
    labels = ["Baseline", "Option A", "Option B"]

    # Violin plot
    plt.figure(figsize=(8,6))
    sns.violinplot(data=data, inner="quartile", palette=["blue", "lightblue", "red"])
    plt.xticks(ticks=[0,1,2], labels=labels)
    plt.title("Violin Plot: Scenario Distributions")
    plt.ylabel("Average Waiting Time")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("violin.png", dpi=300)
    plt.show()

    # Confidence interval plot
    df = pd.DataFrame({
        "Waiting Time": np.concatenate([baseline, optionA, optionB]),
        "Scenario": ["Baseline"]*len(baseline) + ["Option A"]*len(optionA) + ["Option B"]*len(optionB)
    })

    plt.figure(figsize=(8,6))
    sns.pointplot(x="Scenario", y="Waiting Time", data=df, capsize=0.2, join=False, ci=95, palette=["blue","lightblue","red"])
    plt.title("Mean Waiting Time, by Scenario with 95% CI")
    plt.ylabel("Average Waiting Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('ci.png', dpi=300)
    plt.show()


def scenario_comparison_plots(baseline, optionA, optionB, alpha: float = 0.05):
    """
    Create a set of comparison plots for Baseline, Option A and Option B,
    styled similarly to the validation plots:

    1) Violin plot (with means) of replication mean waiting times
    2) Histogram of replication means for one scenario (Baseline)
    3a) Bar chart comparison: Baseline vs Option A
    3b) Bar chart comparison: Baseline vs Option B
    """

    # Prepare data
    data = [np.array(baseline, dtype=float),
            np.array(optionA, dtype=float),
            np.array(optionB, dtype=float)]
    labels = ["Baseline", "Option A", "Option B"]
    R = len(data[0])

    # Basic stats
    means = [d.mean() for d in data]
    stds = [d.std(ddof=1) for d in data]

    # t critical value for 95% CI with R-1 df; for large R this is ~1.96
    t_crit = 1.685 if R == 40 else 1.966

    ses = [s / np.sqrt(R) for s in stds]
    ci_bounds = [(m - t_crit * se, m + t_crit * se) for m, se in zip(means, ses)]

    # 1) Violin plot with quartiles (replication means per scenario)
    df_plot = pd.DataFrame({
        "Waiting Time": np.concatenate([baseline, optionA, optionB]),
        "Scenario": ["Baseline"]*len(baseline) + ["Option A"]*len(optionA) + ["Option B"]*len(optionB)
    })
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Scenario", y="Waiting Time", data=df_plot, inner="quartile", 
                   palette=["blue", "lightblue", "red"])
    plt.scatter([0, 1, 2], means, color='darkred', zorder=5, s=100, label='Mean', marker='o')
    plt.ylabel("Average Waiting Time (minutes)")
    plt.title("Scenario Comparison: Waiting Times (Replication Means)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("violin.png", dpi=300)

    # 2) Histograms of replication means for all scenarios (in one figure)
    plt.figure(figsize=(12, 4))
    for i, (label, d, m, color) in enumerate(zip(labels, data, means, ["blue", "lightblue", "red"])):
        plt.subplot(1, 3, i + 1)
        plt.hist(d, bins=20, edgecolor="black", alpha=0.7, color=color)
        plt.axvline(m, color="black", linestyle="--", label=f"{label} mean")
        plt.xlabel("Replication mean waiting time (minutes)")
        if i == 0:
            plt.ylabel("Frequency")
        plt.title(f"{label} replication means")
        plt.legend()
        plt.tight_layout()
        plt.savefig("histograms_rep_means.png", dpi=300)

    # 3a) Bar chart comparison: Baseline vs Option A
    plt.figure(figsize=(8, 4))
    baseline_a_labels = ["Baseline", "Option A"]
    baseline_a_means = [means[0], means[1]]
    baseline_a_yerr = [ci_bounds[0][1] - means[0], ci_bounds[1][1] - means[1]]
    x_positions_a = np.arange(len(baseline_a_labels))
    plt.bar(x_positions_a, baseline_a_means, 
            yerr=baseline_a_yerr, 
            capsize=5, 
            color=["blue", "lightblue"])
    plt.xticks(x_positions_a, baseline_a_labels)
    plt.ylabel("Average Waiting Time (minutes)")
    plt.title("Baseline vs Option A: Mean Waiting Times with 95% CI")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("base_optA.png", dpi=300)

    # 3b) Bar chart comparison: Baseline vs Option B
    plt.figure(figsize=(8, 4))
    baseline_b_labels = ["Baseline", "Option B"]
    baseline_b_means = [means[0], means[2]]
    baseline_b_yerr = [ci_bounds[0][1] - means[0], ci_bounds[2][1] - means[2]]
    x_positions_b = np.arange(len(baseline_b_labels))
    plt.bar(x_positions_b, baseline_b_means, 
            yerr=baseline_b_yerr, 
            capsize=5, 
            color=["blue", "red"])
    plt.xticks(x_positions_b, baseline_b_labels)
    plt.ylabel("Average Waiting Time (minutes)")
    plt.title("Baseline vs Option B: Mean Waiting Times with 95% CI")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("base_optB.png", dpi=300)

    plt.show()
    
if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(80)
    
    # Main configuration 
    num_simulations = 40
    passengers = 10000
    simulation_time = 1_000_0000
    arrival_lambda = 0.85
    warmup_n = 200 # Probably best to use a warm up period of 20% of the passengers
    number_samples = num_simulations  # for t-test

    (
        avg_waiting_times_40,
        pooled_inter_arrivals,
        pooled_service_times,
        waiting_run1,
        service_run1,
        total_run1,
        arrival_sec_run1,
        start_sec_run1,
        passenger_records_run1
    ) = run_multiple_simulations(
        num_simulations=num_simulations,
        passengers=passengers,
        simulation_time=simulation_time,
        arrival_lambda=arrival_lambda,
        warmup_n=warmup_n,
    )

    # convert NumPy scalars to plain Python floats
    avg_waiting_times_40 = [float(x) for x in avg_waiting_times_40]

    print(f"Average waiting times for {num_simulations} simulations:", avg_waiting_times_40)
    print(f"Mean of those {num_simulations} averages:", round(np.mean(avg_waiting_times_40), 2))
    # Calculate sample standard deviation
    sample_std = np.std(avg_waiting_times_40)
    print("Sample standard deviation:", round(sample_std, 2))

    lambda_test, p_test = test_lambda()
    expected_service = 1
    sigma_service = 0.25

    wait_time = theoretical_wait(lambda_test, expected_service, sigma_service, p_test)
    print(f"Expected wait time is {wait_time}")

    # Hypothesis test: does simulation match theory?
    # H0: The simulation matches the theory
    # H1: The simulation does not match the theory
    # We reject H0 if the t-value is greater than the critical value
    # We accept H0 if the t-value is less than the critical value
    validation(np.mean(avg_waiting_times_40), sample_std, wait_time, number_samples=number_samples)

    # Visual validation: plots comparing simulation and theory
    validation_plots(avg_waiting_times_40, wait_time)

    simulation_plots(
        waiting_times=waiting_run1,
        service_times=service_run1,
        total_times=total_run1,
        arrival_times_sec=arrival_sec_run1,
        start_times_sec=start_sec_run1,
        passenger_information=passenger_records_run1,
        simulated_inter_arrivals=pooled_inter_arrivals,
        simulated_service_times=pooled_service_times,
    )

    scenarios = {
        "Baseline": {"servers": 1, "sigma_service": 0.25},
        "Option A": {"servers": 2, "sigma_service": 0.25},
        "Option B": {"servers": 1, "sigma_service": 0.10}
    }

    R = 40
    n_passengers_2b = 3000
    arrival_lambda_2b =4.2336

    print("Running Baseline scenario...")
    baseline, baseline_details = run_R_replications(R, n_passengers_2b, arrival_lambda_2b, scenarios["Baseline"],  capture_first=True)
    print("Running Option A scenario...")
    optionA, optionA_details= run_R_replications(R, n_passengers_2b, arrival_lambda_2b, scenarios["Option A"], capture_first=True)
    print("Running Option B scenario...")
    optionB, optionB_details = run_R_replications(R, n_passengers_2b, arrival_lambda_2b, scenarios["Option B"], capture_first=True)

    results_df = compare_scenarios_table(baseline, optionA, optionB)
    results_df.to_csv("scenario_comparison.csv", index=False)
    latex_table = results_df.to_latex(index=False,
                                    caption="Scenario Comparison",
                                    label="tab:scenario_comparison",
                                    float_format="%.2f",
                                    column_format="lcccccc")
    with open("scenario_comparison.tex", "w") as f:
        f.write(latex_table)
    print("\nScenario Comparison Results:")
    print(results_df)

    # plot_boxplot_with_mean(baseline, optionA, optionB)
    # plot_violin_and_ci(baseline, optionA, optionB)
    scenario_comparison_plots(baseline, optionA, optionB)


    print("\nGenerating detailed plots for first replication of each scenario...")

    if baseline_details and len(baseline_details['waiting_times']) > 0:
        print(f"Plotting Baseline (first replication) - {len(baseline_details['waiting_times'])} passengers completed...")
        simulation_plots(**baseline_details)
    else:
        print("No baseline data to plot!")

    if optionA_details and len(optionA_details['waiting_times']) > 0:
        print(f"Plotting Option A (first replication) - {len(optionA_details['waiting_times'])} passengers completed...")
        simulation_plots(**optionA_details)
    else:
        print("No Option A data to plot!")

    if optionB_details and len(optionB_details['waiting_times']) > 0:
        print(f"Plotting Option B (first replication) - {len(optionB_details['waiting_times'])} passengers completed...")
        simulation_plots(**optionB_details)
    else:
        print("No Option B data to plot!")

