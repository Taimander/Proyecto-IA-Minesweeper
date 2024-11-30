import pickle
import os
import matplotlib.pyplot as plt

md_names = ["episode", "reward", "steps", "epsilon", "time"]

metadata_to_plot = 2

metadata_file = 'merged_metadata.csv'

with open(metadata_file, 'r') as f:
    data = [line.strip().split(',') for line in f.readlines()]

episodes = [d[0] for d in data]
data_list = [d[metadata_to_plot] for d in data]

# Convert the y-axis data to float for proper plotting
data_list = [float(step) for step in data_list]

if metadata_to_plot == 2:
    new_ep = []
    new_data = []
    for i in range(1, len(data_list)):
        if data_list[i] > 1: # Remove games with one step
            new_ep.append(episodes[i])
            new_data.append(data_list[i])
    episodes = new_ep
    data_list = new_data

plt.figure(figsize=(10, 6))
plt.plot(episodes, data_list, label=md_names[metadata_to_plot])
plt.xlabel("Episode")
plt.ylabel(md_names[metadata_to_plot])
plt.title(f"{md_names[metadata_to_plot]} per Episode")
plt.legend()
plt.grid()
plt.ylim(sorted(data_list)[0], sorted(data_list)[-1])  # Ensure y-axis is ordered correctly
plt.show()
