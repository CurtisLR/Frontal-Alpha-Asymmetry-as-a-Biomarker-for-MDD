import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


POWER_LINE_FREQ = 50
# We are interested in Alpha waves (8-13 Hz)
FREQ_BANDS = {"Alpha": (8, 13)}


def process_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload = True)

    #Clean names
    channel_rename_dict = dict()
    for ch in raw.ch_names:
        channel_rename_dict[ch] = ch.split(' ')[1].split('-')[0]

    raw.rename_channels(channel_rename_dict)
    brain_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3',
        'Pz', 'P4', 'T6', 'O1', 'O2'
    ]
    #Pick only these channels (relevant ones)
    raw.pick_channels(brain_channels)
    raw.set_montage('standard_1020')
    #raw.plot_sensors(show_names=True)

    #Filter
    raw.filter(l_freq=1.0, h_freq=40.0, verbose='error')
    raw.notch_filter(freqs=POWER_LINE_FREQ, verbose='error')

    #Epoch: split into 4 second epochs. Typical length for
    #resting state EEG
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose='error')

    #Artifact removal (Dropping bad chunks)
    #If any chunk has a swing bigger than 100 microvolts, it's probably an eyeblink.
    reject_criteria = dict(eeg=100e-6)
    epochs.drop_bad(reject=reject_criteria, verbose='error')

    #Transform into a power tensor using Welch's method; store the tensor in psd_data
    spectrum = epochs.compute_psd(method='welch', fmin=8, fmax=13, verbose='error')
    psd_data = spectrum.get_data(return_freqs=False)

    #Since this is a personal project I will provide a brief explanation (mostly for myself to remember)
    #The power in frequency bins (axis 2) are averaged for every single channel
    #in every single epoch -> each channel x epoch then has a single power value

    #Then for every single channel, the power value for each
    #epoch is averaged -> each channel contains a single power value

    mean_power_per_channel = np.mean(psd_data, axis=(0, 2))

    #Create a simple dictionary to return the results
    #Channel names are mapped to their power values
    channel_names = epochs.ch_names
    power_dict = dict(zip(channel_names, mean_power_per_channel))

    return power_dict

#first_file = process_file(file_path)
#print(first_file["F3"])

all_data = []  # This will hold our rows

#Define where data folders are
data_folders = {
    "Healthy": Path("data/Healthy"),
    "MDD": Path("data/MDD")
}



for group_name, folder_path in data_folders.items():
    #Find every .edf file in the folder
    for file_path in folder_path.glob("*.edf"):

        if "EC" not in file_path.name: #We only want eyes closed
            continue
        
        print(f"Processing: {file_path.name}")

        try:
            power_values = process_file(file_path)

            #Extract F3 and F4 power values
            f3 = power_values['F3']
            f4 = power_values['F4']

            #Calculate FAA: ln(Right) - ln(Left)
            faa = np.log(f4) - np.log(f3)

            #Create a row for this person that will be used for data frame
            row = {
                "Subject_ID": file_path.stem,
                "Group": group_name, #Healthy or MDD
                "F3_Power": f3,
                "F4_Power": f4,
                "FAA": faa
            }

            all_data.append(row)

        except Exception as e:
            print(f"Could not process {file_path.name}: {e}")

#Create dataframe and save to csv file
df = pd.DataFrame(all_data)
df.to_csv("faa_analysis_results.csv", index=False)

print(df['Group'].value_counts())

#Statistics
mdd_faa = df[df['Group'] == 'MDD']['FAA']
healthy_faa = df[df['Group'] == 'Healthy']['FAA']

#T test
t_stat, p_val = stats.ttest_ind(mdd_faa, healthy_faa, equal_var=False)
print("\nSTATISTICAL RESULTS")
print(f"Mean FAA (Healthy): {healthy_faa.mean():.4f}")
print(f"Mean FAA (MDD):     {mdd_faa.mean():.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_val:.4f}")


#Visualisation

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7))

#Boxplot (The main distribution)
#showfliers=False hides the default outlier diamonds so we can draw our own dots
#Show where the mean is
ax = sns.boxplot(x='Group', y='FAA', data=df,
                 hue='Group', legend=False, # Fixed FutureWarning by assigning hue
                 palette="Set2", showfliers=False, width=0.5,
                 showmeans=True,
                 meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black"})

#Draw the Stripplot (The individual dots)
sns.stripplot(x='Group', y='FAA', data=df,
              color='black', alpha=0.5, jitter=True, size=3)

plt.axhline(0, color='red', linestyle='--', alpha=0.5)

#Customize Titles and Labels
plt.title('Figure 1: Frontal Alpha Asymmetry in MDD vs. Healthy Controls',
          fontsize=14, pad=20)
plt.xlabel('Group', fontsize=12, labelpad=10)
plt.ylabel('FAA Score (ln F4 - ln F3)', fontsize=12)

#Add the P-Value Annotation
max_y = df['FAA'].max()
plt.text(0.5, max_y + 0.05, f'p = {p_val:.3f}',
         ha='center', fontsize=11, fontweight='bold')

legend_elements = [
    Line2D([0], [0], marker='^', color='w', label='Mean',
           markerfacecolor='white', markeredgecolor='black', markersize=10),
    Line2D([0], [0], color='red', linestyle='--', label='Symmetry (Left=Right)')
]
plt.legend(handles=legend_elements, loc='upper right')

sns.despine(offset=10, trim=True)
plt.tight_layout()
plt.show()
