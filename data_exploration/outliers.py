import pandas as pd
import matplotlib.pyplot as plt


rows = ['TH_SP15_GEN-APND', 'Brookings Dew Point Forecast']
df = pd.read_csv(r"data\\dataset_dropNA.csv")
df_copy = df[rows].copy()
ax = df_copy.plot()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)

plt.show()