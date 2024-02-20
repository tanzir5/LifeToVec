import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_lines(data_tuples):
    # Create an empty figure
    sns.set(style="whitegrid")#, rc={"axes.facecolor": "#f0f0f0"})

    plt.figure(figsize=(10, 6))
    max_y = float('-inf')
    min_y = float('inf')
    
    # Loop through each (X, Y) tuple and plot a line
    for i, (x, y, label) in enumerate(data_tuples):
        x = x[:-2]
        y = y[:-2]
        if len(y) == 0:
          continue
        x = np.array(x)
        #x += i*2
        #x += 2010
        sns.lineplot(x=x, y=y, label=label)
        print(max(y))
        max_y = max(max_y, max(y))
        min_y = min(min_y, min(y))
    print("SDF", max_y)
    #plt.ylim(min_y - 1e-2, max_y + 1e-2)

    # Add labels and title
    plt.xlabel('Years into future')
    plt.ylabel('R^2 for income')
    plt.title('Groningen income prediction')

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

# Example usage:
data_tuples = [
    (
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ], 
      [-0.012, .03, -.013, -.006, .074, .125, .107, .167, .077, .133, .079], 
      '2010'
    ),
    (
      [1, 2, 3, 4, 5, 6, 7, 8, 9], 
      [-.05, -.002, .025, .092, .103, .141, .154, .101, .066], 
      '2012'
    ),
    (
      [1, 2, 3, 4, 5, 6, 7,], 
      [.069, .055, .096, .145, .174, .142, .064], 
      '2014'
    ),
    (
      [1, 2, 3, 4, 5], 
      [.169, .152, .145, .102, .029], 
      '2016'
    ),
    (
      [1, 2, 3], 
      [.153, .15, .079], 
      '2018'
    ),
    (
      [1], 
      [.132], 
      '2020'
    ),
]

plot_lines(data_tuples)
