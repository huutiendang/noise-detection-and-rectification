import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

def calculate_mean_stddev(lst):
    num_lists = len(lst)
    list_length = len(lst[0])

    means = np.zeros(list_length)
    stddevs = np.zeros(list_length)

    for i in range(list_length):
        elements = [sublist[i] for sublist in lst]
        means[i] = np.mean(elements)
        stddevs[i] = np.std(elements)

    return means, stddevs

def read_json(file):
    data = json.load(open(file))
    return data

def plot(key, SC, NM, CE, KNN, std_SC, std_NM, std_CE, std_KNN):
# Sample data
    x = np.arange(1, 11)
    # y = np.arange(1, 11)
    # Plotting
    plt.plot(x, SC, label='SC', marker='.')
    plt.fill_between(x, SC - std_SC, SC + std_SC, alpha=0.15)

    plt.plot(x, NM, label='NM', marker='.')
    plt.fill_between(x, NM - std_NM, NM + std_NM, alpha=0.15)

    plt.plot(x, CE, label='CE', marker='.')
    plt.fill_between(x, CE - std_CE, CE + std_CE, alpha=0.15)

    plt.plot(x, KNN, label='KNN', marker='.')
    plt.fill_between(x, KNN - std_KNN, KNN + std_KNN, alpha=0.15)

    plt.xlabel('Percent')
    plt.ylabel('Error Detection Accuracy')
    plt.title('{}'.format(str(key)))
    plt.legend()
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.ylim(0.1, 1.0)
    # plt.yticks(np.arange(min(y), max(y)+1, 1))
    plt.grid(True) 
    plt.savefig("snippets_output/{}.png".format(str(key)))
    plt.clf()





# Example usage


if __name__ == "__main__":
    data = read_json("outputs/snippets/output_snippets.json")
    experiments = {}
    for key, value in data.items():
        exps = {}
        for k, v in value.items():
            exps[k] = calculate_mean_stddev(v)
            # print(calculate_mean_stddev(v))
        experiments[key] = exps
    
    for key, value in experiments.items():
        # SC, std_SC, NM, std_NM, CE, std_CE = [], [], [], [], [], []
        SC = value["self_confidence"][0]
        std_SC = value["self_confidence"][1]
        NM = value["normalized_margin"][0]
        std_NM = value["normalized_margin"][1]
        CE = value["confidence_weighted_entropy"][0]
        std_CE = value["confidence_weighted_entropy"][1]
        KNN = value["knn"][0]
        std_KNN = value["knn"][1]
       
        plot(key, SC, NM, CE, KNN, std_SC, std_NM, std_CE, std_KNN)
    


    # Plot density distributions
    # list1 = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    # list2 = [2, 3, 4, 4, 5, 5, 5, 6, 6]

    # # Calculate density for list 1
    # # Plot histograms
    # sns.kdeplot(list1, shade=True, label='List 1')
    # sns.kdeplot(list2, shade=True, label='List 2')

    # # Compute the means of each list
    # mean_list1 = sum(list1) / len(list1)
    # mean_list2 = sum(list2) / len(list2)

    # # Add vertical lines for the means
    # plt.axvline(mean_list1, color='red', linestyle='--', label='Mean of List 1')
    # plt.axvline(mean_list2, color='blue', linestyle='--', label='Mean of List 2')

    # plt.xlabel('Values')
    # plt.ylabel('Density')
    # plt.title('Density Distribution with Means')
    # plt.legend()
    # plt.show()