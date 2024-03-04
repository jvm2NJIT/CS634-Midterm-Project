import itertools
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

def findsubsets(s, k):
    '''
    Returns all k-itemsets from list s
    Parameters:
        s (list): List of items
        k (int): Size of subsets you wish to find
    Returns:
        subsets (list): List of k-subsets
    '''
    return list(itertools.combinations(s, k))

def generate_possible_rules(freq):
    '''
    Returns possible association rules from an itemset
    Parameters:
        freq (tuple): Frequent itemset
    Returns:
        possible_rules (list): List of possible rules
    '''
    n = len(freq)
    possible_rules = []
    for i in range(1, n):
        for combo in itertools.combinations(freq, i):
            tuple1 = combo
            tuple2 = tuple(item for item in freq if item not in combo)
            possible_rules.append([tuple1, tuple2])
    return possible_rules

def brute_force(items, filename, support, confidence):
    '''
    Uses brute force method to generate frequent items and generate association rules
    Paramaters:
        items (list): Sorted list of items
        filename (str): Filename of database you want to read from
        support (float): Desired support level for finding frequent itemsets
        confidence (float): Desired support level for finding association rules
    Returns:
        [freq_itemsets, rules, sups, cons] (list): A list of frequent items, a list of association rules, and their corresponding support values and confidence values
    '''
    df = pd.read_csv(filename)
    freq_itemsets = []
    supports = {}
    sups = []
    cons = []

    for i in range (1, 11):
        i_sets = findsubsets(items, i)
        for subset in i_sets:
            freq_count = 0
            for ind in df.index:
                match = True
                for item in subset:
                    if not df[item][ind]:
                        match = False
                        break
                if match:
                    freq_count = freq_count + 1

            supports.update({subset: freq_count / 20})
            if freq_count / 20 >= support:
                freq_itemsets.append(subset)
                sups.append(freq_count / 20)
    
    rules = []
    for freq in freq_itemsets:
        possible_rules = generate_possible_rules(freq)
        for p in possible_rules:
            X = p[0]
            Y = p[1]
            X_and_Y = tuple(sorted(X+Y))
            if supports[X_and_Y] / supports[X] >= confidence:
                rules.append(p)
                cons.append(supports[X_and_Y] / supports[X])

    
    return([freq_itemsets, rules, sups, cons])

print("Welcome to the Transaction Manager!\n")
print("Databases on file:")
print("Database1.csv")
print("Database2.csv")
print("Database3.csv")
print("Database4.csv")
print("Database5.csv\n")

# User inputs filename
filename = ""
databases = ['Database1.csv', 'Database2.csv', 'Database3.csv', 'Database4.csv', 'Database5.csv']
while filename not in databases:
    filename = input("Enter the full name of the database you'd like to analyze: ")

# User inputs support
print("\n")
support = 0
while True:
    try:
        support = float(input("Enter your desired support (0 <= support <= 1): "))       
    except ValueError:
        continue
    else:
        if support >= 0 and support <= 1:
            break 

# User inputs confidence
print("\n")
confidence = 0
while True:
    try:
        confidence = float(input("Enter your desired confidence (0 <= con <= 1): "))       
    except ValueError:
        continue
    else:
        if confidence >= 0 and confidence <= 1:
            break
     
# Find frequent itemsets and association rules using - Brute Force
items = sorted(['Shampoo', 'Apple', 'Banana', 'Milk', 'Eggs', 'Soap', 'Bacon', 'Sugar', 'Water', 'Yogurt'])
bf_start = time.time()
bf = brute_force(items, filename, support, confidence)
bf_time = time.time() - bf_start
print('-----------------------------------------------------------')
print('Brute Force')
print('Time Taken: {0} seconds\n'.format(bf_time))

bf_freq = bf[0]
bf_assoc = bf[1]
bf_sups = bf[2]
bf_cons = bf[3]

for i in range(0, len(bf_freq)):
    print('{0}: Support {1}'.format(bf_freq[i], bf_sups[i]))
print('\n')
for i in range(0, len(bf_assoc)):
    print('{0} --> {1}: Confidence {2}'.format(bf_assoc[i][0], bf_assoc[i][1], bf_cons[i]))

# Apriori
df = pd.read_csv(filename)
df.drop(columns=df.columns[0], axis=1, inplace=True)
apr_start = time.time()
a = apriori(df, min_support=support, use_colnames=True)
if not a.empty:
    apr_rules = association_rules(a, metric='confidence', min_threshold=confidence)
apr_time = time.time() - apr_start
print('-----------------------------------------------------------')
print('Apriori')
print('Time Taken: {0} seconds\n'.format(apr_time))
for index, row in a.iterrows():
    print('{0}: Support {1}'.format(tuple(row['itemsets']), row['support']))
print('\n')
if not a.empty:
    for index, row in apr_rules.iterrows():
        print('{0} --> {1}: Confidence {2}'.format(tuple(row['antecedents']), tuple(row['consequents']), row['confidence']))

# FP Tree
fp_start = time.time()
fp = fpgrowth(df, min_support=support, use_colnames=True)
if not fp.empty:
    fp_rules = association_rules(fp, metric='confidence', min_threshold=confidence)
fp_time = time.time() - fp_start
print('-----------------------------------------------------------')
print('FP Tree')
print('Time Taken: {0} seconds\n'.format(fp_time))
for index, row in fp.iterrows():
    print('{0}: Support {1}'.format(tuple(row['itemsets']), row['support']))
print('\n')
if not fp.empty:
    for index, row in fp_rules.iterrows():
        print('{0} --> {1}: Confidence {2}'.format(tuple(row['antecedents']), tuple(row['consequents']), row['confidence']))

# Summary of Results
print('\n-----------------------------------------------------------')
print('{0}  ---  Support: {1}  ---  Confidence: {2}'.format(filename, support, confidence))
print('Runtime of the 3 methods:')
print('Brute Force: {0} seconds'.format(bf_time))
print('Apriori: {0} seconds'.format(apr_time))
print('FP Tree: {0} seconds\n'.format(fp_time))