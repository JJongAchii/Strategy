import pandas as pd

#allo = pd.read_excel("us_allocation_case.xlsx", parse_dates=['date'])

import itertools

lst = [1, 1, -1, -1]
unique_permutations = set(itertools.permutations(lst))
print(unique_permutations)

id = 0
for perm in unique_permutations:
    wtilt = {}
    for state in ["expansion", "slowdown", "contraction", "recovery"]:
        wtilt[state] = {}
        wtilt[state]["equity"] = perm[0]
        wtilt[state]["fixedincome"] = perm[1]
        wtilt[state]["alternative"] = perm[2]
        wtilt[state]["liquidity"] = perm[3]

    print(wtilt)

    print(id)
    id = id + 1