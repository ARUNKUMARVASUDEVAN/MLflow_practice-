import os
 
n_estimators=[110,100,150,200,210]
max_Depth=[20,25,15,5,10]

for n in n_estimators:
    for m in max_Depth:
        os.system(f"python basic_ml_model.py -n{n} -m{m}")
        print(n,m)