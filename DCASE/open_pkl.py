import pickle

with open('model\score_distr_human.pkl', 'rb') as file:
    data = pickle.load(file)
    print(data)

#[2.4019031488488194, 17.540299416050956, 12.058855942187428]