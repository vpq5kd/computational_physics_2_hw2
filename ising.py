import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

def energy(state,n):
  energy = 0
  for i in range(0,n):
    energy += state[i] * state[(i + 1) % n]
  return -energy

def magnetization(state):
    return np.sum(state)

def transition_probability(delta_energy,n,T):
    return (1/n)*np.min([1, np.exp(-delta_energy/T)])

def metropolis(n):
    temp_array = np.linspace(1e-6, 5.1, 1000)
    energy_array = []
    magnetization_array = [] 
    
    ising_start_state = np.random.choice([-1,1], size=n)
    
    state = ising_start_state
    for T in temp_array:
        index_to_flip = rng.integers(n)
        new_state = state.copy()
        new_state[index_to_flip] = -new_state[index_to_flip]

        starting_state_magnetization = magnetization(state)
        starting_state_energy = energy(state, n) 
        new_state_energy = energy(new_state, n)
        delta_energy = new_state_energy - starting_state_energy
        probability = transition_probability(delta_energy, n, T)

        energy_array.append(starting_state_energy)
        magnetization_array.append(starting_state_magnetization)
       
        accept = rng.random() < probability
        if accept: 
            state = new_state
    return temp_array, energy_array, magnetization_array

def main():
    N = 50
    temp_array, energy_array, magnetization_array = metropolis(N)
    
    plt.figure()
    plt.plot(temp_array, energy_array, 'o')
    plt.show()

main()



        
