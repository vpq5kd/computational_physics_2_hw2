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
    temp_array = np.linspace(0.1, 5.1, 1000)[::-1]
    energy_array = []
    energy_array_squared = []
    magnetization_array = [] 
    
    ising_start_state = np.random.choice([-1,1], size=n)
    
    state = ising_start_state
    for T in temp_array:
        temp_energy_array = []
        for i in range(0, 10000):
            index_to_flip = rng.integers(n)
            new_state = state.copy()
            new_state[index_to_flip] = -new_state[index_to_flip]

            starting_state_energy = energy(state, n) 
            new_state_energy = energy(new_state, n)
            delta_energy = new_state_energy - starting_state_energy
            probability = transition_probability(delta_energy, n, T) 
            
            accept = rng.random() < probability
            if accept: 
                state = new_state
            temp_energy_array.append(energy(state,n)/n)
        
        temp_energy_array_np = np.array(temp_energy_array)
        energy_array.append(np.mean(temp_energy_array_np))
        energy_array_squared.append((np.mean(temp_energy_array_np**2)))
        magnetization_array.append(magnetization(state))

    return temp_array, energy_array, energy_array_squared, magnetization_array

def main():
    N = 50
    temp_array, energy_array, energy_array_squared, magnetization_array = metropolis(N)
  
    energy_array_np = np.array(energy_array)
    energy_array_squared_np = np.array(energy_array_squared)
    specific_heat_array = (energy_array_squared_np - energy_array_np**2)/N*temp_array
    plt.figure()
    fig, ax = plt.subplots(1,3)
    ax[0].plot(temp_array, energy_array, 'o')
    ax[1].plot(temp_array, specific_heat_array, 'o')
    plt.show()

main()



        
