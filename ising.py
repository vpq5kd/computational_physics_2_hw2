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
    return np.min([1, np.exp(-delta_energy/T)])

def metropolis(n):
    temp_array = np.linspace(0.1, 5.1, 100)[::-1]
    energy_array = []
    energy_array_squared = []
    magnetization_array = [] 
    magnetization_array_squared = []

    ising_start_state = np.random.choice([-1,1], size=n)
    state = ising_start_state
    for T in temp_array:
        temp_energy_array = []
        temp_magnetization_array = []
        for i in range(0, 1000):
            index_to_flip = rng.integers(n)
            new_state = state.copy()
            new_state[index_to_flip] = -new_state[index_to_flip]

            #starting_state_energy = energy(state, n) 
            #new_state_energy = energy(new_state, n)
            
            i_left = (index_to_flip - 1) % n
            i_right = (index_to_flip + 1) % n
            delta_energy = 2 * state[index_to_flip] * (state[i_left] + state[i_right])
            probability = transition_probability(delta_energy, n, T) 
            
            accept = rng.random() < probability
            if accept: 
                state = new_state
            temp_energy_array.append(energy(state,n)/n)
            temp_magnetization_array.append(magnetization(state))

        temp_energy_array_np = np.array(temp_energy_array)
        energy_array.append(np.mean(temp_energy_array_np))
        energy_array_squared.append((np.mean(temp_energy_array_np**2)))
        
        temp_magnetization_array_np = np.array(temp_magnetization_array)
        magnetization_array.append(np.mean(temp_magnetization_array_np))
        magnetization_array_squared.append(np.mean(temp_magnetization_array_np**2))
        
    return temp_array, energy_array, energy_array_squared, magnetization_array, magnetization_array_squared

def main():
    N = 50
    temp_array, energy_array, energy_array_squared, magnetization_array, magnetization_array_squared = metropolis(N)
  
    energy_array_np = np.array(energy_array)
    energy_array_squared_np = np.array(energy_array_squared)
    specific_heat_array = N*(energy_array_squared_np - energy_array_np**2)/(temp_array**2)
    
    magnetization_array_np = np.array(magnetization_array)
    magnetization_array_squared_np = np.array(magnetization_array_squared)
    magnetic_susceptibility_array = (magnetization_array_squared_np - magnetization_array_np**2)/(N*temp_array)
    
    fig, ax = plt.subplots(1,3)
    marker_size = 3
    ax[0].plot(temp_array, energy_array, 'o',markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[0].set_xlabel("T")
    ax[0].set_ylabel("E")
    ax[1].plot(temp_array, specific_heat_array, 'o', markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[1].set_xlabel("T")
    ax[1].set_ylabel("C")
    ax[2].plot(temp_array, magnetic_susceptibility_array, 'o', markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[2].set_yscale("log")
    ax[2].set_xlabel("T")
    ax[2].set_ylabel(r"$\Chi$")
    
    plt.tight_layout()
    plt.show()

main()



        
