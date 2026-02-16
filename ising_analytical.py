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
        for i in range(0, 20000):
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
        print(T)
        temp_energy_array_np = np.array(temp_energy_array)
        energy_array.append(np.mean(temp_energy_array_np))
        energy_array_squared.append((np.mean(temp_energy_array_np**2)))
        
        temp_magnetization_array_np = np.array(temp_magnetization_array)
        magnetization_array.append(np.mean(temp_magnetization_array_np))
        magnetization_array_squared.append(np.mean(temp_magnetization_array_np**2))
        
    return temp_array, energy_array, energy_array_squared, magnetization_array, magnetization_array_squared

def lambda_positive(T, J, H):
    B = 1 / T
    return (
        np.exp(B * J) * np.cosh(B * H)
        + np.sqrt(
            np.exp(2 * B * J) * np.sinh(B * H)**2
            + np.exp(-2 * B * J)
        )
    )
def lambda_negative(T,J,H):
    B = 1 / T
    return (
        np.exp(B * J) * np.cosh(B * H)
        - np.sqrt(
            np.exp(2 * B * J) * np.sinh(B * H)**2
            + np.exp(-2 * B * J)
        )
    )
def energy_analytical(T, J, H, N):
    lambda_p = lambda_positive(T, J, H)
    lambda_m = lambda_negative(T, J, H)
    
    Z = lambda_p**N + lambda_m**N
    
    numerator = (
        lambda_p**(N-1) * lambda_m
        + lambda_m**(N-1) * lambda_p
    )
    
    return -J * numerator / Z
def specific_heat_analytical(T, J, H, N):
    B = 1 / T
    
    lambda_p = lambda_positive(T, J, H)
    lambda_m = lambda_negative(T, J, H)
    
    Z = lambda_p**N + lambda_m**N
    
    term = (
        lambda_p**(2*N - 2)
        - lambda_m**(2*N - 2)
        + 4*(N - 1)*(lambda_p * lambda_m)**(N - 2)
    )
    
    return 4 * (B * J)**2 * term / (Z**2)
def magnetization_susceptibility_analytical(T, J, H, N):
    B = 1 / T

    lambda_p = lambda_positive(T, J, H)
    lambda_m = lambda_negative(T, J, H)

    Z = lambda_p**N + lambda_m**N

    numerator = (
        lambda_p**(N - 1) * (np.exp(B * J) + np.exp(3 * B * J))
        + lambda_m**(N - 1) * (np.exp(B * J) - np.exp(3 * B * J))
    )

    return B * numerator / Z

def main():
    N = 50
    temp_array, energy_array, energy_array_squared, magnetization_array, magnetization_array_squared = metropolis(N)
  
    energy_array_np = np.array(energy_array)
    energy_array_squared_np = np.array(energy_array_squared)
    specific_heat_array = N*(energy_array_squared_np - energy_array_np**2)/(temp_array**2)
    
    magnetization_array_np = np.array(magnetization_array)
    magnetization_array_squared_np = np.array(magnetization_array_squared)
    magnetic_susceptibility_array = (magnetization_array_squared_np - magnetization_array_np**2)/(N*temp_array)
    
    temp_array_np = np.array(temp_array)
    fig, ax = plt.subplots(1,3)
    marker_size = 3
    ax[0].plot(temp_array, energy_array, 'o',markersize = marker_size, markerfacecolor='none',markeredgecolor='purple',label='MCMC')
    ax[0].set_xlabel("T")
    ax[0].set_ylabel("E",rotation=0,labelpad=15)
    ax[0].plot(temp_array_np, energy_analytical(temp_array_np,1,0,N),color='orange',label='Exact')
    ax[1].plot(temp_array, specific_heat_array, 'o', markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[1].set_xlabel("T")
    ax[1].set_ylabel("C",rotation=0,labelpad=15)
    ax[1].plot(temp_array_np, specific_heat_analytical(temp_array_np,1,0,N),color='orange')
    ax[2].plot(temp_array, magnetic_susceptibility_array, 'o', markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[2].set_yscale("log")
    ax[2].set_xlabel("T")
    ax[2].set_ylabel(r"$\chi$",rotation=0,labelpad=15)
    ax[2].plot(temp_array_np, magnetization_susceptibility_analytical(temp_array_np,1,0,N),color='orange')
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("ising_with_analytical.png")
    plt.show()
    
    fig, ax = plt.subplots(1,4)
    marker_size = 3
    ax[0].plot(temp_array, energy_array, 'o',markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[0].set_xlabel("T")
    ax[0].set_ylabel("E",rotation=0,labelpad=15)
    ax[1].plot(temp_array, specific_heat_array, 'o', markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[1].set_xlabel("T")
    ax[1].set_ylabel("C",rotation=0,labelpad=15)
    ax[2].plot(temp_array, magnetic_susceptibility_array, 'o', markersize = marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[2].set_yscale("log")
    ax[2].set_xlabel("T")
    ax[2].set_ylabel(r"$\chi$",rotation=0,labelpad=15)
    ax[3].plot(temp_array, magnetization_array_np/N, 'o', markersize=marker_size, markerfacecolor='none',markeredgecolor='purple')
    ax[3].set_xlabel("T")
    ax[3].set_ylabel("M",rotation=0,labelpad=15)
    plt.tight_layout()
    plt.savefig("ising_without_analytical.png")
    plt.show()

main()
