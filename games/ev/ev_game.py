import gymnasium as gym
from gymnasium.spaces import Discrete,Box
import numpy as np
import matplotlib.pyplot as plt


class EVGame(gym.Env):
    def __init__(self,env_config):
        self.env_config = env_config
        # Maximum Capacity of the parking site (parking spaces with ev charging)
        self.max_capacity = env_config['max_capacity']

        # Max power of the ev site (we assume each parking space can output the maximum power)
        self.max_power = env_config['max_power']

        # Number of maximum timesteps
        self.maximum_timesteps = env_config['maximum_timesteps']

        # Number of evs that will come through in this day
        self.number_of_evs = env_config['number_of_evs']

        # Number of Vacant Spots
        self.number_vacant_spots = self.max_capacity

        # Average battery capacity of ev cars (in kwh)
        self.average_battery = 60
        self.std = 40


        self.action_space = Discrete(2)
        self.observation_space = Box(low = 0, high = 500,shape=(6,))
        self.state = self.observation_space.sample()
        self.new_state = self.observation_space.sample()

        # Current timestep
        self.timestep = 0

        # Total power being consumed this hour
        self.total_power = 0

        #Evs currently in the parking site
        self.evs_parked = []

        self.shortest_time = 0

        self.action_mask = 0

        self.index = 0

        self.deterministic = True

        self.t_parks = []

        self.power_timesteps = []

        self.ev_arrivals_array = self.ev_arrivals()


    def ev_arrivals(self):
        if self.deterministic:
            return np.array([0,1,2,4])
        else:
            # Returns an array with the timesteps at which the evs will arrive
            return np.sort(np.random.choice(np.arange(0, self.maximum_timesteps), self.number_of_evs, replace=False))
    
    def find_shortest_time(self):
        if len(self.evs_parked)>=1:
            self.shortest_time = min(sublist[0] for sublist in self.evs_parked)
        else:
            self.shortest_time = 0

    def generate_evs(self):
        #self.t_parks = [5,3,2,0,1,0]
        self.t_parks = [5,3,2,0,1]
        self.power_timesteps = [5,5,30,0,30]

        shuffled_indices = list(range(len(self.t_parks)))
        np.random.shuffle(shuffled_indices)

        #            Create new shuffled lists using the shuffled indices
        self.t_parks = [self.t_parks[i] for i in shuffled_indices]
        self.power_timesteps = [self.power_timesteps[i] for i in shuffled_indices]

        self.t_parks.append(0)
        self.power_timesteps.append(0)


    def generate_ev(self):
        # Does a car arrive at this timestep
        if self.timestep in self.ev_arrivals_array:

            sample_car_battery = max(int(round(np.random.normal(self.average_battery,self.std))),10)

            # Timesteps to park there
            if self.maximum_timesteps - self.timestep >=2:
                t_park = min(np.random.randint(low=1,high=8),self.maximum_timesteps-self.timestep)
            else:
                t_park = 1

            # How much power per timestep will require (in kwh)
            power_timestep = int(round(sample_car_battery/t_park))

        else:
            #Since no car arrives here, both the t_park and the power_timestep are 0
            t_park = 0
            power_timestep = 0

        self.find_shortest_time()

        return np.array([t_park,power_timestep,self.max_power-self.total_power,self.shortest_time])
    
    def verify_subtract_and_remove(self):
        for sublist in self.evs_parked:
            if sublist[0] <= 0:
                self.total_power -= sublist[1]
                self.number_vacant_spots += 1
        self.evs_parked = [sublist for sublist in self.evs_parked if sublist[0]>0]

    
    def step(self, action):
        assert action == 0 or action == 1
        self.timestep += 1

        if self.maximum_timesteps - self.timestep == 0:
            done = True
        else:
            done = False
        
        if (action == 1 and self.total_power + self.state[2]>self.max_power) or (action == 1 and self.number_vacant_spots == 0):
            reward = -10000
            done = True
        elif (action == 0) or (action == 1 and self.state[1]==0):

            if len(self.evs_parked)>=1:
                self.evs_parked = [[item[0]-1,item[1]] for item in self.evs_parked]

            self.verify_subtract_and_remove()
            # Remove 1 hour from the remaining timesteps of all the parked evs

            if self.deterministic:
                self.find_shortest_time()
                self.state = np.concatenate((np.array([self.number_vacant_spots]),np.array([self.t_parks[self.timestep],self.power_timesteps[self.timestep],self.max_power - self.total_power,self.shortest_time])))
            else:
                new_ev = self.generate_ev()
                self.state = np.concatenate((np.array([self.number_vacant_spots]),new_ev))

            if (self.total_power + self.state[2]>self.max_power) or (self.number_vacant_spots == 0):
                self.action_mask = 1
            else:
                self.action_mask = 0

            self.state = np.concatenate((self.state,np.array([self.action_mask])))
            reward = 0
        elif action == 1:
            assert self.number_vacant_spots > 0

            # Remove 1 hour from the remaining timesteps of all the parked evs
            if len(self.evs_parked)>=1:
                self.evs_parked = [[item[0]-1,item[1]] for item in self.evs_parked]

            # Remove any ev from the list of the parked evs if it is already leaving
            self.verify_subtract_and_remove()
        

            # Add the new ev
            if self.state[1] >=1:
                self.evs_parked.append([self.state[1]-1,self.state[2]])

            # Add the power consumption of this EV to the total power being consumed per timestep
            self.total_power += self.state[2]
            
            # The reward is how much the EV owner pays the agent which is the number_timesteps*power_per_timestep (assuming 1 dollar per kwh)
            reward = self.state[1]*self.state[2]

            self.number_vacant_spots = self.state[0]-1

            if self.deterministic:
                self.find_shortest_time()
                self.state = np.concatenate((np.array([self.number_vacant_spots]),np.array([self.t_parks[self.timestep],self.power_timesteps[self.timestep],self.max_power - self.total_power,self.shortest_time])))
            else:
                new_ev = self.generate_ev()
                self.state = np.concatenate((np.array([self.number_vacant_spots]),new_ev))

            if (self.total_power + self.state[2]>self.max_power) or (self.number_vacant_spots == 0):
                self.action_mask = 1
            else:
                self.action_mask = 0

            self.state = np.concatenate((self.state,np.array([self.action_mask])))

        return self.state, reward, done, False, {}
            
    def reset(self, seed = None, options = None):
        self.number_vacant_spots = self.max_capacity
        self.timestep = 0
        self.total_power = 0
        self.shortest_time = 0
        self.evs_parked = []
        self.ev_arrivals_array = self.ev_arrivals()
        if self.deterministic:
            self.generate_evs()
            self.state = np.concatenate((np.array([self.number_vacant_spots]),np.array([self.t_parks[0],self.power_timesteps[0],self.max_power-self.total_power,self.shortest_time])))
        else:
            self.state = np.concatenate((np.array([self.number_vacant_spots]),self.generate_ev()))
        if (self.total_power + self.state[2]>self.max_power):
            self.action_mask = 1
        else:
            self.action_mask = 0
        self.state = np.concatenate((self.state,np.array([self.action_mask])))
        return self.state,{}
    

if __name__ == "__main__":
    env_config = {
        "max_capacity":2,
        "max_power":55,
        "maximum_timesteps":5,
        "number_of_evs":4
    }

    env = EVGame(env_config=env_config)
    np.random.seed(42)

    rewards_episodes_agents = []
    for i in range(5):
        rewards_episodes = []
        for i in range(42*15):
            rewards_episode = []
            state,_ = env.reset()
            while(True):
                if state[-1] == 0:
                    action = 1
                else:
                    action = 0
                state, reward, done, _, _ = env.step(action)
                rewards_episode.append(reward)
                if done:
                    break
            rewards_episodes.append(np.sum(rewards_episode))
        rewards_episodes_agents.append(rewards_episodes)

    plt.plot(np.mean(rewards_episodes_agents,axis = 0))
    plt.savefig("plot_rc")

    

    