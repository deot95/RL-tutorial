import numpy as np
import matplotlib.pyplot as plt

class grid_world():
    def __init__(self):
        dimW, dimH = 8, 8
        self._dims = (dimW,dimH)
        self.num_states, self.num_actions = dimW*dimH, 4
        self.state_dim, self.action_dim = 2, 1
        self._states = []
        for i in range(dimH):
            for j in range(dimW):
                self._states.append([i,j])

        self._target = dimW*dimH -1
        self._obstacles = [11,12,19,20,25,26,27,28,29,30,33,34,35,36,37,38,43,44,51,52]
        reserved_pos = self._obstacles+[self._target]
        self._possible_positions = np.delete(np.arange(0,dimW*dimH),reserved_pos)
        self._window = None #Render window
        self.reset()

    def reset(self,random_initial=False,coords=False):
        if not random_initial:
            self._agent = 0
        else:
            self._agent = np.random.choice(self._possible_positions)
        if(not coords):
            return self._agent
        else:
            return self.get_coords()

    def observe(self):
        return self._agent, self._target

    def get_coords(self):
        coords = self._states[self._agent]
        return np.array(coords).reshape(self.state_dim,1)

    def step(self,action,coords=False):
        assert 0<=action<=3, "Invalid action. The action must be an integer within [0,3]"
        self.move(action)
        state,target = self.observe()
        reward = self.reward()
        done = True if (state==target) else False
        if(coords):
            state = self.get_coords()
        return state, reward, done

    def move(self,action):
        #ACTIONS: 0-UP, 1-RIGHT, 2-DOWN, 3-LEFT
        y,x = self._states[self._agent]
        if action==0:
            new_x = x
            new_y = y-1
        elif action==1:
            new_x = x+1
            new_y = y
        elif action==2:
            new_x = x
            new_y = y+1
        elif action==3:
            new_x = x-1
            new_y = y
        new_x = np.clip(new_x,0,self._dims[0]-1)
        new_y = np.clip(new_y,0,self._dims[1]-1)
        new_state = self._states.index([new_y,new_x])
        if new_state not in self._obstacles:
            self._agent = new_state

    def reward(self):
        state,target = self.observe()
        if(state==target):
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def get_map(self):
        self._world = np.zeros((self._dims[1],self._dims[0]))
        for cell in self._obstacles:
            y,x = self._states[cell]
            self._world[y,x] = 1
        y,x = self._states[self._target]
        self._world[y,x] = 2
        y,x = self._states[self._agent]
        self._world[y,x] = 3
        return self._world

    def render(self):
        if self._window == None:
            plt.ion()
            self._window = plt.figure(figsize=(3,3))
        map_env = self.get_map()
        plt.clf()
        plt.imshow(map_env,cmap='viridis',vmin=0,vmax=3)
        plt.axis('off')
        self._window.canvas.draw()
        self._window.canvas.flush_events()
        return self.get_map()

