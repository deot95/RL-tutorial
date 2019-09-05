import numpy as np
from collections import deque
import random

class tabular_Qlearning():
    def __init__(self,num_states,num_actions,lr=0.1):
        self._num_states = num_states
        self._num_actions = num_actions
        self._gamma = 0.99
        self._initial_alpha = lr
        self.reset()

    def reset(self,initial_epsilon=1.0):
        self.Qtable = np.zeros((self._num_states,self._num_actions))
        self._epsilon = initial_epsilon
        self._alpha = self._initial_alpha

    def greedy(self,state):
        return np.argmax(self.Qtable[state,:])

    def epsilon_greedy(self,state):
        if np.random.random() <= self._epsilon:
            return np.random.randint(0,4)
        else:
            return self.greedy(state)

    def epsilon_decay(self,rate=0.999,min=0.01):
        self._epsilon = np.clip(rate*self._epsilon,min,1.0)

    def alpha_decay(self,rate=0.999,min=0.01):
        self._alpha = np.clip(rate*self._alpha,min,self._initial_alpha)

    def train(self,state,action,next_state,reward,done):
        if not done:
            BE = reward + self._gamma*np.amax(self.Qtable[next_state,:]) - self.Qtable[state,action]
        else:
            BE = reward - self.Qtable[state,action]
        self.Qtable[state,action] += self._alpha*BE
        return 0.5*np.square(BE)

    def get_policy(self):
        return np.argmax(self.Qtable,axis=1)

class double_deep_Qlearning():
    def __init__(self,dims=[2,15,15,4],lr=0.01):
        self._dims = dims #list with layer dimensions: [input, hidden1, hidden2, ..., output]
        self._n_layers = len(dims)-1
        self._initial_lr = lr
        self._gamma = 0.99
        self._tau = 1e-3
        self.reset()

    def reset(self,fix_seed=False,seed=123):
        if(fix_seed):
            np.random.seed(seed)
        self._epsilon = 1.0
        self._lr = self._initial_lr
        self._weights, self._target_weights  = self.create_weights()

    def greedy(self,state):
        prediction = self.forward_pass(state)[-1]
        return np.argmax(prediction)

    def epsilon_greedy(self,state):
        if np.random.random() <= self._epsilon:
            return np.random.randint(0,self._dims[-1])
        else:
            return self.greedy(state)

    def epsilon_decay(self,rate=0.99,min=0.1):
        self._epsilon = np.clip(rate*self._epsilon,min,1.0)

    def learning_rate_decay(self,rate=0.99,min=1e-3):
        self._lr = np.clip(rate*self._lr,min,1.0)

    def train(self,batch):
        for i in range(len(batch)):
            state,action,next_state,reward,done = batch[i]
            if not done:
                target = reward + self._gamma*np.amax(self.forward_pass(next_state,self._target_weights)[-1])
            else:
                target = reward
            outs = self.forward_pass(state)
            target_f = outs[-1].copy()
            target_f[action] = target
            if(i==0):
                layers_outs = outs
                targets = target_f
            else:
                for j in range(self._n_layers+1):
                    layers_outs[j] = np.hstack((layers_outs[j],outs[j]))
                targets = np.hstack((targets,target_f))
        BE = self.backward_pass(targets,layers_outs)
        return BE

    def ReLU(self,x,deriv=False):
        if not deriv:
            return np.maximum(x,0.0)
        else:
            return np.maximum(np.sign(x),0.0)

    def linear(self,x,deriv=False):
        if not deriv:
            return x
        else:
            return 1.0

    def create_weights(self):
        weights,target_weights = [],[]
        for i in range(self._n_layers):
            #Fan in (XAVIER INITIALIZATION)
            fan_in = 1.0/np.sqrt(self._dims[i])
            W = 2.0*fan_in*np.random.random((self._dims[i+1],self._dims[i]))-fan_in    # Shape: (layer_i+1 x layer_i)
            B = 2.0*fan_in*np.random.random((self._dims[i+1],1))-fan_in                # Shape: (layer_i+1 x 1)
            weights.append((W,B))
            target_weights.append((W.copy(),B.copy()))
        return weights, target_weights

    def update_target_weights(self):
        for i,w in enumerate(self._target_weights):
            new_W = self._tau*self._weights[i][0] + (1.0-self._tau)*w[0]
            new_B = self._tau*self._weights[i][1] + (1.0-self._tau)*w[1]
            self._target_weights[i] = (new_W,new_B)

    def forward_pass(self,input_data,weights=None):
        assert input_data.shape[0]==self._dims[0], "Input must have shape: (in x batch)"
        if(weights is None):
            weights = self._weights
        layers = [input_data]
        for i in range(self._n_layers):
            if i<self._n_layers-1:
                layer_output = self.ReLU(np.dot(weights[i][0],layers[i])+weights[i][1]) # Shape: (layer_i x m)
            else:
                layer_output = self.linear(np.dot(weights[i][0],layers[i])+weights[i][1]) # Shape: (layer_i x m)
            layers.append(layer_output)
        return layers

    def backward_pass(self,target,layers):
        #LOSS
        assert target.shape[0]==self._dims[-1], "Target must have shape: (out x batch)"
        batch_size = target.shape[1]
        loss = np.square(target-layers[-1])/(2.0*batch_size)   # Shape: (out x m)
        grad_loss = -(target-layers[-1])/(batch_size)          # Shape: (out x m)

        grad_in = grad_loss
        weight_grads = [0]*len(self._weights)
        for i in reversed(range(1,self._n_layers+1)):
            if i==self._n_layers:
                grad_out = self.linear(layers[i],True)*grad_in          # Shape: (layer_i x m)
            else:
                grad_in = np.dot(self._weights[i][0].T,grad_in)         # Shape: (layer_i x m)
                grad_out = self.ReLU(layers[i],True)*grad_in            # Shape: (layer_i x m)
            grad_W = np.dot(grad_out,layers[i-1].T)                     # Shape: (layer_i x layer_(i-1))
            grad_B = np.sum(grad_out,axis=1).reshape(self._dims[i],1)   # Shape: (layer_i x 1)
            weight_grads[i-1] = (grad_W,grad_B)
            grad_in = grad_out.copy()

        # Update weights (simple gradient descent)
        for i,w in enumerate(self._weights):
            new_W = w[0] - self._lr*weight_grads[i][0]
            new_B = w[1] - self._lr*weight_grads[i][1]
            self._weights[i] = (new_W,new_B)
        return loss

class ReplayBuffer(object):
    def __init__(self, buffer_size=100000,state_dim=2,action_dim=1,random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self,s,a,s2,r,t):
        experience = (s,a,s2,r,t)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def clear(self):
        self.buffer.clear()
        self.count = 0