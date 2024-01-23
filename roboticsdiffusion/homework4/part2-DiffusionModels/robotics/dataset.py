import numpy as np
import torch
import zarr

class PushTDataset():

    def __init__(self, dataset_path, batch_size = 256, action_horizon = 8, state_horizon = 2, shuffle = True):
        self.batch_size = batch_size
        self.action_horizon = action_horizon
        self.state_horizon = state_horizon
        self.normalization_stats = {}
        self.shuffle = shuffle
        self.data_loader = self.load_dataset(dataset_path)        

    # normalize data
    def get_data_stats(self, data, stats_key):
        """ Get min and max of data"""
        data = data.reshape(-1,data.shape[-1])
        stats = {
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
        self.normalization_stats[stats_key] = stats

    def normalize_data(self, data, stats_key):
        """ Normalize data to [-1, 1]"""
        stats = self.normalization_stats[stats_key]
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, stats_key):
        """ Unnormalize data from [-1, 1] to original range"""
        stats = self.normalization_stats[stats_key]
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data
    
    def load_dataset(self, dataset_path):
        """ 
            Construct the dataset and initialize the data loader
            Args:
                dataset_path (str): path to the dataset

            Returns:
                data_loader (torch.utils.data.DataLoader): data loader for the dataset
                    - each batch is a tuple of (actions, states)
                    - actions: (batch_size, diffusion model input shape)
                    - states: (batch_size, diffusion model condition dim)
        """
        raw_data = zarr.open(dataset_path, 'r')
        all_actions = raw_data['data']['action'][:]
        all_states = raw_data['data']['state'][:]
        episode_ends = raw_data['meta']['episode_ends'][:]

        self.get_data_stats(all_actions, 'action')
        self.get_data_stats(all_states, 'state')

        all_actions = self.normalize_data(all_actions, 'action')
        all_states = self.normalize_data(all_states, 'state')

        all_actions = torch.from_numpy(all_actions)
        all_states = torch.from_numpy(all_states)
        final_actions = []
        final_states = []
        for i in range(len(episode_ends)):
            # Initializing some variables that may be useful for you
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i-1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx
            min_start = -self.state_horizon + 1
            max_start = episode_length - 1
            for idx in range(min_start, max_start + 1):
                ###################################################################
                # TODO: Append datapoints to final_actions and final_states       #
                #       Hint: Use the variables defined above                     #
                ###################################################################
                if start_idx + idx < 0:
                  pad_a = all_actions[0].repeat(abs(idx), 1)
                  nonpadded_a = all_actions[start_idx: start_idx + self.action_horizon - abs(idx)]
                  pad_s = all_states[0].repeat(abs(idx), 1)
                  nonpadded_s = all_states[start_idx: start_idx + self.state_horizon - abs(idx)]
                  a = torch.cat((pad_a, nonpadded_a), 0)
                  s = torch.cat((pad_s, nonpadded_s), 0)
                  final_actions.append(a)
                  final_states.append(s.flatten())
                  #final_actions.append(all_actions[start_idx: start_idx + self.action_horizon])
                  #final_states.append(all_states[start_idx: start_idx + self.state_horizon].view(-1))
                else:
                  if start_idx + idx + self.action_horizon > len(all_actions):
                    
                    diff = start_idx + idx + self.action_horizon - len(all_actions)
                    pad_a = all_actions[-1].repeat(abs(diff), 1)
                    nonpadded_a = all_actions[start_idx + idx: len(all_actions)]
                    a = torch.cat((nonpadded_a, pad_a), 0)
                    final_actions.append(a)
                    
                    #beginning_action = min(start_idx + idx, len(all_actions) - self.action_horizon)
                    #beginning_state = min(start_idx + idx, len(all_actions) - self.state_horizon)
                  else:
                    curr = start_idx + idx
                    final_actions.append(all_actions[curr: curr + self.action_horizon])
                  if start_idx + idx + self.state_horizon > len(all_states):
                    diff = start_idx + idx + self.state_horizon - len(all_states)
                    pad_s = all_states[-1].repeat(abs(diff), 1)
                    nonpadded_s = all_states[start_idx + idx: len(all_states)]
                    s = torch.cat((nonpadded_s, pad_s), 0)
                    final_states.append(s.flatten())
                  else:
                    curr = start_idx + idx
                    final_states.append(all_states[curr: curr+ self.state_horizon].flatten())

                
                #################################################################
                #                         END OF YOUR CODE                     #
                #################################################################
        #print(episode_ends[-1])
        #print(final_actions[-1])
        #print(final_actions[-8])
        #print(final_actions[-9])
        #25601
        #25650
        inputs = torch.stack(final_actions, dim=0)
        conds = torch.stack(final_states, dim=0)
        combined_dataset = torch.utils.data.TensorDataset(inputs, conds)
        tmp = torch.utils.data.DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=True, num_workers = 1, persistent_workers = True)
        data_point = next(iter(tmp))
        print(data_point[0].shape)
        return tmp


