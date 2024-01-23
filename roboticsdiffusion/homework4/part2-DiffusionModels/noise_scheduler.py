from randomizer import DL_random
import numpy as np
import torch

class NoiseScheduler:

    def __init__(self, num_steps, beta_start = 0.0001, beta_end = 0.02):
        # initialize the beta parameters (variance) of the scheduler
        self.beta_start = beta_start
        self.beta_end = beta_end

        # number of inference steps (same as num training steps)
        self.num_steps = num_steps

        # linear schedule for beta
        self.betas = np.linspace(self.beta_start, self.beta_end, self.num_steps)

        ###########################################################
        # TODO: Compute alphas and alpha_bars (refer to DDPM paper)
        ###########################################################
        self.alphas = 1 - self.betas
        self.alpha_bars = np.ones(self.alphas.shape)
        self.alpha_bars[0] = self.alphas[0]
        for i in range(1,self.alphas.shape[0]):
          self.alpha_bars[i] = self.alpha_bars[i-1] * self.alphas[i]
        ###########################################################
        #                     END OF YOUR CODE                    #
        ###########################################################

        # convert to tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alphas = torch.from_numpy(self.alphas).to(self.device).float()
        self.alpha_bars = torch.from_numpy(self.alpha_bars).to(self.device).float()
        self.betas = torch.from_numpy(self.betas).to(self.device).float()

    def denoise_step(self, model_prediction, t, x_t, threshold = False, seed = None):
        """
        ** Use DL_random() to generate any random numbers **

        Implement a step of the reverse denoising process
        Args:
            model_prediction (torch.Tensor): the output of the noise prediction model (B, input_shape)
            t (int): the current timestep
            x_t (torch.Tensor): the previous timestep (B, input_shape)
            threshold (bool): whether to threshold x_0, implemented in part 2.3
        Returns:
            x_t_prev (torch.Tensor): the denoised previous timestep (B, input_shape)
        
        """

        x_t_prev = None
        if not threshold:
            #####################################
            # TODO: Implement a denoising step  #
            # Hint: 1 call to DL_random         #    
            #####################################
            z = DL_random(shape = x_t.shape, seed = seed).to(self.device).float()
            if t == 0:
              z = 0
            first = 1/(self.alphas[t]**.5)
            second = x_t - ((1-self.alphas[t])/((1-self.alpha_bars[t])**.5))*model_prediction.to(self.device).float()
            third = (self.betas[t] ** .5) * z
            x_t_prev = first * second + third

            #####################################
            #          END OF YOUR CODE         #
            #####################################
        
        else:
            ######################################################
            # TODO: Implement a denoising step with thresholding #
            #       Hint: the main difference is how you compute #
            #              the mean of the x_t_prev              #
            #       Hint: 1 call to DL_random                    #
            ######################################################
            z = DL_random(shape = x_t.shape, seed = seed).to(self.device).float()
            x0 = (x_t - ((1-self.alpha_bars[t])**.5) * model_prediction.to(self.device))/(self.alpha_bars[t]**.5).to(self.device).float()
            x0 = torch.clamp(x0, -1, 1)
            if t == 0: alpha_bar = 1
            else: alpha_bar = self.alpha_bars[t-1]
            first = (alpha_bar**.5)*self.betas[t] * x0/(1-self.alpha_bars[t])
            #first.to(self.device).float()
            second = (self.alphas[t]**.5)*(1 - alpha_bar)*x_t/(1-self.alpha_bars[t])
           # second.to(self.device).float()
           # beta = (1-alpha_bar)*self.betas[t]/(1-self.alpha_bars[t])
            mean = first + second
            x_t_prev = z*(self.betas[t]**.5) + mean
            ######################################################
            #                  END OF YOUR CODE                  #
            ######################################################

        return x_t_prev
        
    def add_noise(self, original_samples, noise, timesteps):
        """
        add noise to the original samples - the forward diffusion process.  
        Args:
            original_samples (torch.Tensor): the uncorrupted original samples (B, input_shape)
            noise (torch.Tensor): random gaussian noise (B, input_shape)
            timesteps (torch.Tensor): the timesteps for noise addition (B,)
        Returns:
            noisy_samples (torch.Tensor): corrupted samples with amount of noise added based 
                                          on the corresponding timestep (B, input_shape)
        """
        noisy_samples = None
        ###########################################
        # TODO: Implement forward noising process #
        ###########################################
        noisy_samples = torch.ones(original_samples.shape)
        for i in range(timesteps.shape[0]):
          mean = self.alpha_bars[timesteps[i]] ** .5
          cvar = (1 - self.alpha_bars[timesteps[i]]) ** .5
          noisy_samples[i] = original_samples[i] * mean + cvar * noise[i]
          
        
        ##########################################
        #          END OF YOUR CODE              #
        ##########################################

        return noisy_samples