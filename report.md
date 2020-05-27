# Report: Project2-Continuous-Control


[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

### Learning Algorithm

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent! 


The hyperparameters used for training:
- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 0.999             # for soft update of target parameters
`target_param.data.copy_(tau*target_param.data + (1.0-tau)*param.data)`
- LR_ACTOR = 5e-4         # learning rate of the actor
- LR_CRITIC = 5e-4        # learning rate of the critic
- WEIGHT_DECAY = 0.0      # L2 weight decay
- EPSILON = 1.0           # explore->exploit noise process added to act step
- EPSILON_DECAY = 0.99    # decay rate for noise process
- UPDATE_EVERY = 1        # how often to update the target network
- LEARN_NUM = 1



The expected outputs of this implement are:
- critic network weights: `checkpoint_critic.pth`
- actor network weights: `checkpoint_actor.pth`

### Plot of rewards
![Image description](download.png)

### Ideas for Future Work
- implement Proximal Policy Optimization (PPO) for better performance
- try prioritized experience replay
- try N-step returns
- try different agents: D4PG, TD3, SAC, PlanNet and Dreamer.


