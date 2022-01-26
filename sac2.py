from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core2 as core2
from spinup.utils.logx import EpochLogger
from double_cartpole_envs import DoubleCartPoleEnv


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):

        self.obs_buf = np.zeros(core2.combined_shape(size, obs_dim), dtype=np.float32)
        self.obsp_buf = np.zeros(core2.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core2.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):

        self.obs_buf[self.ptr] = obs
        self.obsp_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obsp=self.obsp_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items() }

def sac(_env, actor_critic=core2.MLPActorCritic, ac0_kwargs=dict(), ac1_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):

#def sac(env_fn, actor_critic=core2.MLPActorCritic, ac0_kwargs=dict(), ac1_kwargs=dict(), seed=0, 
#        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
#        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
#        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
#        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
            
        batch_size (int): Minibatch size for SGD.


        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = _env, _env
    obs_dim0 = env.observation_space0.shape
    act_dim0 = env.action_space0.shape[0]
    obs_dim1 = env.observation_space1.shape
    act_dim1 = env.action_space1.shape[0]
    

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit0 = env.action_space0.high[0]
    act_limit1 = env.action_space1.high[0]

    # Create actor-critic module and target networks
    ac0 = actor_critic(env.observation_space0, env.action_space0, **ac0_kwargs)
    ac_targ0 = deepcopy(ac0)
    ac1 = actor_critic(env.observation_space1, env.action_space1, **ac1_kwargs)
    ac_targ1 = deepcopy(ac1)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p0 in ac_targ0.parameters():
        p0.requires_grad = False
    for p1 in ac_targ1.parameters():
        p1.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params0 = itertools.chain(ac0.q01.parameters(), ac0.q02.parameters())
    q_params1 = itertools.chain(ac1.q11.parameters(), ac1.q12.parameters())

    # Experience buffer
    replay_buffer0 = ReplayBuffer(obs_dim=obs_dim0, act_dim=act_dim0, size=replay_size)
    replay_buffer1 = ReplayBuffer(obs_dim=obs_dim1, act_dim=act_dim1, size=replay_size)
    

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts0 = tuple(core2.count_vars(module) for module in [ac0.pi0, ac0.q01, ac0.q02])
    logger.log('\nNumber of parameters: \t pi0: %d, \t q01: %d, \t q02: %d\n'%var_counts0)
    var_counts1 = tuple(core2.count_vars(module) for module in [ac1.pi1, ac1.q11, ac1.q12])
    logger.log('\nNumber of parameters: \t pi1: %d, \t q11: %d, \t q12: %d\n'%var_counts1)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data0, data1):
        o0, a0, r, op0, d0 = data0['obs'], data0['act'], data0['rew'], data0['obsp'], data0['done']
        o1, a1, r, op1, d1 = data1['obs'], data1['act'], data1['rew'], data1['obsp'], data1['done']

        q01 = ac0.q01(o0,a0)
        q02 = ac0.q02(o0,a0)
        q11 = ac1.q11(o1,a1)
        q12 = ac1.q12(o1,a1)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            ap0, logp_ap0 = ac0.pi0(op0)

            # Target Q-values
            q01_pi_targ = ac_targ0.q01(op0, ap0)
            q02_pi_targ = ac_targ0.q02(op0, ap0)
            q0_pi_targ = torch.min(q01_pi_targ, q02_pi_targ)
            backup0 = r + gamma * (1 - d0) * (q0_pi_targ - alpha * logp_ap0)

            ap1, logp_ap1 = ac1.pi1(op1)

            # Target Q-values
            q11_pi_targ = ac_targ1.q11(op1, ap1)
            q12_pi_targ = ac_targ1.q12(op1, ap1)
            q1_pi_targ = torch.min(q11_pi_targ, q12_pi_targ)
            backup1 = r + gamma * (1 - d1) * (q1_pi_targ - alpha * logp_ap1)

        # MSE loss against Bellman backup
        loss_q01 = ((q01 - backup0)**2).mean()
        loss_q02 = ((q02 - backup0)**2).mean()
        loss_q0 = loss_q01 + loss_q02
        loss_q11 = ((q11 - backup1)**2).mean()
        loss_q12 = ((q12 - backup1)**2).mean()
        loss_q1 = loss_q11 + loss_q12

        # Useful info for logging
        q_info0 = dict(Q01Vals=q01.detach().numpy(),
                      Q02Vals=q02.detach().numpy())
        q_info1 = dict(Q11Vals=q11.detach().numpy(),
                      Q12Vals=q12.detach().numpy())

        return loss_q0, q_info0, loss_q1, q_info1

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data0, data1):
        o0 = data0['obs']
        pi0, logp_pi0 = ac0.pi0(o0)
        q01_pi0 = ac0.q01(o0, pi0)
        q02_pi0 = ac0.q02(o0, pi0)
        q_pi0 = torch.min(q01_pi0, q02_pi0)

        o1 = data1['obs']
        pi1, logp_pi1 = ac1.pi1(o1)
        q11_pi1 = ac1.q11(o1, pi1)
        q12_pi1 = ac1.q12(o1, pi1)
        q_pi1 = torch.min(q11_pi1, q12_pi1)

        # Entropy-regularized policy loss
        loss_pi0 = (alpha * logp_pi0 - q_pi0).mean()
        loss_pi1 = (alpha * logp_pi1 - q_pi1).mean()

        # Useful info for logging
        pi_info0 = dict(LogPi0=logp_pi0.detach().numpy())
        pi_info1 = dict(LogPi1=logp_pi1.detach().numpy())

        return loss_pi0, pi_info0, loss_pi1, pi_info1

    # Set up optimizers for policy and q-function
    pi_optimizer0 = Adam(ac0.pi0.parameters(), lr=lr)
    q_optimizer0 = Adam(q_params0, lr=lr)
    pi_optimizer1 = Adam(ac1.pi1.parameters(), lr=lr)
    q_optimizer1 = Adam(q_params1, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac0)
    logger.setup_pytorch_saver(ac1)

    def update(data0, data1):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer0.zero_grad()
        q_optimizer1.zero_grad()
        loss_q0, q_info0, loss_q1, q_info1 = compute_loss_q(data0, data1)
        #loss_q0, q_info0 = compute_loss_q(data0)
        #loss_q1, q_info1 = compute_loss_q(data1)
        loss_q0.backward()
        loss_q1.backward()
        q_optimizer0.step()
        q_optimizer1.step()
        
        #q_optimizer1.zero_grad()
        #loss_q1, q_info1 = compute_loss_q(data)
        #loss_q1.backward()
        #q_optimizer1.step()

        # Record things
        logger.store(LossQ0=loss_q0.item(), **q_info0)
        logger.store(LossQ1=loss_q1.item(), **q_info1)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p0 in q_params0:
            p0.requires_grad = False
        for p1 in q_params1:
            p1.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer0.zero_grad()
        pi_optimizer1.zero_grad()
        loss_pi0, pi_info0, loss_pi1, pi_info1 = compute_loss_pi(data0, data1)
        #loss_pi0, pi_info0 = compute_loss_pi(data0)
        #loss_pi1, pi_info1 = compute_loss_pi(data1)
        loss_pi0.backward()
        loss_pi1.backward()
        pi_optimizer0.step()
        pi_optimizer1.step()
       
        #pi_optimizer1.zero_grad()
        #loss_pi1, pi_info1 = compute_loss_pi(data)
        #loss_pi1.backward()
        #pi_optimizer1.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p0 in q_params0:
            p0.requires_grad = True
        for p1 in q_params1:
            p1.requires_grad = True

        # Record things
        logger.store(LossPi0=loss_pi0.item(), **pi_info0)
        logger.store(LossPi1=loss_pi1.item(), **pi_info1)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p0, p_targ0 in zip(ac0.parameters(), ac_targ0.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ0.data.mul_(polyak)
                p_targ0.data.add_((1 - polyak) * p0.data)
            for p1, p_targ1 in zip(ac1.parameters(), ac_targ1.parameters()):
                p_targ1.data.mul_(polyak)
                p_targ1.data.add_((1 - polyak) * p1.data)
    
    #def get_action(o0, o1, deterministic=False):
    #    return ac0.act(torch.as_tensor(o0, dtype=torch.float32), deterministic), ac1.act(torch.as_tensor(o1, dtype=torch.float32), deterministic)
        
    def get_action0(o, deterministic=False):
        return ac0.act0(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def get_action1(o, deterministic=False):
        return ac1.act1(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            #o0, o1, d0, d1, ep_ret, ep_len = test_env.reset(), test_env.reset(), False, False, 0, 0
            #o0, o1, d0, d1, ep_ret, ep_len = test_env.reset(), False, False, 0, 0
            o0, o1 = test_env.reset()
            d0, d1, ep_ret, ep_len = False, False, 0, 0
           
            while not(d0 or d1 or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                op0, op1, r, d0, d1, _ = test_env.step([a0,a1])
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    #o0, o1, ep_ret, ep_len = env.reset(), 0, 0
    o0, o1 = env.reset()
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        env.render();
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a0, a1 = get_action0(o0), get_action1(o1)
            #a0, a1 = get_action(o0, o1)
        else:
            a0, a1 = env.action_space0.sample(), env.action_space1.sample()

        # Step the env
        #action = [a0,a1]
        #op0, op1, r, d0, d1, _ = env.step(action)
        op0, op1, r, d0, d1, _ = env.step([a0,a1])

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d0 = False if ep_len==max_ep_len else d0
        d1 = False if ep_len==max_ep_len else d1

        # Store experience to replay buffer
        replay_buffer0.store(o0, a0, r, op0, d0)
        replay_buffer1.store(o1, a1, r, op1, d1)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o0 = op0
        o1 = op1

        # End of trajectory handling
        if d0 or d1 or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o0, o1 = env.reset()
            ep_ret, ep_len = 0, 0
            #o0, o1, ep_ret, ep_len = env.reset(), 0, 0
             

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch0 = replay_buffer0.sample_batch(batch_size)
                batch1 = replay_buffer1.sample_batch(batch_size)
                update(data0=batch0, data1=batch1)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q01Vals', with_min_and_max=True)
            logger.log_tabular('Q02Vals', with_min_and_max=True)
            logger.log_tabular('Q11Vals', with_min_and_max=True)
            logger.log_tabular('Q12Vals', with_min_and_max=True)
            logger.log_tabular('LogPi0', with_min_and_max=True)
            logger.log_tabular('LogPi1', with_min_and_max=True)
            logger.log_tabular('LossPi0', average_only=True)
            logger.log_tabular('LossPi1', average_only=True)
            logger.log_tabular('LossQ0', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    #parser.add_argument('--env', type=str, default='TwoCartPole-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    from double_cartpole_envs import DoubleCartPoleEnv

    sac(DoubleCartPoleEnv(), actor_critic=core2.MLPActorCritic,
        ac0_kwargs=dict(hidden_sizes=[args.hid]*args.l), ac1_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

    #sac(lambda : gym.make(args.env), actor_critic=core2.MLPActorCritic,
    #    ac0_kwargs=dict(hidden_sizes=[args.hid]*args.l), ac1_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
    #    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
    #    logger_kwargs=logger_kwargs)
