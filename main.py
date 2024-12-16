import os
import yaml
import pickle
import argparse
import subprocess
import numpy as np

import gym
import torch
from torch.optim import Adam

import lop.envs
from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.agent import Agent
from lop.algos.rl.ppo import PPO
from lop.utils.miscellaneous import compute_matrix_rank_summaries

def save_data(cfg, rets, termination_steps,
              pol_features_activity, stable_rank, mu, pol_weights, val_weights,
              action_probs=None, weight_change=[], friction=-1.0, num_updates=0, previous_change_time=0):
    data_dict = {
        'rets': np.array(rets),
        'termination_steps': np.array(termination_steps),
        'pol_features_activity': pol_features_activity,
        'stable_rank': stable_rank,
        'action_output': mu,
        'pol_weights': pol_weights,
        'val_weights': val_weights,
        'action_probs': action_probs,
        'weight_change': torch.tensor(weight_change).numpy(),
        'friction': friction,
        'num_updates': num_updates,
        'previous_change_time': previous_change_time
    }
    with open(cfg['log_path'], 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data(cfg):
    with open(cfg['log_path'], 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def save_checkpoint(cfg, step, learner):
    # Save step, model and optimizer states
    ckpt_dict = dict(
        step = step,
        actor = learner.pol.state_dict(),
        critic = learner.vf.state_dict(),
        opt = learner.opt.state_dict()
    )
    torch.save(ckpt_dict, cfg['ckpt_path'])
    print(f'Save checkpoint at step={step}')


def load_checkpoint(cfg, device, learner):
    # Load step, model and optimizer states
    step = 0
    ckpt_dict = torch.load(cfg['ckpt_path'], map_location=device)
    step = ckpt_dict['step']
    learner.pol.load_state_dict(ckpt_dict['actor'])
    learner.vf.load_state_dict(ckpt_dict['critic'])
    learner.opt.load_state_dict(ckpt_dict['opt'])
    print(f"Successfully restore from checkpoint: {cfg['ckpt_path']}.")
    return step, learner

def main():

    #终端传入参数操作
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, default='./cbp.yml')
    parser.add_argument('-s', '--seed', required=False, type=int, default="1")
    parser.add_argument('-d', '--device', required=False, default='cpu')
    args = parser.parse_args() #解析从命令行传入的参数，并将解析后的结果存入 args 对象中。


    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'


    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    cfg['log_path'] = cfg['dir'] + "random_seed_" +str(args.seed) + '.log'
    cfg['ckpt_path'] = cfg['dir'] + "random_seed_" + str(args.seed) + '.pth'
    cfg['done_path'] = cfg['dir'] + "random_seed_" + str(args.seed) + '.done'

    os.makedirs(cfg['dir'], exist_ok=True) #如果文件不存在则创建文件夹

    # Set default values
    cfg.setdefault('wd', 0) #如果不存在key 则创建，如果存在则不变
    cfg.setdefault('init', 'lecun')
    cfg.setdefault('to_log', [])
    cfg.setdefault('beta_1', 0.9)
    cfg.setdefault('beta_2', 0.999)
    cfg.setdefault('eps', 1e-8)
    cfg.setdefault('no_clipping', False)
    cfg.setdefault('loss_type', 'ppo')
    cfg.setdefault('frictions_file', 'cfg/frictions')
    cfg.setdefault('max_grad_norm', 1e9)
    cfg.setdefault('perturb_scale', 0)
    cfg['n_steps'] = int(float(cfg['n_steps']))
    cfg['perturb_scale'] = float(cfg['perturb_scale'])
    n_steps = cfg['n_steps']    


    # Set default values for CBP
    cfg.setdefault('mt', 10000)
    cfg.setdefault('rr', 0)
    cfg['rr'] = float(cfg['rr'])
    cfg.setdefault('decay_rate', 0.99)
    cfg.setdefault('redo', False)
    cfg.setdefault('threshold', 0.03)
    cfg.setdefault('reset_period', 1000)
    cfg.setdefault('util_type_val', 'contribution')
    cfg.setdefault('util_type_pol', 'contribution')
    cfg.setdefault('pgnt', (cfg['rr']>0) or cfg['redo'])
    cfg.setdefault('vgnt', (cfg['rr']>0) or cfg['redo'])

    # Initialize env
    seed = cfg['seed']
    friction = -1.0
    if cfg['env_name'] in ['SlipperyAnt-v2', 'SlipperyAnt-v3']:
        xml_file = os.path.abspath(cfg['dir'] + f'slippery_ant_{seed}.xml')
        cfg.setdefault('friction', [0.02, 2])
        cfg.setdefault('change_time', int(2e6))

        with open(cfg['frictions_file'], 'rb+') as f:
            frictions = pickle.load(f)
        friction_number = 0
        new_friction = frictions[seed][friction_number]

        if friction < 0: # If no saved friction, use the default value 1.0
            friction = 1.0
        env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
        print(f'Initial friction: {friction:.6f}')
    else:
        env = gym.make(cfg['env_name'])
    env.name = None
    print(cfg)
main()
print("end")