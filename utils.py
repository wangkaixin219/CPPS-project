import torch
import torch.nn as nn
from torch.autograd import Variable
import xlrd
from collections import namedtuple

SKU = namedtuple('SKU', ['code', 'suppliers', 'unit', 'holding_cost', 'disposal_cost', 'shortage_cost', 'shelf_life'])
Supplier = namedtuple('Supplier', ['sku_code', 'supplier_name', 'mode', 'min_lead_time', 'max_lead_time',
                                   'min_order', 'max_order', 'unit_cost', 'transport_cost'])


class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.done = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.done[:]


def hard_update(target: nn.Module, src: nn.Module):
    for target_params, src_params in zip(target.parameters(), src.parameters()):
        target_params.data.copy_(src_params.data)


def soft_update(target: nn.Module, src: nn.Module, tau):
    for target_params, src_params in zip(target.parameters(), src.parameters()):
        target_params.data.copy_(target_params.data * (1.0 - tau) + src_params.data * tau)


def to_numpy(var: torch.Tensor):
    return var.data.numpy()


def to_tensor(var, requires_grad=False):
    return Variable(torch.from_numpy(var), requires_grad=requires_grad)


def choose_sku(sku_code, sku_list):
    selected_sku = None
    for sku in sku_list:
        if sku_code == sku.code:
            selected_sku = sku
            break
    print(sku.holding_cost, sku.disposal_cost, sku.shortage_cost, sku.suppliers[0].unit_cost, sku.suppliers[0].transport_cost)
    return selected_sku


def load_sku_data(file):
    data = xlrd.open_workbook(file)

    # Read supplier attributions
    supply_lead_time_table = data.sheet_by_name('Supply Lead-time')
    supplier_list = []
    row_start_idx = 6
    row_end_idx = supply_lead_time_table.nrows
    for row in range(row_start_idx, row_end_idx):
        _, sku_code, _, supplier_name, min_order, mode, \
            unit_cost, transport_cost, _, min_lead_time, _, max_lead_time = supply_lead_time_table.row_values(row)
        supplier = Supplier(sku_code=sku_code, supplier_name=supplier_name, min_order=0, max_order=10*min_order,
                            mode=mode, min_lead_time=min_lead_time, max_lead_time=max_lead_time,
                            unit_cost=unit_cost, transport_cost=transport_cost)
        supplier_list.append(supplier)

    # Read sku attributions
    part_master_table = data.sheet_by_name('Part Master')
    sku_list = []
    row_start_idx = 6
    row_end_idx = part_master_table.nrows
    for row in range(row_start_idx, row_end_idx):
        _, code, _, _, _, _, holding_cost, shortage_cost, disposal_cost, unit, _ = part_master_table.row_values(row)
        sku = SKU(code=code, suppliers=[], unit=unit, shelf_life=24, holding_cost=holding_cost,
                  shortage_cost=holding_cost+1000000, disposal_cost=disposal_cost)
        sku_list.append(sku)

    for sku in sku_list:
        for supplier in supplier_list:
            if sku.code == supplier.sku_code:
                sku.suppliers.append(supplier)

    return sku_list


def train(agent, env, args):
    print('----------- Training -----------')
    running_reward = 0

    for episode in range(1, args.max_episodes + 1):

        state = env.reset()

        while True:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            agent.memory.rewards.append(reward)
            agent.memory.done.append(done)
            running_reward += reward
            if done:
                break

        if episode % args.update_per_episodes == 0:
            agent.update()
            agent.memory.clear()

        if episode % args.save_per_episodes == 0:
            torch.save(agent.policy.state_dict(), './output/PPOAgent_{}.pth'.format(episode))

        if episode % args.log_per_episodes == 0:
            running_reward /= args.log_per_episodes
            print('Episode {} \t Avg reward: {:.3e}'.format(episode, running_reward))
            running_reward = 0


def test(agent, env):
    print('\n----------- Testing -----------')
    running_reward = 0
    state = env.reset()
    while True:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        print(action, state, '{:.3e}'.format(reward))
        agent.memory.rewards.append(reward)
        agent.memory.done.append(done)
        running_reward += reward
        if done:
            break
    print('Reward: {:.3e}'.format(running_reward))

