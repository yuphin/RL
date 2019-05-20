import numpy as np

nodes = []
learning_rate = discount_factor = None
T_dt = None
Q_dt = None
Q_prob = None
actions_prob = []
states_prob = {}
transitions_count = None


class ActionTable:
    def __init__(self, reward, node_id):
        self.node_id = reward
        self.transitions = {}
        self.reward = node_id


# Parse input file to fill the global variables
def parseInput():
    global learning_rate, discount_factor, T_dt, transitions_count, Q_dt
    global Q_prob, actions_prob, states_prob
    prob_node_count = 0
    with open('the3.inp') as f:
        l_nodes = f.readline()[:-1]
        for idx, n in enumerate(l_nodes):
            if n == 'V' or n == 'O':
                nodes.append(idx)
            if n == 'S' or n == 'O':
                prob_node_count += 1

        learning_rate, discount_factor = map(float, f.readline()[:-1].split())
        num_transitions = int(f.readline()[:-1])
        transitions_inp = []
        max_n = -1
        for i in range(num_transitions):
            line = f.readline()[:-1]
            t = list(map(int, line.split()))
            max_n = max(max_n, t[0], t[1])
            transitions_inp.append(t)
        T_dt = np.full((max_n + 1, max_n + 1), -np.inf)

        transitions_count = np.zeros((max_n + 1, max_n + 1))
        Q_dt = np.copy(T_dt)
        for t in transitions_inp:
            T_dt[t[0]][t[1]] = t[2]
            Q_dt[t[0]][t[1]] = 0
        for n in nodes:
            Q_dt[n] = 0

        num_actions = int(f.readline()[:-1])
        for i in range(prob_node_count):
            line = f.readline()[:-1]
            t = list(map(int, line.split()))
            states_prob[t[0]] = t[1:]

        for i in range(num_actions):
            action_num = int(f.readline()[:-1].split(':')[1].strip())
            tables = []
            while (True):
                line = f.readline()[:-1]
                if line == '#':
                    break
                reward = int(line)
                node_id = int(f.readline()[:-1])
                table = ActionTable(reward, node_id)
                while (True):
                    line = f.readline()[:-1]
                    if line == '$':
                        break
                    transition = list(map(int, line.split()))
                    table.transitions[transition[0]] = transition[1]
                tables.append(table)
            actions_prob.append(tables)


# Select an appropiate action using Epsilon-greed method
def get_action(Q, s, p=True):
    if p:
        a = np.array([0, 1])
        choice = np.random.choice(a, 1, p=[0.4, 0.6])
        if choice:
            ind = np.argmax(Q[s])

            return ind
        else:
            ft = Q[s]
            val = np.random.choice(ft[np.isfinite(ft)])
            ind = np.where(ft == val)
            return np.random.choice(ind[0])
    else:
        ind = np.argmax(Q[s])
        return ind


# Returns Q values to be printed
def q_learn(episodes):
    for end_node in episodes:
        rate = learning_rate
        s = 0
        admission_list = nodes + [end_node]
        while (True):
            if s in admission_list:
                break
            a = get_action(Q_dt, s)
            new_action = get_action(Q_dt, a, False)
            delta = T_dt[s][a] + discount_factor * Q_dt[a][new_action] - Q_dt[s][a]
            Q_dt[s][a] += rate * delta
            s = a
    return Q_dt


if __name__ == "__main__":
    inp = [4, 5, 6]
    episodes = []
    for i in range(5000):
        episodes.append(np.random.choice(inp))
    parseInput()
    q_learn(episodes)
    print(learning_rate)
