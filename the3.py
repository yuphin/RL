import numpy as np
#!!! NOTE: This script is only compatible with Python 2.7. Please run as 'python the3.py', not python3
nodes = []
learning_rate = discount_factor = None
T_dt = None
Q_dt = None
Q_prob = None
goal_nodes = []
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
            if n == 'G':
                goal_nodes.append(idx)

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
                if line.startswith('#'):
                    break
                reward = int(line)
                node_id = int(f.readline()[:-1])
                table = ActionTable(reward, node_id)
                while (True):
                    line = f.readline()[:-1]
                    if line.startswith('$'):
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
# Prints the Q values
def print_Q(Q):
    m_str = ''
    for i in range(len(Q)):
        if i in nodes:
            continue
        for j in range(len(Q[i])):

            if Q[i][j] == -np.inf:
                if j == len(Q[i]) - 1:
                    m_str += '_'
                else:
                    m_str += '_ '
            else:
                if j == len(Q[i]) -1:
                    m_str += str(Q[i][j])
                else:
                    m_str += str(Q[i][j]) + ' '
        m_str += '\n'
    print m_str
#For Q-Learning algorithm, not needed for this homework
def print_p_v(Q):
    str_v = ''
    str_p = ''
    policy = []
    for idx,row in enumerate(Q):
        max_v = max(row)
        max_va = np.where(row == max_v)
        str_v += str(idx) + ' ' + str(max_v) +'\n'
        str_p += str(idx) + ' ' + ','.join(max_va[0].astype(str)) + '\n'
    print(str_v)
    print(str_p)

# Returns Q values to be printed
def q_learn():
    print('Please enter episodes, $ to cancel')
    episode_tables = []
    while True:
        text = raw_input()
        if text == '$':
            break
        else:
            episodes = list(map(int, text.split()))
            s = episodes[0]
            for idx,episode in enumerate(episodes[1:]):
                rate = learning_rate
                a = episode
                if episode in nodes:
                    nxt = 0
                else:
                    nxt = Q_dt[a][episodes[idx+2]]
                delta = T_dt[s][a] + discount_factor * nxt - Q_dt[s][a]
                Q_dt[s][a] += rate * delta
                s = a
                if episode in nodes:
                    break
            episode_tables.append(np.copy(Q_dt))
            print_Q(Q_dt)

def print_v_p(V,P):
    for k,v in V.items():
        print k,v
    print
    for k,v in P.items():
        str_buf = ''
        if isinstance(v,list):
            str_buf += str(k) + ' ' + ','.join(map(str, v))
        else:
            str_buf += str(k) + ' ' + str(v)
        print str_buf
    print

def value_iteration():
    V = {}
    P = {}
    for k, v in states_prob.items():
        V[k] = 0
        P[k] = []
    delta = 0.01
    print('Starting value iteration...')

    while(True):

        prev = {}
        for k, v in V.items():
            prev[k] = v
        for state, actions in states_prob.items():
            for action_idx in actions:
                for at in actions_prob[action_idx]:
                    if at.node_id == state:
                        e = []
                        actions = []
                        flag = -1
                        for k, v in at.transitions.items():
                            if not k in goal_nodes:
                                e.append(at.reward + discount_factor * float(float(v) / 100) * V[k])
                                actions.append(k)
                            else:
                                flag =k
                                actions.append(k)
                        if not e:
                            e.append(at.reward)
                        V[state] = max(e)
                        max_ar = np.where(np.array(e) == V[state])
                        opt_actions = list(np.take(np.array(actions),max_ar[0]))
                        P[state] = opt_actions


                    else:
                        continue
        print_v_p(V,P)
        print('To continue, enter c, to terminate enter $')
        text = raw_input()
        if text == '$':
            break
        #For the convergence case, not needed for this homework
        '''
        df = 0
        for k, v in V.items():
            df += abs(V[k] - prev[k])

        if df < delta:
           break
        '''
    return V


if __name__ == "__main__":
    parseInput()
    q_learn()
    value_iteration()






