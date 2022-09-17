from random import choice


class State:
    def __init__(self, state, coords, reward, problem):
        self.coords = coords
        self.reward = reward
        self.is_terminal = problem.is_goal_state(state)
        self.value = 0
        self.actions = list(problem.get_actions(state))
        self.next_states = {}

        for action in self.actions:
            self.next_states[action] = problem.get_next_states_and_probs(state, action)

        self.policy = None if self.is_terminal else choice(self.actions)


def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    possible_states, non_terminal_states = initialize_states(problem)

    while True:
        max_change = 0
        new_values = []

        # sweep through all the non-terminal states
        for state in non_terminal_states:
            best_value, best_action = find_best_value_and_action(state, possible_states, discount_factor)

            new_values.append(best_value)
            state.policy = best_action

            if max_change < abs(best_value - state.value):
                max_change = abs(best_value - state.value)

        update_values(new_values, non_terminal_states)

        # check for stopping condition
        if max_change < epsilon:
            return extract_policy(problem.get_all_states(), possible_states)


def find_policy_via_policy_iteration(problem, discount_factor):
    possible_states, non_terminal_states = initialize_states(problem)

    while True:
        updates = 0
        new_values = []

        # sweep through all the non-terminal states
        for state in non_terminal_states:
            best_value, best_action = find_best_value_and_action(state, possible_states, discount_factor)

            new_values.append(best_value)

            if best_action != state.policy:
                state.policy = best_action
                updates += 1

        update_values(new_values, non_terminal_states)

        # check for stopping condition
        if updates == 0:
            return extract_policy(problem.get_all_states(), possible_states)


def initialize_states(problem):
    possible_states = {}
    non_terminal_states = []

    for found_state in problem.get_all_states():
        coords = (found_state.x, found_state.y)
        state = State(found_state, coords, found_state.reward, problem)

        possible_states[coords] = state
        if not state.is_terminal:
            non_terminal_states.append(state)

    return possible_states, non_terminal_states


def find_best_value_and_action(state, possible_states, discount_factor):
    best_value = -float("inf")
    best_action = state.policy

    for action in state.actions:
        value = 0
        reachable_states = state.next_states[action]

        # computing value of the current action
        for reachable_state in reachable_states:
            value += discount_factor * reachable_state[1] * possible_states[reachable_state[0]].value + \
                     reachable_state[1] * possible_states[reachable_state[0]].reward

        # finding if current action is the best one
        if value > best_value:
            best_value = value
            best_action = action

    return best_value, best_action


def update_values(values, states):
    for value, state in zip(values, states):
        state.value = value


def extract_policy(all_states, possible_states):
    found_policy = {}

    for found_state in all_states:
        coords = tuple(found_state[:-1])
        found_policy[coords] = possible_states[coords].policy

    return found_policy
