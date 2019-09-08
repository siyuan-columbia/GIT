import numpy as np


class DPMethod:

    @classmethod
    def get_reward(cls, current_wealth, n, is_bad_side):
        reward = np.arange(n) + 1
        reward[is_bad_side == 1] = -current_wealth
        return reward

    @classmethod
    def keep_roll(cls, current_wealth, n, is_bad_side):
        reward = cls.get_reward(current_wealth, n, is_bad_side)
        if np.mean(reward) > 0:
            return True
        else:
            return False

    @classmethod
    def calculate_expectation(cls, start_wealth, n, is_bad_side):
        result = np.zeros(n)

        if cls.keep_roll(start_wealth, n, is_bad_side):
            for i in range(n):
                if is_bad_side[i] == 1:
                    result[i] = 0
                else:
                    current_wealth = start_wealth + i + 1
                    result[i] = cls.calculate_expectation(current_wealth, n, is_bad_side)
        else:
            return start_wealth
        return np.mean(result)


class ValueIteration:

    def __init__(self, n, is_bad_side):
        self.n = n
        self.is_bad_side = is_bad_side
        self.utility = dict()
        self.upper_bound = 0

    def initialize_utility(self):
        p_bad = np.mean(is_bad_side)

        upper_bound = int(np.ceil(self.n * (1 - p_bad) / p_bad))
        for i in range(upper_bound + self.n + 1):
            self.utility[i] = 0
        self.upper_bound = upper_bound

    def utility_update(self, s, new_utility):

        if s >= self.upper_bound:
            new_utility[s] = s
        else:
            sum = 0
            for i in range(self.n):
                if self.is_bad_side[i] == 1:
                    continue
                else:
                    s_next = s + i + 1
                    sum += self.utility[s_next]
            roll = sum / self.n
            stop = s
            # print(s, roll, stop)
            new_utility[s] = max(roll, stop)

        return new_utility

    def get_expectation(self):

        self.initialize_utility()

        converge = False
        while not converge:

            new_utility = self.utility.copy()
            for s in self.utility.keys():
                new_utility = self.utility_update(s, new_utility)

            for k in self.utility.keys():
                if self.utility[k] != new_utility[k]:
                    converge = False
                    break
                else:
                    converge = True

            self.utility = new_utility
            print(new_utility)
        return self.utility[0]


def main(n, is_bad_side):
    if len(is_bad_side) != n:
        raise ValueError("bad side length {} not equal to dice sides {}!".format(len(is_bad_side), n))

    if np.all(is_bad_side == 0):
        raise ValueError("no bad side, won't stop!")

    die_n = ValueIteration(n, is_bad_side)

    expectation = die_n.get_expectation()

    return expectation


if __name__ == "__main__":
    n = 11
    is_bad_side = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])

    # expectation1 = DPMethod.calculate_expectation(0, n, is_bad_side)
    # print("expectation: {}".format(np.round(expectation1, 3)))

    expectation2 = main(n, is_bad_side)
    print("expectation: {}".format(np.round(expectation2, 3)))
