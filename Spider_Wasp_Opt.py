import numpy as np
import tools


def SWO(ub_num, lb_num, Tmax, f_pre, datas, fn):
    dim = fn.shape[0]
    SearchAgents_no = dim * 3
    ub, lb = ub_num * np.ones_like(fn), lb_num * np.ones_like(fn)
    fobj, fhd = tools.opl(f_pre, datas, fn)
    #SearchAgents_no, Tmax, ub, lb, dim, fobj, fhd
    Best_SW = np.zeros(dim)  # A vector to include the best-so-far spider wasp(Solution)
    Best_score = np.inf  # A scalar variable to include the best-so-far score
    Convergence_curve = np.zeros(Tmax)

    TR = 0.3  # The trade-off probability between hunting and mating behaviors
    Cr = 0.2  # The crossover probability
    N_min = 20  # The minimum population size

    Positions = initialization(SearchAgents_no, dim, ub, lb)  # Initialize positions of spider wasps
    t = 0  # Function evaluation counter

    SW_Fit = np.zeros(SearchAgents_no)
    for i in range(SearchAgents_no):
        SW_Fit[i] = fhd(Positions[i, :], fobj)
        if SW_Fit[i] < Best_score:  # Update the best-so-far solution
            Best_score = SW_Fit[i]
            Best_SW = Positions[i, :]

    while t < Tmax:
        a = 2 - 2 * (t / Tmax)  # a decreases linearly from 2 to 0
        a2 = -1 + -1 * (t / Tmax)  # a2 decreases linearly from -1 to -2
        k = (1 - t / Tmax)  # k decreases linearly from 1 to 0

        JK = np.random.permutation(SearchAgents_no)  # Random permutation of the search agents' indices

        if np.random.rand() < TR:  # Hunting and nesting behavior
            # Update the position of search agents
            for i in range(SearchAgents_no):
                r1 = np.random.rand()  # Random number in [0, 1]
                r2 = np.random.rand()  # Random number in [0, 1]
                r3 = np.random.rand()  # Random number in [0, 1]
                p = np.random.rand()  # Random number in [0, 1]
                C = a * (2 * r1 - 1)  # Eq. (11)
                l = (a2 - 1) * np.random.rand() + 1  # Eq. (7)
                L = Levy(1)  # Levy flight-based number
                vc = np.random.uniform(-k, k, dim)  # Eq. (12)
                rn1 = np.random.randn()  # Normal distribution-based number

                O_P = np.copy(Positions[i, :])  # Storing the current position of the i-th solution

                for j in range(dim):
                    if i < k * SearchAgents_no:
                        if p < (1 - t / Tmax):  # Exploration
                            if r1 < r2:
                                m1 = np.abs(rn1) * r1  # Eq. (5)
                                Positions[i, j] = Positions[i, j] + m1 * (
                                            Positions[JK[0], j] - Positions[JK[1], j])  # Eq. (4)
                            else:
                                B = 1 / (1 + np.exp(l))  # Eq. (8)
                                m2 = B * np.cos(l * 2 * np.pi)  # Eq. (7)
                                Positions[i, j] = Positions[JK[i], j] + m2 * (
                                            lb[j] + np.random.rand() * (ub[j] - lb[j]))  # Eq. (6)
                        else:  # Exploration and exploitation
                            if r1 < r2:
                                Positions[i, j] = Positions[i, j] + C * np.abs(
                                    2 * np.random.rand() * Positions[JK[2], j] - Positions[i, j])  # Eq. (10)
                            else:
                                Positions[i, j] = Positions[i, j] * vc[j]  # Eq. (12)
                    else:
                        if r1 < r2:
                            Positions[i, j] = Best_SW[j] + np.cos(2 * l * np.pi) * (
                                        Best_SW[j] - Positions[i, j])  # Eq. (16)
                        else:
                            Positions[i, j] = Positions[JK[0], j] + r3 * np.abs(L) * (
                                        Positions[JK[0], j] - Positions[i, j]) + (1 - r3) * (
                                                          np.random.rand() > np.random.rand()) * (
                                                          Positions[JK[2], j] - Positions[JK[1], j])  # Eq. (17)

                # Ensure positions are within bounds
                Positions[i, :] = np.clip(Positions[i, :], lb, ub)

                SW_Fit1 = fhd(Positions[i, :], fobj)  # Fitness of the new solution
                if SW_Fit1 < SW_Fit[i]:
                    SW_Fit[i] = SW_Fit1  # Update local best fitness
                    if SW_Fit[i] < Best_score:  # Update global best solution
                        Best_score = SW_Fit[i]
                        Best_SW = Positions[i, :]
                else:
                    Positions[i, :] = O_P  # Restore last best solution

                t += 1
                if t >= Tmax:
                    break
                Convergence_curve[t] = Best_score

        else:  # Mating behavior
            # Update position of search agents
            for i in range(SearchAgents_no):
                l = (a2 - 1) * np.random.rand() + 1  # Eq. (7)
                SW_m = np.zeros(dim)  # Spider wasp male
                O_P = np.copy(Positions[i, :])  # Current position of the i-th solution

                # Update male spider position
                if SW_Fit[JK[0]] < SW_Fit[i]:  # Eq. (23)
                    v1 = Positions[JK[0], :] - Positions[i, :]
                else:
                    v1 = Positions[i, :] - Positions[JK[0], :]

                if SW_Fit[JK[1]] < SW_Fit[JK[2]]:  # Eq. (24)
                    v2 = Positions[JK[1], :] - Positions[JK[2], :]
                else:
                    v2 = Positions[JK[2], :] - Positions[JK[1], :]

                rn1 = np.random.randn()  # Normal distribution-based number
                rn2 = np.random.randn()  # Normal distribution-based number

                for j in range(dim):
                    SW_m[j] = Positions[i, j] + np.exp(l) * np.abs(rn1) * v1[j] + (1 - np.exp(l)) * np.abs(rn2) * v2[
                        j]  # Eq. (22)
                    if np.random.rand() < Cr:  # Eq. (21)
                        Positions[i, j] = SW_m[j]

                # Ensure positions are within bounds
                Positions[i, :] = np.clip(Positions[i, :], lb, ub)

                SW_Fit1 = fhd(Positions[i, :], fobj)  # Fitness of new solution
                if SW_Fit1 < SW_Fit[i]:
                    SW_Fit[i] = SW_Fit1  # Update local best fitness
                    if SW_Fit[i] < Best_score:
                        Best_score = SW_Fit[i]  # Update global best solution
                        Best_SW = Positions[i, :]
                else:
                    Positions[i, :] = O_P  # Restore last best solution

                t += 1
                if t >= Tmax:
                    break
                Convergence_curve[t] = Best_score

        # Population reduction
        SearchAgents_no = int(N_min + (SearchAgents_no - N_min) * ((Tmax - t) / Tmax))  # Eq. (25)

    Convergence_curve[t - 1] = Best_score
    return Best_score, Best_SW

def Levy(d):
    beta = 3 / 2
    sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    L = 0.05 * step
    return L

def initialization(SearchAgents_no, dim, ub, lb):
    Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    return Positions
