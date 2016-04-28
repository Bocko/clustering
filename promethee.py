#!/usr/bin/env python
from math import exp

class PreferenceType2:
    q = 0
    def __init__(self, valQ):
        self.q = valQ
    def value(self, diff):
        if (diff <= self.q):
            return 0
        return 1

class PreferenceType5:
    q = 0
    p = 1
    def __init__(self, valQ, valP):
        self.q = valQ
        self.p = valP
    def value(self, diff):
        if (diff <= self.q):
            return 0
        if (diff <= self.p):
            return (diff - self.q) / (self.p - self.q)
        return 1

class PreferenceType6:
    s = 0.5
    valSquare = 0.5
    def __init__(self, valS):
        self.s = valS
        self.valSquare = -1 * (2 * valS * valS)
    def value(self, diff):
        if (diff <= 0):
            return 0
        return 1 - exp(diff * diff / self.valSquare)


def rankingP2(alternatives, criteria, weights, funcPrefCrit):
    netflows = []
    if (len(alternatives) == 1):
            return alternatives[0]

    for alt1 in alternatives:
        outRankingPlus = 0
        outRankingMoins = 0
        for alt2 in alternatives:
            if alt1 == alt2:
                continue
            PiXA = 0
            PiAX = 0
            for k in range(len(criteria)):
                weight = weights[k]
                funcPref = funcPrefCrit[k]
                valAlt1 = alt1[k]
                valAlt2 = alt2[k]
                val1 = valAlt1 - valAlt2
                val2 = valAlt2 - valAlt1
                PiXA = PiXA + weight * funcPref.value(val1)
                PiAX = PiAX + weight * funcPref.value(val2)
            outRankingPlus = outRankingPlus + PiXA
            outRankingMoins = outRankingMoins + PiAX
        outRankingPlus = outRankingPlus / (len(alternatives) - 1)
        outRankingMoins = outRankingMoins / (len(alternatives) - 1)
        outRanking = outRankingPlus - outRankingMoins
        netflows.append(outRanking)

    return netflows

def uniflows_eval(alternatives, criteria, funcPrefCrit):
    uniposflows = []
    uninegflows = []
    for alt1 in alternatives:
        uniposflows.append([])
        uninegflows.append([])
        for k in range(len(criteria)):
            uniposval = 0
            uninegval = 0
            for alt2 in alternatives:
                if alt1 == alt2:
                    continue

                funcPref = funcPrefCrit[k]
                valAlt1 = alt1[k]
                valAlt2 = alt2[k]
                val1 = valAlt1 - valAlt2
                val2 = valAlt2 - valAlt1
                uniposval += funcPref.value(val1)
                uninegval += funcPref.value(val2)

            uniposflows[-1].append(uniposval/(len(alternatives)-1))
            uninegflows[-1].append(uninegval/(len(alternatives)-1))

    return uniposflows, uninegflows

def uninetflows_eval(alternatives, criteria, weights, funcPrefCrit):
    uniposflows, uninegflows = uniflows_eval(alternatives, criteria, funcPrefCrit)
    uninetflows = [[0 for col in range(len(uniposflows[0]))] for row in range(len(uniposflows))]
    for i in range(len(uninetflows)):
        for j in range(len(uninetflows[0])):
            uninetflows[i][j] += uniposflows[i][j] - uninegflows[i][j]
    return uninetflows

def walking_weights_eval(uninetflows, weights):
    crits = len(weights)
    alts = len(uninetflows)
    netflows = [sum([uninetflows[i][k]*weights[k] for k in range(crits)]) for i in range(alts)]
    ranking = sorted(range(alts), key=lambda k: netflows[k], reverse=True)
    # delta = netflows[a] - netflows[b]
    # delta_i = uninetflows[a][4] - uninetflows[b][4]
    # print("Test")
    # print(delta,delta_i)
    # print("calc",delta*delta_i < 0)
    # print("calc",delta*delta_i > delta**2)
    # print(delta*delta_i/(delta*delta_i-delta**2))
    omega_zero = [[] for i in range(crits)]
    omega_minus = [[] for i in range(crits)]
    omega_plus = [[] for i in range(crits)]
    alphas_minus = [[] for i in range(crits)]
    alphas_plus = [[] for i in range(crits)]
    walking_weights = []
    for k in range(crits):
        for i in ranking[:1]:
            for j in ranking:
                if i != j:
                    delta = netflows[i] - netflows[j]
                    delta_i = uninetflows[i][k] - uninetflows[j][k]
                    if delta*delta_i < 0 and (i,j) not in omega_minus[k]:
                        omega_minus[k].append((i,j))
                        alphas_minus[k].append(delta*delta_i/(delta*delta_i-delta**2))
                    elif delta*delta_i > delta**2 and (i,j) not in omega_plus[k]:
                        omega_plus[k].append((i,j))
                        alphas_plus[k].append(delta*delta_i/(delta*delta_i-delta**2))
                    elif delta == 0 and delta_i != 0 and (i,j) not in omega_zero[k]:
                        # TODO: non-empty case to develop
                        omega_zero[k].append((i,j))

        if omega_zero[k] != []:
            print("warning")

        if alphas_plus[k] == []:
            w_minus = 0
        else:
            alpha_plus = min(alphas_plus[k])
            beta_minus = (1-weights[k])/weights[k]*(1-alpha_plus)
            w_minus = (1+beta_minus)*weights[k]
            if w_minus < 0:
                w_minus = 0

        if alphas_minus[k] == []:
            w_plus = 1
        else:
            alpha_minus = max(alphas_minus[k])
            beta_plus = (1-weights[k])/weights[k]*(1-alpha_minus)
            w_plus = (1+beta_plus)*weights[k]
            if w_plus > 1:
                w_plus = 1

        walking_weights.append((w_minus,w_plus))

    return walking_weights

def weights_update(init_weights, new_weight, crit):
    crits = len(init_weights)
    beta = (new_weight-init_weights[crit])/init_weights[crit]
    alpha = (1-(1+beta)*init_weights[crit])/(1-init_weights[crit])
    new_weights = []
    for k in range(crits):
        if k == crit:
            new_weights.append(new_weight)
        else:
            new_weights.append(alpha*init_weights[k])

    return new_weights

def si_weights_update(walking_weights, init_weights, alternatives, criteria, funcPrefCrit):
    EPS=1e-5
    si_weights = [[[],[]] for i in range(len(criteria))]
    si_rankings = [[[],[]] for i in range(len(criteria))]
    si_firsts =  [[[],[]] for i in range(len(criteria))]
    si_diffs =  [[[],[]] for i in range(len(criteria))]
    for k in range(len(criteria)):
        if walking_weights[k][0] != 0:
            si_weights[k][0] = weights_update(init_weights,walking_weights[k][0]-EPS,k)
            si_ranking = rankingP2(alternatives,criteria,si_weights[k][0],funcPrefCrit)
            ind = sorted(range(len(si_ranking)), key=lambda k: si_ranking[k], reverse=True)
            si_rankings[k][0] = ind
            si_firsts[k][0] = ind[0]
            si_diffs[k][0] = 2*abs(init_weights[k] - walking_weights[k][0])
            # print('--')
            # for i in ind:
            #     print(names[i])
        if walking_weights[k][1] != 1:
            si_weights[k][1] = weights_update(init_weights,walking_weights[k][1]+EPS,k)
            si_ranking = rankingP2(alternatives,criteria,si_weights[k][1],funcPrefCrit)
            ind = sorted(range(len(si_ranking)), key=lambda k: si_ranking[k], reverse=True)
            si_rankings[k][1] = ind
            si_firsts[k][1] = ind[0]
            si_diffs[k][1] = 2*abs(init_weights[k] - walking_weights[k][1])
            # print('--')
            # for i in ind:
            #     print(names[i])

    return si_weights, si_rankings, si_firsts, si_diffs

def paretoFilter(alternatives, criteria):
    altFilt = []
    for alt1 in alternatives:
        paretoFront = True
        for alt2 in alternatives:
            if alt1 == alt2:
                continue
            if paretoInf(alt1, alt2, criteria):
                paretoFront = False
                break
        if (paretoFront):
            altFilt.append(alt1)

    ind = []
    for alt in altFilt:
        ind.append(alternatives.index(alt))
    return altFilt, ind

def paretoInf(alt1, alt2, criteria):
    equals = 0
    for i in range(len(criteria)):
        valAlt1 = alt1[i]
        valAlt2 = alt2[i]
        if (valAlt1 > valAlt2):
            return False
        if (valAlt1 == valAlt2):
            equals = equals + 1
    return equals < len(criteria)

if __name__ == '__main__':

    criteria = ['Stability', 'Healthcare', 'Culture and Environment', 'Education', 'Infrastructure', 'Spatial Characteristics'] #noms des criteres utilises

    # poids = {'Infrastructure': 0.15, 'Spatial': 0.25, 'Education': 0.075, 'CultureEnv': 0.1875, 'Healthcare': 0.15, 'Stability': 0.1875} #poids des criteres utilises

    weights = [0.1875, 0.15, 0.1875, 0.075, 0.15, 0.25]

    # funcPrefCrit = {'Infrastructure': PreferenceType2(0), 'Spatial': PreferenceType2(0), 'Education': PreferenceType2(0), 'CultureEnv': PreferenceType2(0), 'Healthcare': PreferenceType2(0), 'Stability': PreferenceType2(0)} #fonction de preference utilisee (voir article pour typologie)
    funcPrefCrit = [PreferenceType2(0), PreferenceType2(0), PreferenceType2(0),  PreferenceType2(0), PreferenceType2(0), PreferenceType2(0)]

    # names = ['Hong Kong', 'Stockholm', 'Rome', 'New York', 'Atlanta', 'Buenos Aires', 'Santiago', 'Sao Paulo', 'Mexico City', 'New Delhi', 'Istanbul', 'Jakarta', 'Tehran', 'Dakar']
    #
    # alternatives = [[96.4, 75.0, 100.0, 85.9, 87.5, 95.0],
    # [96.4, 58.9, 100.0, 91.2, 95.8, 95.0],
    # [92.9, 67.3, 100.0, 91.7, 87.5, 80.0],
    # [89.3, 65.2, 100.0, 91.7, 91.7, 70.0],
    # [92.9, 42.9, 100.0, 91.7, 91.7, 85.0],
    # [85.7, 42.3, 100.0, 85.9, 87.5, 70.0],
    # [85.7, 35.1, 83.3, 89.1, 70.8, 75.0],
    # [66.1, 52.4, 66.7, 80.3, 70.8, 60.0],
    # [46.4, 65.8, 75.0, 82.4, 66.7, 45.0],
    # [58.9, 58.6, 75.0, 55.6, 58.3, 55.0],
    # [67.9, 47.5, 58.3, 68.8, 50.0, 55.0],
    # [57.1, 42.3, 66.7, 59.3, 45.8, 50.0],
    # [33.9, 53.6, 50.0, 35.9, 62.5, 50.0],
    # [37.5, 22.6, 50.0, 59.7, 41.7, 50.0]]

    # names = ["Hong Kong", "Amsterdam", "Osaka", "Paris", "Sydney", "Stockholm", "Berlin", "Toronto", "Munich", "Tokyo", "Rome", "London", "Madrid", "Washington DC", "Chicago", "New York", "Los Angeles", "San Francisco", "Boston", "Seoul", "Atlanta", "Singapore", "Miami", "Budapest", "Lisbon", "Buenos Aires", "Moscow", "St Petersburg", "Athens", "Beijing", "Santiago", "Warsaw", "Shanghai", "Shenzhen", "Lima", "Sao Paulo", "Kuala Lumpur", "Tianjin", "Guangzhou", "Johannesburg", "Mexico City", "Rio de Janeiro", "Bucharest", "Kiev", "Belgrade", "New Delhi", "Dalian", "Manila", "Bangkok", "Bogota", "Istanbul", "Mumbai", "Casablanca", "Caracas", "Cairo", "Jakarta", "Hanoi", "Tashkent", "Damascus", "Ho Chi Minh City", "Tehran", "Nairobi", "Lusaka", "Phnom Penh", "Karachi", "Dakar", "Abidjan", "Dhaka", "Lagos", "Harare"]
    #
    # alternatives = [[95, 87.5, 85.9, 100, 96.4, 75],
    # [80, 100, 97.2, 91.7, 96.4, 71.3],
    # [90, 100, 93.5, 100, 96.4, 64],
    # [85, 100, 97.2, 100, 96.4, 63.7],
    # [90, 100, 94.4, 100, 100, 55.7],
    # [95, 95.8, 91.2, 100, 96.4, 58.9],
    # [85, 100, 97.2, 91.7, 96.4, 61.7],
    # [100, 100, 97.2, 100, 89.3, 50],
    # [85, 100, 97.2, 91.7, 89.3, 62.5],
    # [90, 100, 94.4, 100, 92.9, 53.3],
    # [80, 87.5, 91.7, 100, 92.9, 67.3],
    # [70, 87.5, 97.2, 100, 89.3, 72.6],
    # [85, 87.5, 94.4, 100, 92.9, 61.3],
    # [80, 91.7, 94.4, 100, 96.4, 55.1],
    # [85, 91.7, 91.7, 100, 92.9, 52.7],
    # [70, 91.7, 91.7, 100, 89.3, 65.2],
    # [80, 91.7, 94.4, 100, 89.3, 50.3],
    # [85, 91.7, 94.4, 83.3, 85.7, 53],
    # [80, 91.7, 91.7, 100, 96.4, 46.7],
    # [80, 83.3, 85.6, 100, 89.3, 58.8],
    # [85, 91.7, 91.7, 100, 92.9, 42.9],
    # [95, 87.5, 76.6, 83.3, 100, 46.7],
    # [85, 91.7, 91.7, 100, 92.9, 39.3],
    # [85, 91.7, 90, 100, 83.9, 43],
    # [80, 87.5, 95.1, 91.7, 80.4, 41.7],
    # [70, 87.5, 85.9, 100, 85.7, 42.3],
    # [65, 79.2, 81.5, 91.7, 83.9, 54.2],
    # [65, 87.5, 81.5, 83.3, 80.4, 48.2],
    # [75, 83.3, 83.1, 75, 75, 47.3],
    # [80, 66.7, 72.2, 83.3, 85.7, 51.5],
    # [75, 70.8, 89.1, 83.3, 85.7, 35.1],
    # [80, 70.8, 80.3, 75, 82.1, 39],
    # [80, 62.5, 75, 75, 75, 46.1],
    # [85, 62.5, 63.7, 66.7, 82.1, 48.5],
    # [60, 66.7, 81.7, 91.7, 75, 47.3],
    # [60, 70.8, 80.3, 66.7, 66.1, 52.4],
    # [80, 62.5, 67.8, 91.7, 76.8, 36.6],
    # [90, 66.7, 65.3, 66.7, 82.1, 27.7],
    # [80, 62.5, 61.1, 66.7, 76.8, 42.9],
    # [50, 58.3, 90.5, 83.3, 69.6, 44.9],
    # [45, 66.7, 82.4, 75, 46.4, 65.8],
    # [55, 66.7, 77.5, 83.3, 71.4, 44],
    # [80, 66.7, 74.3, 66.7, 66.1, 34.7],
    # [70, 75, 73.4, 83.3, 50, 33.3],
    # [60, 75, 73.1, 75, 57.1, 36],
    # [55, 58.3, 55.6, 75, 58.9, 58.6],
    # [85, 62.5, 62, 66.7, 75, 21],
    # [60, 58.3, 63.2, 66.7, 64.3, 46.1],
    # [50, 62.5, 64.4, 100, 69.6, 36.3],
    # [35, 62.5, 75.2, 66.7, 64.3, 50.6],
    # [55, 50, 68.8, 58.3, 67.9, 47.5],
    # [60, 54.2, 56.3, 66.7, 51.8, 52.1],
    # [65, 45.8, 60.9, 58.3, 60.7, 43.8],
    # [30, 41.7, 76.6, 75, 60.7, 52.1],
    # [55, 45.8, 54.9, 58.3, 53.6, 48.2],
    # [50, 45.8, 59.3, 66.7, 57.1, 42.3],
    # [55, 54.2, 53.7, 58.3, 51.8, 38.4],
    # [50, 58.3, 55.3, 75, 51.8, 26.8],
    # [55, 50, 54.2, 41.7, 55.4, 36.5],
    # [55, 50, 49.5, 66.7, 48.2, 35.1],
    # [50, 62.5, 35.9, 50, 33.9, 53.6],
    # [40, 45.8, 69.9, 66.7, 42.9, 33.9],
    # [60, 33.3, 59.7, 41.7, 55.4, 23.2],
    # [60, 37.5, 49.3, 58.3, 53.6, 24.1],
    # [20, 45.8, 38.7, 66.7, 51.8, 48.5],
    # [50, 41.7, 59.7, 50, 37.5, 22.6],
    # [25, 45.8, 54.2, 50, 53.6, 30.1],
    # [50, 29.2, 43.3, 41.7, 26.8, 35.7],
    # [25, 33.3, 52.3, 33.3, 48.2, 22.3],
    # [30, 20.8, 53, 66.7, 35.7, 17.3]]

    alternatives = [[55, 54.2, 53.7, 58.3, 51.8, 38.4], [25, 33.3, 52.3, 33.3, 48.2, 22.3], [85, 91.7, 91.7, 100, 92.9, 52.7], [95, 95.8, 91.2, 100, 96.4, 58.9], [70, 87.5, 97.2, 100, 89.3, 72.6], [75, 70.8, 89.1, 83.3, 85.7, 35.1], [85, 100, 97.2, 91.7, 89.3, 62.5], [50, 62.5, 35.9, 50, 33.9, 53.6], [80, 87.5, 91.7, 100, 92.9, 67.3], [80, 91.7, 91.7, 100, 96.4, 46.7], [70, 91.7, 91.7, 100, 89.3, 65.2], [90, 100, 94.4, 100, 92.9, 53.3], [65, 45.8, 60.9, 58.3, 60.7, 43.8], [70, 75, 73.4, 83.3, 50, 33.3], [25, 45.8, 54.2, 50, 53.6, 30.1]]
    names = ['Hanoi', 'Lagos', 'Chicago', 'Stockholm', 'London', 'Santiago', 'Munich', 'Tehran', 'Rome', 'Boston', 'New York', 'Tokyo', 'Casablanca', 'Kiev', 'Abidjan']

    pareto_alt, pareto_ind = paretoFilter(alternatives,criteria)

    # candidatsFiltres = filtragePareto(candidats,criteres) #permet de supprimer les candidats ne se trouvant pas sur le front de Pareto
    #print('filtrage : ', candidatsFiltres)
    #print('decision : ', decision(candidats,criteres,poids, funcPrefCrit))
    # netflows, uninetflows = decision(candidats,criteres,poids, funcPrefCrit)

    ranking = rankingP2(alternatives,criteria,weights,funcPrefCrit)
    print(ranking)
    ind = sorted(range(len(ranking)), key=lambda k: ranking[k], reverse=True)
    for i in ind:
        print(names[i])

    uninetflows = uninetflows_eval(alternatives,criteria,weights,funcPrefCrit)
    walking_weights = walking_weights_eval(uninetflows,weights)
    print("Walking weights")
    for w in walking_weights:
        print(w)

    si_weights, si_rankings, si_firsts, si_diffs = si_weights_update(walking_weights, weights,alternatives,criteria,funcPrefCrit)

    print(si_rankings)
    print(si_firsts)
    print(si_diffs[0][0])
