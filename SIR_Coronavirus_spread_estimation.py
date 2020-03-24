import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

#Simulation based on SIR model differential equation model
#S = Sane individuals never been infected. 
#I= Infected individuals can't be infected twice. 
#R=Recovered Recovered person 
#Setting initial conditions for France. 65 million people is a bit big so it takes time. 
S_0 = 17000
I_0 = 2
R_0 = 0

first_infection = {'France':'1/24/20'}

class Spread_model(object):
    def __init__(self, nations, function):
        self.nations = nations
        self.function = function

    def infected_person_data(self, nations):
      df = pd.read_csv('data/time_series_19-covid-Confirmed.csv')
      nations_df = df[df['Country/Region'] == nations]
      return nations_df.iloc[0].loc[first_infection[nations]:]

    def recovered_person_data(self, nations):
      df = pd.read_csv('data/time_series_19-covid-Recovered.csv')
      nations_df = df[df['Country/Region'] == nations]
      return nations_df.iloc[0].loc[first_infection[nations]:]

    def extended_dates(self, index, new_dates):
        v = index.values
        dates_as_of_now = datetime.strptime(index[-1], '%m/%d/%y')
        while len(v) < new_dates:
            dates_as_of_now = dates_as_of_now + timedelta(days=1)
            v = np.append(v, datetime.strftime(dates_as_of_now, '%m/%d/%y'))
        return v

    def estimate(self, beta, gamma, data, recovered, nations):
        estimate_range = 150
        new_dates = self.extended_dates(data.index, estimate_range)
        size = len(new_dates)
        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        current = np.concatenate((data.values, [None] * (size - len(data.values))))
        recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        return new_dates, current, recovered, solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1),method='LSODA')

    def train_model(self):
        data = self.infected_person_data(self.nations)
        recovered = self.recovered_person_data(self.nations)
        optimum = minimize(function, [0.001, 0.001], args=(data, recovered), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        beta, gamma = optimum.x
        new_dates, current, recovered, estimation = self.estimate(beta, gamma, data, recovered, self.nations)
        for i in range(len(new_dates)):
        	new_dates[i] = datetime.strptime(new_dates[i], '%m/%d/%y')
        df = pd.DataFrame({'Confirmed': current, 'Recovered': recovered, 'S': estimation.y[0], 'I': estimation.y[1], 'R': estimation.y[2]}, index=new_dates)
        fig, ax = plt.subplots()
        ax.set_title(self.nations)
        df.plot(ax=ax, kind='line')
        fig.savefig(f"{self.nations}.png")


def function(point, data, recovered):
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA')
    solver1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    solver2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    alpha = 0.1
    return alpha * solver1 + (1 - alpha) * solver2


Spread_model = Spread_model('France', function)
Spread_model.train_model()
