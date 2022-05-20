#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:01:25 2022

@author: jackgrady
"""


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib as plt


# Then the file has to be imported into Python

DF1 = pd.read_csv("PCE.csv")

DF1['PCE']=400*(np.log(DF1.PCE)-np.log(DF1.PCE.shift(1)))
DF1['CPCE']=DF1.PCE-DF1.PCE.shift(1)

# Then six lag variables are created
# Lag of the change in inflation 


DF1['CPCE_lag1']=DF1.CPCE.shift(1)
DF1['CPCE_lag2']=DF1.CPCE.shift(2)
DF1['CPCE_lag3']=DF1.CPCE.shift(3)
DF1['CPCE_lag4']=DF1.CPCE.shift(4)
DF1['CPCE_lag5']=DF1.CPCE.shift(5)
DF1['CPCE_lag6']=DF1.CPCE.shift(6)
DF1['CPCE_lag7']=DF1.CPCE.shift(7)


# Now run the regressions
# To see how inflation changes over a certain period of time


reg0 = smf.ols(formula= 'PCE ~ 1', data=DF1[DF1.DATE>="1980-01-01"])
reg1 = smf.ols(formula= 'PCE ~ CPCE_lag1+CPCE_lag2', data=DF1[DF1.DATE>="1980-01-01"])
reg2 = smf.ols(formula= 'PCE ~ CPCE_lag1+CPCE_lag2+CPCE_lag3', data=DF1[DF1.DATE>="1980-01-01"])
reg3 = smf.ols(formula= 'PCE ~ CPCE_lag1+CPCE_lag2+CPCE_lag3+CPCE_lag4', data=DF1[DF1.DATE>="1980-01-01"])
reg4= smf.ols(formula= 'PCE ~ CPCE_lag1+CPCE_lag2+CPCE_lag3+CPCE_lag4+CPCE_lag5', data=DF1[DF1.DATE>="1980-01-01"])
reg5= smf.ols(formula= 'PCE ~ CPCE_lag1+CPCE_lag2+CPCE_lag3+CPCE_lag4+CPCE_lag5+CPCE_lag6+CPCE_lag7', data=DF1[DF1.DATE>="1980-01-01"])

# And now the results
results = reg0.fit()
print(results.summary())


results = reg1.fit()
print(results.summary())

results = reg2.fit()
print(results.summary())

results = reg3.fit()
print(results.summary())

results = reg4.fit()
print(results.summary())

results = reg5.fit()
print(results.summary())



list=[reg0,reg1,reg2,reg3,reg4,reg5]

# The next code calculates AIC and BIC

results = [reg0,reg1,reg2,reg3,reg4,reg5]
for p in range(0,5):
    results.insert(p, list[p].fit())
    T=results[p].nobs
    ssr_T = sum(results[p].resid**2)/T
    ln_ssr_T = np.log(ssr_T)
    BIC=ln_ssr_T + (p+1)*np.log(T)/T
    AIC=ln_ssr_T + (p+1)*(2/T)
    print(f'p={p}, BIC={BIC}, AIC={AIC}')
