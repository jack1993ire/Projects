import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib as plt
from statsmodels.formula.api import ols


# This is the regression data
fileused = "covidemp.csv"


# Then the file has to be imported into Python

DF1 = pd.read_csv("covidemp.csv")

# After that, a dummy variable for employment
DF1['employed'] = np.where((DF1.EMPSTAT>=10)&(DF1.EMPSTAT<=12),1,0)

# A dummy variable saying the sample inludes those who
# Were in the labour force in February and May

DF1['after'] = DF1.MONTH.replace({2:0,5:1})


# A dummy variable that says people
# Are married and the spouse is present
DF1['married'] = np.where(DF1['MARST']==1,0,1)


# Now the dataset is further restricted by education
# To include people who graduated high school
# Or a Bachelor's degree or a doctorate degree
DF1['educlevel']=( (DF1['EDUC']==72) & (DF1['EDUC']==111) & (DF1['EDUC']==125) )

# A variable indicating if someone has 
# Any kind of difficulty 
DF1['diff']=np.where(DF1['DIFFANY']==2,1,0)

DF1['counts']=DF1.groupby('CPSIDP')['CPSIDP'].transform('count')




        
reg1 = smf.ols(formula='employed ~ after + married + diff', data=DF1)

results1= reg1.fit()

print (f' {results1.summary()}' )




