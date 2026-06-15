# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from PIL import Image 
import plotly as plt

# %% [markdown]
# # E-news Express Project 

# %%
data = pd.read_csv('abtest.csv')

# %%
data.head()


# %% [markdown]
# Columns user_id, group (categorical variable: control and treatment), landing_page (cat.var. yes or no), time_sent_on_the_page (numerical), converted (cat. var. yes or no), language_preferred (cat. var.: English Spanish French)

# %%
len(data)

# %%
data.shape

# %% [markdown]
# data has form : 6 columns 100 rows

# %%
data.info()

# %% [markdown]
# Categorical variables are shown as "object" type and the only numerical varable of interest is "time_spent_on_the_page"  (type float).

# %%
data.describe()

# %% [markdown]
# Average time spenton the page is 5.38 min. Standard deviation is 2.38 min. Minimum time is 0.19 min. max. 10.71 min.  

# %% [markdown]
# # Checking the data for counts and  missing values. 

# %%
data.isna().sum()

# %% [markdown]
# Data is 'clean'.

# %%
data.group.value_counts(normalize=True, dropna=False)

# %% [markdown]
# Checked that 'control' (old page) and 'treatment' (visitng new page) groups have the same size (50/50 split).

# %%
data.landing_page.value_counts(normalize=True, dropna=False)

# %% [markdown]
# (Confirming that landing_page has the same sizes as control and treatment groups.) 

# %%
data.converted.value_counts(normalize=True, dropna=False)

# %% [markdown]
# 54% of all people visiting the page converted to subscription.

# %%
data.language_preferred.value_counts(normalize=True, dropna=False)

# %% [markdown]
# Correspondingly 34%, 34%and 32% preferred French, Spanish and English language.

# %%
data.language_preferred.value_counts()

# %%
data.converted.value_counts()

# %% [markdown]
# Conclusions: 
# * we have 'clean' data set for 100 customers. 
# * Data has 50-50 split for categorical variable "group", "landing_page".
# * For variable "converted" the split is 54 (yes) and 46 (no). 
# * there are 3 possible preferred languages: English French and Spanish.
# * Numerical variable "time spenton the page" has values from 0.19min. to 10.71min. with average 5.38 min. 

# %%


# %% [markdown]
# # Univariate analysis

# %% [markdown]
# Boxplot and histogram for time spent on the page.

# %%
sns.boxplot(data=data, x='time_spent_on_the_page',  showmeans=True, color="yellow")

# %%
sns.histplot(
        data=data, x='time_spent_on_the_page', bins=20,palette="winter",kde=True
    )  
#ax_hist2.axvline(
#        data['time_spent_on_the_page'].mean(), color="green", linestyle="--"
#    )  
#ax_hist2.axvline(
#        data['time_spent_on_the_page'].median(), color="black", linestyle="-"

# %% [markdown]
# # Different variables and different categorical data

# %%
sns.displot(data,x='time_spent_on_the_page',hue='language_preferred',stat="probability",element="step",kde=True,common_norm=False,palette='winter')

# %% [markdown]
# Distributions for times spent on the page when considering preferred language - do not look very diferent. This suggest the time spent on the page does not depend on the preferred language.

# %%
sns.displot(data,x='time_spent_on_the_page',hue='converted',stat="probability",kde=True,common_norm=False,element="step")

# %%
sns.displot(data,x='time_spent_on_the_page',hue='landing_page',kde=True,common_norm=False,element="step")

# %% [markdown]
# Distributions for time spent on the page for new and old pages look different. This suggest a trend: customers visitng newpage spent more time 
#     on the page

# %%
sns.displot(data,x='time_spent_on_the_page',hue='converted',common_norm=False,element="step")

# %% [markdown]
# Distribution for time spent on the page for the 2 groups of converted and  not converted to subscription look different. 

# %%
sns.displot(data,x='time_spent_on_the_page',hue='group',stat="probability",kde=True,common_norm=False)

# %% [markdown]
# #  Splitting data  into subsets 

# %% [markdown]
# I will later use this the dataframes created in this section to analyze in detailthe visuaization and the statistics related questions.

# %%
groupcontrol=data[data['group']=='control']

# %%
groupcontrol.describe()

# %%
grouptreatment=data[data['group']=='treatment']

# %%
grouptreatment.describe()

# %%
grE=data[data['language_preferred']=='English']

# %%
grE.describe()

# %%
grS=data[data['language_preferred']=='Spanish']

# %%
grS.describe()

# %%
grF=data[data['language_preferred']=='French']

# %%
grF.describe()

# %%
groupconverted=data[data['converted']=='yes']

# %%
groupconverted.describe()

# %%
groupnonconverted=data[data['converted']=='no']

# %%
groupnonconverted.describe()

# %%
grLPNEW=data[data['landing_page']=='new']

# %%
grLPNEW.describe()

# %%
grLPOLD=data[data['landing_page']=='old']

# %%
grLPOLD.describe()

# %%
grLPOLD.head()

# %% [markdown]
# #  Bi- and multivariate analysis

# %% [markdown]
# 
# I analyze firts the numerical variable- the time spent on the page using different "hue" value to visualize the full data. 
# Later I analyse other variables (categorical) - with binomial character: landing page (new or old), converted (yes or no), and multinomial: language preferred (English,French,Spanish). 
# I plot boxplots, histograms and KDE. 
# 

# %% [markdown]
# ### Visualization with respect to preferred language:

# %%
sns.displot(data=data,x='time_spent_on_the_page',kind='kde',hue='language_preferred')

# %% [markdown]
# Conclusion: In full data one cannot see differences for average times spent on page  for different preferred language groups (English 32%,French 34%,Spanish 34%)

# %% [markdown]
# Visualization for categorical variables:

# %%
sns.displot(data=data, x="converted", hue="language_preferred", multiple="dodge", shrink=.8,palette="winter")

# %% [markdown]
# There is a question number 3 relatedto this figure:Does the converted status dependsn preferred language.
# One can see differences in counts for each of the language, but without calculations  one cannot make definitive conclusion.   

# %%
sns.histplot(data=data, x="landing_page", hue="language_preferred", multiple="dodge", shrink=.8)

# %%
sns.histplot(data=groupconverted, x="time_spent_on_the_page", hue="language_preferred", multiple="dodge", shrink=.8)

# %%


# %%
sns.histplot(data=groupnonconverted, x="time_spent_on_the_page", hue="language_preferred", multiple="dodge", shrink=.8)

# %%
sns.histplot(data=groupcontrol, x="time_spent_on_the_page", hue="language_preferred", multiple="dodge", kde=True,shrink=.8)

# %% [markdown]
# Conclusion: In the "control" group (visiting old landing page) we see that distributions for different languages have diferent shapes. Without additional tests it is difficult to say someting based  only on visualization. 

# %%
sns.histplot(data=grouptreatment, x="time_spent_on_the_page", hue="language_preferred", kde=True,multiple="dodge", shrink=.8)

# %% [markdown]
# Figure: Average time spent on the new page for the 3 customer grups with different preferred language. Conclusion the differences are not significant enough to  say anything about average trends.  

# %% [markdown]
# For the "treatment" group (visiting new landing page)the distributions resemble normal distribution with mean between 4.5 nd 5.5 min. with different standard deviations.

# %%
sns.histplot(data=grE, x="time_spent_on_the_page", hue="converted",kde=True, multiple="dodge", shrink=.8)

# %% [markdown]
# Figure above - only for English preferred language sub-sample.
# Conclusion: In the English language group the visitors who converted to subsrcibing spent on average longer time on the page.

# %%
sns.histplot(data=grS, x="time_spent_on_the_page", hue="converted", multiple="dodge", kde=True,shrink=.8,palette='winter')

# %% [markdown]
# Conclusion: Spanish language group clearly those who converted to subsription on average spent longer time on the new page.

# %%
sns.histplot(data=grF, x="time_spent_on_the_page", hue="converted", multiple="dodge", kde=True, shrink=.8)

# %% [markdown]
# Conclusion: In the French laguage group, it is the same - those who converted to subsription on average spent longer time on the page.

# %%
g = sns.catplot(x="time_spent_on_the_page", y="converted", row="language_preferred",
                kind="box", orient="h", height=1.5, aspect=4,
                data=data,palette='rocket')

# %% [markdown]
# Conclusion: In boxplot here we can see that average times spent on the web page  for those who converted to subscrption for all 3 languages is similar.

# %%
g = sns.catplot(x="time_spent_on_the_page", y="language_preferred", row="converted",
                kind="box", orient="h", height=1.5, aspect=4,
                data=data, palette='winter')

# %% [markdown]
# It is the same figure - but one can more easily compare the averages, interquarile ranges in this view.

# %%
g = sns.catplot(x="time_spent_on_the_page", y="landing_page", row="converted",
                kind="box", orient="h", height=1.5, aspect=4,
                data=data)

# %%
g = sns.catplot(x="time_spent_on_the_page", y="converted", row="landing_page",
                kind="box", orient="h", height=1.5, aspect=4,
                data=grE)

# %% [markdown]
# Figure for preferred language = English customers. Customers who visited new page and converted to subscription spent on average 1 min.longer than those who  visited the old page and 2min.longer than those who did not convert to subscription. 

# %%
g = sns.catplot(x="time_spent_on_the_page", y="converted", row="landing_page",
                kind="box", orient="h", height=1.5, aspect=4,
                data=grS)

# %% [markdown]
# Preferred language - Spanish. For customers who converted status to subscription similar average times spent on both new and old version of the landing page.

# %%
g = sns.catplot(x="time_spent_on_the_page", y="converted", row="landing_page",
                kind="box", orient="h", height=1.5, aspect=4,
                data=grF)

# %% [markdown]
# Preferred language - French: Qualitatively we see similar trend as for Spanish speaking sample.

# %% [markdown]
# # Contingency tables

# %%
pd.crosstab(data['landing_page'],data['language_preferred'])

# %%
pd.crosstab(data['converted'],data['landing_page'])  #I repeat this later insection with inferential stat questions.

# %%
pd.crosstab(data['converted'],data['language_preferred'])


# %% [markdown]
# ### Barplots

# %% [markdown]
# To make sure I did not miss a trend I plot barplot type figures.  

# %%
sns.countplot(x='converted',data=data,hue='landing_page')

# %%
sns.countplot(x='converted',data=data,hue='language_preferred')

# %%
sns.barplot(x='language_preferred', y='time_spent_on_the_page', data=data,hue='converted')  # barplot

# %%
sns.barplot(x='language_preferred', y='time_spent_on_the_page', data=data,hue='landing_page',palette='winter')  # barplot

# %%
sns.catplot(x="group", y="time_spent_on_the_page", hue="language_preferred", kind="box", data=data)

# %%
sns.catplot(x="converted", y="time_spent_on_the_page", hue="language_preferred", kind="box", data=data)

# %% [markdown]
# To study if average time spent on page depends on language I check the averages are different:

# %%
# Check if the trend is similar across Usage/Fitness etc.
data.groupby(by = ['landing_page','language_preferred'])['time_spent_on_the_page'].mean()

# %% [markdown]
# ### Conclusions

# %% [markdown]
# Based on EDA:
# * Distributions for time spent on the page for new and old pages look different. This suggest a trend: customers visitng newpage spent more time 
#     on the page
#     
# * Distributions for times spent on the page when considering preferred language - do not look very diferent. This suggest the time spent on the page does not depend on the preferred language.
# 
# * Distribution for time spent on the page for the 2 groups of converted and  not converted to subscription look different. 
# * Converted status seems to depend on preferred language. One can see differences in counts for each of the language, but without calculations one cannot make definitive conclusion. 
# 
# * Preferred language-Eglish group: Customers who visited new page and converted to subscription spent on average 1 min.longer than those who visited the old page and 2min. longer than those who did not convert to subscription. Spanish and French language groups have different trend.
# * In the "control" group (visiting old landing page) we see that distributions for different languages have diferent shapes. Without additional tests it is difficult to say someting based on only visualization. 
# * Average time spent on the new page for the 3 customer grups with different preferred language. Conclusion the differences are not significant enough to  say anything about average trends.  
# * Average times spent on the web page for those who converted to subscrption for all 3 languages is similar.

# %% [markdown]
# # B. Inferential statistics part: questions,calculations,answers

# %% [markdown]
# # Q.2: Do the users spend more time on the new landing page than the old landing page?

# %% [markdown]
# ### Independent two-sample t-test: 

# %% [markdown]
# * First formulate the null and alternate hypothesis.
# 
# ð»0
# : ðœ‡1=ðœ‡2 (average time spent new landing page = average time spent on old landing page)
# 
# 
# ð»ð‘Ž
# : ðœ‡1â‰ ðœ‡2, There is a difference between times.
# 
# * Then we calculate the test-statistic and based on the p-value provide a conclusion.
# 
# Store relevant data in 2 dataframes grLPNEW and grLPOLD

# %%
grLPNEW.shape,

# %% [markdown]
# grLPNEW is dataframe that contains data for new landing page 

# %%
grLPNEW.describe()

# %%
grLPOLD.shape

# %% [markdown]
# grLPOLD is data frame that contains the data for old landing page.

# %%
grLPOLD.describe()

# %%
print('The mean time spent of the page for old landing page ' + str(round(grLPOLD['time_spent_on_the_page'].mean(),2)))
print('The mean  time spent of the page for new landing page ' + str(grLPNEW['time_spent_on_the_page'].mean()))
print('The standard deviation of time spent of the page for old landing page ' + str(round(grLPOLD['time_spent_on_the_page'].std(), 2)))
print('The standard deviation of time spent of the page for new landing page ' + str(round(grLPNEW['time_spent_on_the_page'].std(), 2)))

# %% [markdown]
# As we can see we have different average times and different standard deviations.

# %% [markdown]
# I use two sample independent t-test (we have 2 samples, continues variable,one-way)

# %%
from scipy import stats
x1=grLPOLD['time_spent_on_the_page']
#x1
x2=grLPNEW['time_spent_on_the_page']
t, p_value = stats.ttest_ind(x2, x1,equal_var=False, alternative='greater')
print("tstat = ",t, ", p_value = ", p_value)

# %% [markdown]
# The question we have to answer is if differences in averages are due to sampling or are significant. 

# %% [markdown]
# * p value is 0.00014 which is smaller than level of significance (0.05) so the null hypothesis can be rejected. 
# 
# * Conclusion: 
#     The users spend more time on the new landing page.(The observation is valid and not due to sampling error).

# %% [markdown]
# * This means that the new  page changed the output significantly.
# 
# * If we compare the means of the two sample distributions, we see that they are different numerically,statistically and visually.Inferential statistics results confirm the first impression conclusions from visualization.
# 
#     

# %% [markdown]
# Below I copy the relevant visualization:

# %%
sns.displot(data,x='time_spent_on_the_page',hue='landing_page',kde=True,common_norm=False,element="step")

# %% [markdown]
# ###  Additional check of the assumptions of the two sample t test

# %% [markdown]
# One of the assumptions of the 2 sample t-test is normality of distributions - below I test the 2 samples for it.

# %% [markdown]
# I use Shapiro-Wilk test:
# Null hypothesis - the distribution is normal.
# Alternative hypothesis the dstribution of the data in a sample is not normal. 

# %%
# Assumption 1: Normality
# Use the shapiro function for the scipy.stats library for this test

# find the p-value
w, p_value = stats.shapiro(grLPNEW['time_spent_on_the_page']) 
print('The p-value is', p_value)

# %%
# Assumption 1: Normality
# Use the shapiro function for the scipy.stats library for this test

# find the p-value
w, p_value = stats.shapiro(grLPOLD['time_spent_on_the_page']) 
print('The p-value is', p_value)

# %% [markdown]
# * Since p-value of the test is very large than the 5% significance level, we fail to reject the null hypothesis that the response follows the normal distribution.

# %% [markdown]
# * Conclusion the samples follow normal distribution. 

# %% [markdown]
# I perform now the Leveneâ€™s test for variances:
# 
# We will test the null hypothesis - ð»0 : the variances are equal
# 
# against the alternative hypothesis: ð»ð‘Ž : variances are different from the rest

# %%
#Assumption 2: Homogeneity of Variance
# use levene function from scipy.stats library for this test

# find the p-value
statistic, p_value = stats.levene(grLPNEW['time_spent_on_the_page'],grLPOLD['time_spent_on_the_page'])
print('The p-value is', p_value)

# %% [markdown]
# * p is 0.009 which is smaller than confidence level 0.05.
# 
# * We can reject null hypothesis that the variances are equal (because of that I used the Welchs testfor t-test function). (Well I know the variances from the data that they are different, as well).

# %% [markdown]
# # Q3. Is the conversion rate (the proportion of users who visit the landing page and get converted) for the new page greater than the conversion rate for the old page?

# %% [markdown]
# I create a contingency table using the pandas.crosstab() function

# %%
pd.crosstab(data['converted'],data['landing_page'])

# %% [markdown]
# Conversion rate:
# 
# C_new=33/50  
# C_old=21/50

# %% [markdown]
# ### Statistical analysis

# %% [markdown]
# First I define null and alternative hypotheses
# 
# ð»0:Conversion rate is the same for both versions of landing page.
# 
# ð»ð‘Ž: Conversion rate is greater for new landing page. C_new > C_old

# %% [markdown]
# ### Two proportion Z- test

# %% [markdown]
# Z-test assumptions:
# * Binomally distributed population - yes  (conversion variable can have values 'yes'or 'no')
# * Random sampling from the population - yes
# *  The standard thing is to check whether np and n(1-p) are greater than or equal to 10. - yes

# %%
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# set the counts of converted to subscription
x1 = np.array([33, 21])

# set the sample sizes 
x2 = np.array([50, 50])

# find the p-value
test_stat, p_value = proportions_ztest(x1, x2,alternative='larger')
print(p_value)

# %% [markdown]
# p value = 0.8% < 5% so we can reject null hypothesis that the proportions are the same, and we can conclude that conversion rate on new page is larger.  

# %% [markdown]
# ### Visualization of the data

# %% [markdown]
# For this question I copy below relevant figure: 

# %%
sns.countplot(x='converted',data=data,hue='landing_page')

# %% [markdown]
# New page (orange) has higher portion of coverted status than blue - old page.

# %% [markdown]
# Conclusion: Statistical analysis and visualization agree.  

# %% [markdown]
# # Q4. Does the converted status depend on the preferred language? [Hint: Create a contingency table using the pandas.crosstab() function]
# 

# %% [markdown]
# I create a contingency table using the pandas.crosstab() function

# %%
pd.crosstab(data['converted'],data['language_preferred'])

# %% [markdown]
# ### Statistical analysis - I sue the chi2 contingency test

# %% [markdown]
# * Null hypothesis = the converted status does not depend on preferred language.
# 
# * Alternative  hypothesis = converted status depends on the language.

# %%
import pandas as pd
from scipy.stats import chi2_contingency
df = pd.DataFrame({'yes': [21, 15,18], 'no': [11, 19,16]}, index = ['E', 'S','F'])
chi2, pval, dof, exp_freq = chi2_contingency(df)
print(pval)

# %% [markdown]
# Conclusion: p value  is larger than confidence level and it means we cannot reject null hypothesis. So we can conclude that converted status does not depend on preferred language.   

# %% [markdown]
# ### Data visualization:

# %%
sns.displot(data=data, x="converted", hue="language_preferred", multiple="dodge", shrink=.8,palette="winter")

# %% [markdown]
# Conclusions: 
# 
# * Visualization:  It is difficult tosee any trend in the data. 
# * Based on statistical analysis we can conclude that converted status does not depend on preferred language. 

# %% [markdown]
# # Q5. Is the mean time spent on the new page same for the different language users?

# %% [markdown]
# ### Visualization (copy of relevant figures) 

# %%

sns.barplot(x='language_preferred', y='time_spent_on_the_page', data=grLPNEW,palette='winter')  # barplot

# %%
sns.boxplot(x='language_preferred', y='time_spent_on_the_page', data=grLPNEW,palette='winter')  # barplot

# %%
sns.histplot(x='time_spent_on_the_page', data=grLPNEW,kde=True,hue='language_preferred')  # barplot

# %% [markdown]
# Boxplot visualitzation might be interpreted that the averages are different (English language preferred group spent the most time on average and French the least). 
# But histogram shows that the 3 distributions have slightly different properties. 

# %% [markdown]
# ### Statistical Analysis - one way ANOVA

# %% [markdown]
# First I define null and alternate hypotheses
# 
# ð»0: The time spent on new page with respect to each language category is equal.
# ð»ð‘Ž: At least one of the mean for time spent on new page with respect to the three laguages is different.
# 
# I select appropriate test
# 
# * This is a problem, concerning three means. One-way ANOVA could be the appropriate test here provided normality and equality of variance assumptions are verified.
# 
# * For testing of normality, I use Shapiro-Wilkâ€™s test.
# * For equality of variance,  I use Levene test below.
# 
# 

# %% [markdown]
# ### Shapiro-Wilkâ€™s test

# %% [markdown]
# We will test the null hypothesis first: 
# 
# ð»0:The time spent on new page follows a normal distribution
# 
# against the alternative hypothesis:
# 
# ð»ð‘Ž:The time spent on new page do not not follow a normal distribution

# %%
# Assumption 1: Normality- Use the shapiro function for the scipy.stats library for this test
# the p-value
w, p_value = stats.shapiro(grLPNEW['time_spent_on_the_page']) 
print('The p-value is', p_value)

# %% [markdown]
# Since the p-value is large than the 5% significance level, we fail to reject the null hypothesis.
# So the full data follows normal distribution.

# %% [markdown]
# One can check the 3 samples distributions:
# 

# %%
newpageE = grLPNEW[grLPNEW['language_preferred']=='English']['time_spent_on_the_page']
newpageS = grLPNEW[grLPNEW['language_preferred']=='Spanish']['time_spent_on_the_page']
newpageF = grLPNEW[grLPNEW['language_preferred']=='French']['time_spent_on_the_page']

# %%
# Assumption 1: Normality
# Use the shapiro function for the scipy.stats library for this test
# find the p-value

w1, p_value1 = stats.shapiro(newpageE) 
w2, p_value2 = stats.shapiro(newpageS) 
w3, p_value3 = stats.shapiro(newpageF) 
print('The p-valuee are', p_value1, p_value2, p_value3)

# %% [markdown]
# p value is larger than confidence level which means we cannt reject null hypothesis. The distributions are normal. 

# %% [markdown]
# 
# 

# %% [markdown]
# ### Levene test

# %% [markdown]
# We will test the null hypothesis
# 
#     ð»0 : All the population variances are equal
# 
# against the alternative hypothesis
# 
#     ð»ð‘Ž : At least one variance is different from the rest

# %%
#Assumption 2: Homogeneity of Variance
# use levene function from scipy.stats 

# find the p-value
statistic, p_value = stats.levene(newpageE,newpageS,newpageF)
print('The p-value is', p_value)

# %% [markdown]
# Since the p-value is large than the 5% significance level, we fail to reject the null hypothesis of homogeneity of variances.
# So all variances are equal.

# %% [markdown]
# ### One-way ANOVA

# %% [markdown]
# Define null and alternate hypotheses
# 
# ð»0: time spent on the new page doesnto depend on preferred language
# 
# ð»ð‘Ž : time spent on the new page depends on preferred language

# %%
# perform one-way anova test using the f_oneway function from scipy.stats library

test_stat, p_value = stats.f_oneway(newpageE,newpageS,newpageF)
print('The p-value is ', p_value)

# %% [markdown]
# Since the p-value is greater than the 5% significance level, we fail to reject the null hypothesis. Hence, we have not enough statistical evidence to say that the mean times spent on the new page  with respect to the preferred laguage are different.

# %%


# %%


# %%


