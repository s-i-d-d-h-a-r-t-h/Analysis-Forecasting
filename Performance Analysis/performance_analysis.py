import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



df_default=pd.read_excel('D:/CDAC/Content/Machine Learning/Datasets/Call centre dataset.xlsx',
                names=['Call_Id', 'Date', 'Agent', 'Department', 'Answered', 'Resolved', 'Speed_of_Answer', 'AvgTalkDuration', 'Satisfaction_rating', 'Hike'])
print('Sample dataset:\n', df_default.head(10))
print()
print('Dataset shape:\n', df_default.shape)
print()
print('Dataset info:\n', df_default.info())
print()

df=df_default.copy()


print('Duplicate values check:\n', df.duplicated().sum())
print()

null_df=pd.DataFrame(df.isna().sum()).reset_index()
null_df.columns=['feature_name', 'null_values']
null_df['null_%']=(null_df['null_values']/df.shape[0])*100
print('Null values check:\n', null_df)
print()


plt.rcParams['figure.figsize']=12,8
sns.heatmap(df.isna(), yticklabels=False, cmap='YlGnBu')
plt.title('Visualizing null values', fontsize=20)
plt.show()


df['Speed_of_Answer']=df['Speed_of_Answer'].fillna(df['Speed_of_Answer'].mean())
df['AvgTalkDuration']=df['AvgTalkDuration'].fillna(method='ffill')
df['Satisfaction_rating']=df['Satisfaction_rating'].fillna(method='ffill')


print('Null values after handling:\n', df.isna().sum().sum())

# ### 1. Accepting the week number and diaplaying the following details in that week:
# 
# - Total Calls, Calls Answered, Avg Speed of Answer, Abandon Rate, Avg Call/Min, Satisfaction Overall, Calls of Less than 180 Seconds, % Calls of Less than 180 Seconds, Satisfaction less than equal to 3.

df['Week_number']=df['Date'].dt.week
week_no=list(df['Week_number'].unique())


lsc=[]
for a in range(df.shape[0]):
    try:
        hr=df['AvgTalkDuration'][a].hour
        mn=df['AvgTalkDuration'][a].minute
        sc=df['AvgTalkDuration'][a].second
        scf=(hr*3600)+(mn*60)+sc
        lsc.append(scf)
    except:
        lsc.append(0)
df['duration_in_second']=lsc


def getfromweekno(wn):
    #duration_convert('AvgTalkDuration')
    df1=df[df['Week_number']==wn]
    
    tot_calls=df1['Answered'].count()
    print('Total number of calls                      -', tot_calls)

    ans_calls=df1['Answered'].value_counts()[0]
    print('Number of calls answered                   -', ans_calls)

    avg_speed=df1['Speed_of_Answer'].mean()
    print('Average speed of answer                    -', round(avg_speed,2))

    sat=df1['Satisfaction_rating'].mean()
    print('Satisfaction Overall                       -', round(sat))

    abndn_rate=(df1['Answered'].value_counts()[1]/df1['Answered'].value_counts()[0])*100
    print('Abandon calls rate                         -', round(abndn_rate,2))

    satl3=0
    for a in df1['Satisfaction_rating']:
        if a<=3:
            satl3=satl3+1
    print('Calls with satisfaction less than 3        -', satl3)
    
    CallLess180s=0
    for a in df1['duration_in_second']:
        if a<180:
            CallLess180s=CallLess180s+1
    print('Calls less than 180 Seconds is             -',CallLess180s)
    
    Lessthan180Sec=(CallLess180s/ans_calls)*100
    print('% Calls less than 180 Seconds              -', round(Lessthan180Sec,2))
    return


# ### 2. Accepting agent names and displaying the following details:
# - Total Calls, Calls Answered, Avg Speed of Answer, Call Resolution %, Call Resolved.

agents_name=list(df['Agent'].unique())


def getfromname(a):
    df2=df[df['Agent']==a]
    
    tot_calls=df2['Answered'].count()
    print('Total number of calls                      -', tot_calls)
    
    ans_calls=df2['Answered'].value_counts()[0]
    print('Number of calls answered                   -', ans_calls)
    
    avg_speed=df2['Speed_of_Answer'].mean()
    print('Average speed of answer                    -', round(avg_speed,2))
    
    res_call=df2['Resolved'].value_counts()[0]
    print('Number of Calls resolved                   -', res_call)
    
    res_precent=(df2['Resolved'].value_counts()[0]/df2['Resolved'].count())*100
    print('Abandon calls rate                         -', round(res_precent,2))
    
    sat=0
    for a in df2['Satisfaction_rating']:
        if a<=3:
            sat=sat+1
    print('Calls with satisfaction less than 3        -', sat)
    
    CallLess180s=0
    for a in df2['duration_in_second']:
        if a<180:
            CallLess180s=CallLess180s+1
    print('Calls less than 180 Seconds is             -',CallLess180s)
    return



# ### 3.  Accepting the department name and displaying the following details:
# 
# - Total Calls, Calls Answered, Avg Speed of Answer, Call Resolution, % Call Resolved.

dep_name=list(df['Department'].unique())


def getfromdep(b):
    df3=df[df['Department']==b]
    
    tot_calls=df3['Answered'].count()
    print('Total calls =', tot_calls)
    
    ans_calls=df3['Answered'].value_counts()[0]
    print('Answered calls =', ans_calls)
    
    avg_speed=df3['Speed_of_Answer'].mean()
    print('Average speed of answer =', round(avg_speed,2))
    
    res_call=df3['Resolved'].value_counts()[0]
    print('Calls resolved =', res_call)
    
    res_precent=(df3['Resolved'].value_counts()[0]/df3['Resolved'].count())*100
    print('abandon rate is', round(res_precent,2))
    
    sat=0
    for a in df3['Satisfaction_rating']:
        if a<=3:
            sat=sat+1
    print('Satisfaction less than 3 =', sat)
    
    CallLess180s=0
    for a in df3['duration_in_second']:
        if a<180:
            CallLess180s=CallLess180s+1
    print('Calls of Less than 180 Seconds is:',CallLess180s)
    return


# ### 5. Accepting week number to display abandoned calls % and resolved calls % employee-wise.


def abndn_emp(wn):
    
    df5=df[df['Week_number']==wn]
    lsemp=[]
    lsac=[]
    lcrp=[]
    for name in df5['Agent'].unique():
        dfn=df5[df5['Agent']==name]
        tc=dfn['Answered'].count()
        tca=dfn['Answered'].value_counts()[1]
        res_precent=round((dfn['Resolved'].value_counts()[0]/dfn['Resolved'].count())*100,2)
        abndn_percent=round((tca/tc)*100,2)
        lsemp.append(name)
        lsac.append(abndn_percent)
        lcrp.append(res_precent)
    d={'Employee':agents_name, 'Call abandon %':lsac, 'Call resolved %': lcrp}
    sub_df5=pd.DataFrame(d)
    print()
    print(sub_df5)
    print()
    print('Mean % abandoned calls - ',round(sum(lsac)/len(lsac),2))
    print('Mean % resolved calls - ',round(sum(lcrp)/len(lcrp),2))

    #PLOTTING
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize']=14,8
    plt.bar(sub_df5['Employee'], sub_df5['Call abandon %'], label='abandon %', color='orange')
    plt.axhline(y=sum(lsac)/len(lsac) , color='red', linestyle='dashed')
    for i in range(len(sub_df5['Employee'])):
        plt.annotate(xy=[agents_name[i], lsac[i]+1], s=round(lsac[i]), fontsize=15)
    
    plt.bar(sub_df5['Employee'], sub_df5['Call resolved %'], label='resolved %', color='green', alpha=0.5)
    plt.axhline(y=sum(lcrp)/len(lcrp) , color='red', linestyle='dashed')
    for i in range(len(sub_df5['Employee'])):
        plt.annotate(xy=[agents_name[i], lcrp[i]+1], s=round(lcrp[i]), fontsize=15)
    plt.xticks(agents_name, rotation=60, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Employees', fontsize=15)
    plt.ylabel('Percentage %', fontsize=15)
    plt.title('EMPLOYEE VS (CALL RESOLVED %, CALL ABANDON %)', fontsize=15)
    plt.ylim(0,120)
    plt.legend(loc=0)
    plt.show()
    
    
# ### 4. Accepting week number to display abandoned calls % and resolved calls % department-wise.


def abndn_dep(wn):
    
    dfad=df[df['Week_number']==wn]
    lsdep=[]
    lsac=[]
    lcrp=[]
    for dept in dfad['Department'].unique():
        dfn=dfad[dfad['Department']==dept]
        tc=dfn['Answered'].count()
        tca=dfn['Answered'].value_counts()[1]
        res_precent=round((dfn['Resolved'].value_counts()[0]/dfn['Resolved'].count())*100,2)
        abndn_percent=round((tca/tc)*100,2)
        lsdep.append(dept)
        lsac.append(abndn_percent)
        lcrp.append(res_precent)
    d={'Department':dep_name, 'Call abandon %':lsac, 'Call resolved %': lcrp}
    sub_dfad=pd.DataFrame(d)
    print()
    print(sub_dfad)
    print()
    print('Mean % abandoned calls - ',round(sum(lsac)/len(lsac),2))
    print('Mean % resolved calls - ',round(sum(lcrp)/len(lcrp),2))
   
    #PLOTTING
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize']=14,8
    plt.bar(sub_dfad['Department'], sub_dfad['Call abandon %'], label='abandon %', color='green', width=0.5)
    plt.axhline(y=sum(lsac)/len(lsac) , color='red', linestyle='dashed')
    for i in range(len(sub_dfad['Department'])):
        plt.annotate(xy=[dep_name[i], lsac[i]+1], s=round(lsac[i]), fontsize=15)
    
    plt.bar(sub_dfad['Department'], sub_dfad['Call resolved %'], label='resolved %', color='blue', width=0.5, alpha=0.5)
    plt.axhline(y=sum(lcrp)/len(lcrp) , color='red', linestyle='dashed')
    for i in range(len(sub_dfad['Department'])):
        plt.annotate(xy=[dep_name[i], lcrp[i]+1], s=round(lcrp[i]), fontsize=15)
    plt.xticks(dep_name, rotation=60, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('DEPARTMENT VS (CALL RESOLVED %, CALL ABANDON %)', fontsize=15)
    plt.xlabel('Departments', fontsize=15)
    plt.ylabel('Percentage %', fontsize=15)
    plt.ylim(0,120)
    plt.legend(loc=0)
    plt.show()



# ### 6. Accepting week number to display total satisfaction score of an employee in that week.


def getsat_agent(wn):
    dfsa=df[df['Week_number']==wn]
    
    lss=[]
    lss2=[]
    for name in dfsa['Agent'].unique():
        dfnn=dfsa[df['Agent']==name]
        abc=dfnn['Satisfaction_rating'].sum()
        lss.append(name)
        lss2.append(abc)
    
    d={'Agent Name':lss, 'Satisfaction score':lss2}
    sub_dfsa=pd.DataFrame(d)
    print(sub_dfsa)
    
    #PLOTTING
    plt.pie(lss2, labels=lss, shadow=True, radius=1.5, autopct='%0.2f%%', textprops={'fontsize': 20})
    plt.show()



df.to_csv('Employee_project (after data analysis).csv')

