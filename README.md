# Probability-Analysis
This tool calculates the probability distribution of the monitoring time series during a particular eruptive phase (like lava extrusion, intrusion etc.), and compares it to the repose periods and precursors. This allows to identify quantitatively relevant thresholds in the time series that indicate an active phase in the volcano. 

import pandas as pd
import numpy as np
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

##################################################
###################################################>>>>FUNCTIONS
#function that define the complement of two lists, = first list -second list
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

##################################################################
#Reads the file and asign a list to the columns MPEnergy and Date
HC=pd.read_csv('SeismisitasEP_MP2.csv')
HCount =list(HC["MPEnergy"])
Datel=list(HC['Date'])

##################################################
#We fill the gaps with zero

HCounts=[]
IndexGaps=[]
for i in range(0,len(HCount)):
        if min(HCount)<= HCount[i] <= max(HCount):
            HCounts.append(HCount[i])
        else:
            IndexGaps.append(i)
            HCounts.append(0)

#We ask if we want to fill the gaps with zero or drop those days

if len(IndexGaps)>1:
    quest2=input("There are gaps in the data! Do you want to fill them with zero? (Otherwise will be removed from the analysis) (y/n)")
    if quest2 =="y":
       nulllist=[-1]
    else: 
       nulllist=IndexGaps
else:
    nulllist=[-1]


##########-Intervals and labels-#######################
Ni=9 #number of intervals 
b=(max(HCounts)-np.exp(Ni)*min(HCounts))/(1-np.exp(Ni))
a=(min(HCounts)-b)

b2=-(max(HCounts)-(Ni^4+1)*min(HCounts))/(Ni^4) 
a2=(min(HCounts)-b2)

bn=[np.exp(i)*a+b for i in range(0,Ni+1)] #logaritmic scale
bn2=[np.exp(i)*a2+b2 for i in range(0,Ni+1)] #lx^4 scale

#histogram of data log and linear
His1=np.histogram(HCounts, bins=bn)[0]
His2=np.histogram(HCounts,bins=9)[0]
His3=np.histogram(HCounts,bins=bn2)[0]

#select the bins that have more counts
# 
if  np.median(His1)== max((np.median(His1),np.median(His2),np.median(His2))):
    bin0=bn
else:
    if np.median(His2) > np.median(His3):
        bin0=np.linspace(min(HCounts), max(HCounts),10)
    else:
        bin0=bn2


#if np.median(His1) > np.median(His2):
#    bin0=bn
#else:
#    bin0=np.linspace(min(HCounts), max(HCounts),10)


###########labels#######################
Labelbin=['['+str(round(bin0[i],1))+','+str(round(bin0[i+1],1))+')' for i in range(0,9)]
Label2=[str(round(bin0[i+1],1)) for i in range(0,9)]

########################################################
###################################################
#Plot of data
Date2= dates.datestr2num(Datel) #Import Dates
s = pd.DataFrame({'Values': HCounts}, index=Date2)

fig, ax = plt.subplots(figsize=[12,6])
ax.plot(s, lw=1)
ax.xaxis.set_major_locator(YearLocator()) #Import Year/Month locator
ax.xaxis.set_major_formatter(DateFormatter("\n%Y"))
ax.xaxis.set_minor_locator(MonthLocator((1,12)))
ax.xaxis.set_minor_formatter(DateFormatter("%b"))
plt.xlabel('Seismic Energy ') #Change according to selected data
plt.xlabel('Dates ')
plt.axhline(y=bin0[1], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[2], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[3], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[4], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[5], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[6], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[7], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[8], color="black", linestyle="--",linewidth=0.8)
plt.axhline(y=bin0[9], color="black", linestyle="--",linewidth=0.8)
plt.show()

##################################################
#Plot of probability distribution
AllDist=np.histogram(HCounts, bins=bin0)[0]

plt.plot(range(0,9),AllDist/sum(AllDist))
plt.xticks(range(0,9), Labelbin, rotation=90)
plt.ylabel('Probability Distribution')
plt.xlabel('Intervals of Seismic Energy (input units)')
plt.legend(['All data'], loc='upper right')
plt.subplots_adjust(bottom=0.35)
plt.show()

##################################################
#Description of Data

print('Statistic description of the time series:')
print('Total number of days: '+ str(len(HCounts)))
print('Percentage of gaps in data= '+ str(round(len(IndexGaps)/len(HCounts),5))+'%')
print('time series mean= '+ str(round(np.mean(HCounts),5)))
print('standard deviation= '+ str(round(np.std(HCounts),5)))
print('time series median= '+ str(round(np.median(HCounts),5)))
print('Max value in data= '+ str(round(max(HCounts),5)))
print('Min value in data= '+ str(round(min(HCounts),5)))

####################################

DateEvents = [] #Dates of onset/end for each event 
ErrDate=[] #uncertainty on the dates (not used in this code yet)

#We ask the user to input the dates of the events when there was lava extrusion (or any other eruptive phase)
while(True):
    nameInput=input("Onset and End separated by , (to finish write .) ")

    if nameInput!=".":      
       DateEvents.append(nameInput.split(","))
       uncInput=int(input("Write uncertainty +/- days for onset: "))
       EUncInput=int(input("Write uncertainty +/- days for end: "))
       ErrDate.append(uncInput)
       ErrDate.append(EUncInput)
    else:
        break


#############################################################
##we define a vector with the uncertainties, if the possible combinations are small 
##the vector runs over all combinations otherwise we use MonteCarlo to select a random vector within the possible range

Ncomb=np.prod([2*i+1 for i in ErrDate])
NN=1 #num of times we will iterate randomly


nmax=min(Ncomb,NN)-1 if min(Ncomb,NN)-1>=1 else 1


def Vunc(nn,vec=[]):#vector of uncertainty
    if Ncomb>NN:
        vec=[] #define a random vector (Monte Carlo)
        for j in ErrDate:
            if j!=0:
                vec.append(np.random.randint(-j,j))
            else:
                vec.append(0)
    else: 
        v=[-i for i in ErrDate]
        v_0=[-i for i in ErrDate]
        n=0
        vv=[[-i for i in ErrDate]]#append all possible combinations
        while n< len(ErrDate):
            v[n]+=1
            if v[n]>ErrDate[n]:
                v[n]=v_0[n]
                n+=1
            else:
                n=0
                vv.append([v[i] for i in range(len(v))])
        vec=vv[nn]
    return vec

#######################################################################

#trial dates
#2/14/1994,11/22/1994
#9/9/1996,1/29/1997
#6/9/1997,9/22/1997
#6/30/1998,1/1/1999
#10/31/2000,2/10/2001
#4/10/2006,10/30/2006
#10/21/2010,11/8/2010


#localize the index of the dates given
OnEnd0 = [Datel.index(i) for x in DateEvents for i in x] 

#We plot the time series applying the "lava filter" in the loop
#first we make a moving window that calculate the distribution for that window
mwl=7 #moving window length
WH=[np.histogram(HCounts[i-6:i], bins=bin0)[0] for i in range(mwl,len(HCounts))] #list of histograms for the moving window
Dates3=[Date2[i] for i in range(7,len(HCounts))]#dates that correspond to the LF

listLavH=[] #histogram lava days, given a lava day prob to have a value
listPrecH=[] #histogram precursor days
listNLavH=[] #histogram days no lava
listPLav=[]# given a monitoring value prob to have lava
listPPrec=[]
listPNLav=[]
listPrec=[] #given a value >threshold prob to have a lava day
listSens=[] #fraction of days with lava above the threshold
listSpec=[]
listSpec=[]

pn= 15 #precursor days to set max to 15 days

for a in range(0,nmax):
    print(a)
    OnEnd =[OnEnd0[i]+Vunc(a)[i] for i in range(len(OnEnd0))]
#llists of the index for days with lava and precursors 
    Lx= [list(range(OnEnd[i],OnEnd[i+1]+1)) for i in range(0,len(OnEnd),2)]
    Px=[list(range(OnEnd[i]-pn,OnEnd[i])) for i in range(0,len(OnEnd),2)]
 #flatten list
    FLx= [i for x in Lx for i in x]
    FPx= [i for x in Px for i in x]
    AllLP= FLx + FPx 
#llists of the index for days with lava and precursors (we chose 15 days but this is arbitrary)
    LavIndex=diff(FLx,nulllist) #nullist removes the incomplete dates if user selects n
    PrecIndex= diff(diff(FPx,nulllist),list(range(-15,0))) #range (-15,0) removes the precursor days that go beyond the data range
    NoLIndex= diff(diff(list(range(0,len(Datel ))), AllLP),nulllist) #index for days with no activity
#3 clusters, energy values for the days with lava, days that were precursors and rest of days
    AllLava= [HCounts[i] for i in LavIndex]
    AllPrec= [HCounts[i] for i in PrecIndex]
    NoLava= [HCounts[i] for i in NoLIndex]
#calculate the histogram for each cluster using the intervals given in bin0 (either lineal or log) 
    LavH=np.histogram(AllLava, bins=bin0)[0]
    PrecH=np.histogram(AllPrec, bins=bin0)[0]
    NLavH=np.histogram(NoLava, bins=bin0)[0]
#Relative Probability for each cluster
    PLav=[LavH[i]/(LavH[i]+NLavH[i]+PrecH[i]) for i in range(0,len(LavH))]
    PPrec=[PrecH[i]/(LavH[i]+NLavH[i]+PrecH[i]) for i in range(0,len(LavH))]
    PNLav=[NLavH[i]/(LavH[i]+NLavH[i]+PrecH[i]) for i in range(0,len(LavH))]
#probability to be above threshold*************************************
    P1k=[sum(LavH[i:])/(sum(LavH[i:])+sum(NLavH[i:])+sum(PrecH[i:])) for i in range(0,len(NLavH))]#Prob to have phase above threshold
    Pk1=[sum(LavH[i:])/sum(LavH) for i in range(0,len(LavH))]#fraction of days with phase above threshold
    Pk2=[1-(sum(NLavH[i:])+sum(PrecH[i:]))/(sum(NLavH)+sum(PrecH)) for i in range(0,len(LavH))]#fraction of days with no phase above threshold
 #append to lists
    listLavH.append(LavH/sum(LavH))
    listPrecH.append(PrecH/sum(PrecH))
    listNLavH.append(NLavH/sum(NLavH))
    listPLav.append(PLav)
    listPPrec.append(PPrec)
    listPNLav.append(PNLav)
    listPrec.append(P1k)
    listSens.append(Pk1)
    listSpec.append(Pk2)


####Mean of lists
MLavH=np.mean(listLavH, axis=0) 
MPrecH=np.mean(listPrecH,axis=0)
MNLav=np.mean(listNLavH,axis=0)

MPLav=np.mean(listPLav, axis=0) 
MPPrec=np.mean(listPPrec, axis=0) 
MPNLav=np.mean(listPNLav, axis=0) 

MPrec=np.mean(listPrec,axis=0)#*****************
MSens=np.mean(listSens,axis=0)#*****************
MSpec=np.mean(listSpec,axis=0)#*****************


###Error of list
ELavH=np.std(listLavH, axis=0) 
EPrecH=np.std(listPrecH,axis=0)
ENLav=np.std(listNLavH,axis=0)

EPLav=np.std(listPLav, axis=0) 
EPPrec=np.std(listPPrec, axis=0) 
EPNLav=np.std(listPNLav, axis=0) 

EPrec=np.std(listPrec,axis=0)#*****************
ESens=np.std(listSens,axis=0)#*****************
ESpec=np.std(listSpec,axis=0)


#Plots of Relative Probability
fig, ax = plt.subplots()
ax.errorbar(range(0,9),MPLav,
            yerr=EPLav)
ax.errorbar(range(0,9),MPPrec,
            yerr=EPPrec)
ax.errorbar(range(0,9),MPNLav,
            yerr=EPNLav)
ax.legend(['Lava', 'Precursors','NoLava'], loc='upper right')
ax.set_xlabel('Intervals')
ax.set_ylabel('Probability to Observe a Phase')
plt.xticks(range(0,9), Labelbin, rotation=90)
plt.subplots_adjust(bottom=0.35)
plt.show()


#Plot of probability distribution for each cluster
####plots with errors
fig, ax = plt.subplots()
ax.errorbar(range(0,9),MLavH,
            yerr=ELavH)
ax.errorbar(range(0,9),MPrecH,
            yerr=EPrecH)
ax.errorbar(range(0,9),MNLav,
            yerr=ENLav)
ax.legend(['Lava', 'Precursors','NoLava'], loc='upper right')
ax.set_xlabel('Intervals')
ax.set_ylabel('Probability Distribution')
plt.xticks(range(0,9), Labelbin, rotation=90)
plt.subplots_adjust(bottom=0.35)
plt.show()

#Plots of Probs above threshold
fig, ax = plt.subplots()
ax.errorbar(range(0,9),MPrec,
            yerr=EPrec)
ax.errorbar(range(0,9),MSens,
            yerr=ESens)
ax.errorbar(range(0,9),MSpec,
            yerr=ESpec)
ax.legend(['Precision', 'Sensitivity','Specificity'], loc='upper right')
ax.set_xlabel('Threshold')
ax.set_ylabel('Probabilities above threshold')
plt.xticks(range(0,9), [round(bin0[i],1) for i in range(0,len(NLavH))], rotation=90)
plt.subplots_adjust(bottom=0.35)
plt.show()
