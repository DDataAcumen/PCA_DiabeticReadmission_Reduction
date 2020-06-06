# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:59:46 2020

@author: DemiiGod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv('diabetic_data.csv') 
dataset = dataset.drop(['weight', 'payer_code', 'diag_1','diag_2', 'diag_3'], axis=1)
dataset['race'] = dataset['race'].ffill()
dataset.medical_specialty.mode()
dataset["medical_specialty"].fillna("InternalMedicine", inplace = True) 

#Defining Dtypes
dataset[['race','gender', 'medical_specialty', 
         'change' , 'diabetesMed' ,
         'age']] = dataset[['race','gender',
                            'medical_specialty'
                            , 'change', 'diabetesMed', 'age']].astype('category')
dataset[['encounter_id', 'patient_nbr']] = dataset[['encounter_id',
                                                    'patient_nbr']].astype('int')                            
dataset[['admission_type_id', 'discharge_disposition_id',
         'admission_source_id', 'time_in_hospital']] = dataset[['admission_type_id', 
                                                                'discharge_disposition_id', 
                                                                'admission_source_id' , 
                                                                'time_in_hospital']].astype('int')
dataset[['num_lab_procedures' , 'num_procedures' ,
         'num_medications', 'number_outpatient' , 
         'number_emergency' , 'number_inpatient', 
         'number_diagnoses']] = dataset[['num_lab_procedures' , 'num_procedures' ,
         'num_medications', 'number_outpatient' , 
         'number_emergency' , 'number_inpatient', 
         'number_diagnoses']].astype("int")                                                        
dataset.dtypes                           
features_discription = dataset.describe(include = "all")

#Outlier ANalysis
plt.boxplot(dataset['number_diagnoses'] ,vert=True )
plt.ylabel('Number of Diagonsis Entered')
plt.title("Outliers in Attribute :Number of diagonsis")
plt.show()

plt.boxplot(dataset['time_in_hospital'] ,vert=True )
plt.ylabel('Time spent in Hospital(Days)')
plt.title("Outliers in Attribute :Time in hospital")
plt.show()

plt.boxplot(dataset['num_lab_procedures'] , vert=True)
plt.ylabel('Number of Lab Procedures for the patient')
plt.title("Outlier Analysis for numlab_procedures") 
plt.show()

plt.boxplot(dataset['num_procedures'] , vert=True)
plt.ylabel('Number of Procedures for the patient ')
plt.title("Outlier Analysis for num_procedures") 
plt.show()

plt.boxplot(dataset['num_medications'] , vert=True)
plt.ylabel('Number of Medication recorded at time of Admission ')
plt.title("Outlier Analysis for num_medications") 
plt.show()

dummy_dataset1 = dataset[(dataset['num_lab_procedures'] >= 8) & (dataset['num_lab_procedures'] <= 96) ]
dummy_dataset2 =  dataset[(dataset['num_medications'] >= 5) & (dataset['num_medications'] <= 35) ]

temp1 = pd.DataFrame()
temp1 = temp1.assign(encounter=dummy_dataset1['encounter_id'], 
                     labprocedure=dummy_dataset1['num_lab_procedures'])

Unwanted = pd.DataFrame()
Unwanted =  Unwanted.assign(encounter=dummy_dataset2['encounter_id'], 
                            medications=dummy_dataset2['num_medications'])

temp1['medications_no'] = Unwanted.medications
temp1 = temp1.dropna()

plt.boxplot(temp1['labprocedure'] , vert= True)
plt.title("Outlier Analysis for numlab_procedures after elimating Outliers")
plt.ylabel('Number of Lab Procedures for the patient ') 
plt.show()

plt.boxplot(temp1['medications_no'] , vert= True)
plt.title("Outlier Analysis for num_medications after elimating Outliers") 
plt.ylabel('Number of Medication recorded at time of Admission ')
plt.show()


plt.scatter(temp1['labprocedure'], temp1['medications_no'])
plt.xlabel('Number of Lab Procedures for the patient')
plt.ylabel('Medication recorded at the time of Admission')
plt.title('Scatter Plot')
plt.show()

temp1['Diagonsis_entered'] = dataset.number_diagnoses
plt.scatter(temp1['labprocedure'] , temp1['Diagonsis_entered'])
plt.xlabel('Number of Lab Procedures for the patient')
plt.ylabel('Number of Diagonsis Entered at the time of Admission')
plt.title("Scatter Plot" , fontsize = 14)
plt.show()

plt.scatter(temp1['Diagonsis_entered'] , temp1['medications_no'])
plt.xlabel('Number of Diagonsis Entered at the time of Admission')
plt.ylabel('Medication recorded at the time of Admission')
plt.title("Scatter Plot" , fontsize = 14)
plt.show()                                      


#Exploratory Analysis
AgeClassification = dataset.age.value_counts()
AgeClassification = AgeClassification.reset_index()
AgeClassification.columns = ['Age_Bins' , 'Number_of_Patients']
#AgeClassification = AgeClassification.sort_values('Number_of_Patients')
#AgeClassification.to_frame()
sns.barplot(x='Number_of_Patients' , y='Age_Bins' , 
            data=AgeClassification , order=AgeClassification['Age_Bins'], palette="coolwarm" , )
plt.title("Barplot:No of Patients Across each Age Group", fontsize =18)
plt.ylabel('Age Bins', fontsize=10)
plt.legend(labels=['Number of Patients'], loc=4)
plt.show() 


SexBifercation = dataset.gender.value_counts()
SexBifercation = SexBifercation.reset_index()
SexBifercation.columns = ['Sex_Patient', 'Total_No_of_Patients']
MalePercentage = SexBifercation.Total_No_of_Patients[1]/len(dataset)*100
FemalePercentage = SexBifercation.Total_No_of_Patients[0]/len(dataset)*100

#pip install pywaffle
from pywaffle import Waffle

data= {'Male': 46.23, 'Female': 53.75, 'NotDisclosed': 0.28}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    colors=("#983D3D", "#232066", "#DCB732"),
    title={'label': 'Waffle Graph:Patient Sex Bifercation', 'loc': 'left'},
    labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0}
)
fig.gca().set_facecolor('#EEEEEE')
fig.set_facecolor('#EEEEEE')
plt.show()


Stay_Patient_Hospital = dataset.time_in_hospital.value_counts()
Stay_Patient_Hospital = Stay_Patient_Hospital.reset_index()
Stay_Patient_Hospital.columns = ['No_of_Days' , 'Total_Number_of_Patients']
Labels = ['1-3 Days', '4-7 Days' , '8-11 Days', '12-14 Days']
Values = [49188 , 37288 , 11590 , 3700 ]
legend = ['48.4%' , '36.7%' , '11.3%' , '3.6%']

Donut=plt.Circle( (0,0), 0.7, color='white')
plt.pie(Values, labels=Labels, colors=['red','Skyblue','blue','Green'])
plt.title("Donut Chart: Patients Stay at Hospitals in terms of binned  Days", fontsize=16)
p=plt.gcf()
p.gca().add_artist(Donut)
p.legend(legend , loc= 'lower left')
plt.show()


#STACKED COLUMN PYTHON LINE OF CODE
# xvt = dataset.race.value_counts()
# xvt = xvt.reset_index()
# xvt.columns = ['Race' , 'Total_patients']
# xvs = dataset[(dataset['race']=='Caucasian')]
# tt = xvs.gender.value_counts()
# tt = tt.reset_index()
# tt.columns = ['Gender' , 'No_of_Patient']

# sns.barplot(x = xvt.Race , y =xvt.Total_patients , color = "red" , order = xvt.Race)
# bottom_plot = sns.barplot(x = tt.Gender, y = tt.No_of_Patient , color = "#0000A3")

# topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
# bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
# l = plt.legend([bottombar, topbar], ['Bottom Bar', 'Top Bar'], loc=1, ncol = 2, prop={'size':16})
# l.draw_frame(False) 

dummy_dataset3 = dataset[(dataset['admission_type_id'] <= 4)]
dummy_dataset3 = dummy_dataset3.admission_type_id.value_counts()
dummy_dataset3 = dummy_dataset3.reset_index()
dummy_dataset3.columns = ['Type of Admission' , 'Frequency']

temp2 = dummy_dataset3['Frequency']
temp2 = list(temp2)

v1 = 53990/91349*100
v2 = 18869/91349*100
v3 = 18480/91349*100
v4 = 10/91349*100

# from pywaffle import Waffle

data1= {'Emergency': 59.10  , 'Urgent' : 20.65  , 'Elective' : 20.23  ,}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data1,
    colors=("#983D3D", "#232066", "#DCB732"),
    title={'label': 'Waffle Graph:Admission Type for the patients', 'loc': 'left'},
    labels=["{0} ({1}%)".format(k, v) for k, v in data1.items()],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0}
)
fig.gca().set_facecolor('#EEEEEE')
fig.set_facecolor('#EEEEEE')
plt.show() 


dict1 = {'Insulin' : 54383,
         'Metformin' : 19998,
         'Glipizide' : 12686,
         'Glyburide' : 10650, 
         'Pioglitazone' : 7328,
         'Rosiglitazone' : 6365
         }

plt.bar(dict1.keys() , dict1.values() , color = 'Brown')
plt.xticks(rotation = 20)
plt.xlabel('Name of Medicine' , fontsize = 12)
plt.ylabel(' Number of Patients' , fontsize = 12)
plt.title('Top Medicine that were adviced by the Doctors' , fontsize = 16)
plt.show()

#Data Modelling
 X_df = dataset.gender
 X_df = X_df.replace(to_replace = "Unknown/Invalid" , value = "Female")
 X_df.value_counts()
 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X_df = le.fit_transform(X_df)
X_df = pd.DataFrame(X_df , columns=['Gender'])

Medication_Status = dataset.diabetesMed
Medication_Status = le.fit_transform(Medication_Status)
X_df['Medication_status'] = Medication_Status

X_df['Days_in_hospital'] = dataset.time_in_hospital
X_df['Number_of_Procedures'] = dataset.num_procedures
X_df['Number_of_lab_procedures'] = dataset.num_lab_procedures
X_df['Number_Outpatient_visit'] = dataset.number_outpatient
X_df['Number_Inpatient_visit'] = dataset.number_inpatient
X_df['Number_Emergency_visit'] = dataset.number_emergency
X_df['Number_of_DiagonsisEntered'] = dataset.number_diagnoses

X_df['EncounterNo'] = dataset.encounter_id
X_df = X_df.set_index('EncounterNo')

Y_df = pd.read_csv('NominalVar_Encoded.csv')
Y_df = Y_df.set_index('encounter_id')
Y_Dependent_df = Y_df.iloc[: , 41].values
Y_Dependent_df = pd.DataFrame(Y_Dependent_df , columns=['Readmission'])
Y_df = Y_df.drop(columns = ['readmitted'])

Var_Independent_df = pd.concat([X_df, Y_df], axis=1)


from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(Var_Independent_df , Y_Dependent_df , 
                                                       test_size = 0.2 , random_state = 0) 

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_varience = pca.explained_variance_ratio_

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
Y_test = Y_test.to_numpy()
#Y_pred = Y_pred.to_numpy()
Y_test = Y_test.astype(int)
Y_pred = Y_pred.astype(int)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , Y_pred)

Model_Accuracy = (9643+199+914)/20354*100
print("The Model Accuracy is '\n'",  str(Model_Accuracy) + 'percent')
Accuracy_NoReadmissionCase = (9643/10997)*100
print("Accuracy for No Readmission is '\n'",  str(Accuracy_NoReadmissionCase) + 'percent')




