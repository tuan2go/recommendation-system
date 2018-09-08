import csv
import numpy as np
import pandas as pd


def SVD(M):

    Mt=np.transpose(M)
    prd=np.dot(M, Mt)
    eigenvalue, eigenvec=np.linalg.eig(prd)
    
    #Indirect sort on eigenvalue to find out the proper indices, the same can 
    #be used with corresponding eigenvectors
    sortindex=eigenvalue.argsort()[::-1]
    
    #Sort Eigen values
    eigenvalue=eigenvalue[sortindex]    

    #To calculate sigma
    sigma=np.sqrt(abs(eigenvalue))
    sigma=np.around(sigma,decimals=2)    
    totalsigma=np.sum(sigma,dtype=float)
    
    #To Calculate Variance of Data preserved
    dim=600
    sumsigma=0.0
    cs=0
    while(cs<600):
        sumsigma+=sigma[cs]
        cs+=1
    print('We have', dim, 'components preserving',(sumsigma/totalsigma)*100,'% variance of data')
    sigma=sigma[0:dim]

    
    #To Calculate U - we had earlier calculated eigenvec for MMt
    #Sort and reduce U to nXdim
    U=eigenvec[:,sortindex]
    U=U[:,0:dim]
    U=np.real(U)
    U=np.around(U,decimals=2)
    
   
    
    #To Calculate V
    prd=np.dot(Mt,M)
    eigenvalue,eigenvec=np.linalg.eig(prd)
    sortindex=eigenvalue.argsort()[::-1]
    V=eigenvec[:,sortindex]
    V=V[:,0:dim]
    V=np.real(V)
    V=np.around(V,decimals=2) 
    
    return U,sigma,V
    


def query(q,V):
    #find q*v, w
    prd=np.dot(q,V)
    Vt=np.transpose(V)
    other=np.dot(prd,Vt)
    return other


#To Prepare list of movies - for recommending
print('Movie Recommender using SVD')

fileh=open('u.item','r',encoding='latin-1')
reader = csv.reader(fileh, delimiter='|')
movienames=list()
# The list of all the movies with movieid-1 as list index
count_movies = 0
for row in reader:
    count_movies += 1
    movienames.append(row[1])

fp2=open('u.data','r')
reader = csv.reader(fp2, delimiter='\t')
temp = set()
for index in reader:
    temp.add(index[0])
count_users = len(temp)

fp3=open('u.data','r')
reader2 = csv.reader(fp3, delimiter='\t')
m=list()
for j in range(count_users):
    m.append([0]*count_movies)
for row in reader2:
    m[int(row[0])-1][int(row[1])-1]=float(row[2])

M=np.array(m)
print(M)
U,sigma,V=SVD(M)

print("Enter userid (1-943)")
uid=int(input())    
q=m[uid-1]
predict=query(q,V)

#Sorting the user_rating row based on index
idx=predict.argsort()[::-1]
predicted=predict[idx]

#To display 10 movies, can change it by taking input from user
num_movies_display = int(input("How many relative movies do you want to display: "))
i=0
j=0

mr=list()

print("\n\nRecommended movies for UserID",uid,'\n')
while(i<num_movies_display):
    if(m[uid-1][idx[j]]==0):
        mr1=list()
        mr1.append(idx[j])
        mr1.append(movienames[idx[j]-1])
        mr.append(mr1)
        print(idx[j],'\t','\t',predict[idx[j]], j)
        i+=1
    j+=1

df=pd.DataFrame(mr,  columns=['MovieID', 'MovieName'])
for m in mr:
    print(m)
