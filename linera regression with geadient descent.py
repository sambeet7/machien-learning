from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df

def sum_of_errors(b,m,points):
    totalerror=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalerror+=((y-(m*x+b)))**2
    return totalerror / float(len(points))

def step_gradient(b_current,m_current,points,learning_rate):
    b_gradient=0
    m_gradient=0
    n=float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+=-(1/n)*(y-((m_current*x)+b_current))
        m_gradient+=-(1/n)*x*(y-((m_current*x)+b_current))
    new_b=b_current-learning_rate*b_gradient
    new_m=m_current-learning_rate*m_gradient
    return [new_b,new_m]
    




def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
    b=starting_b
    m=starting_m
    for i in range(num_iterations):
        b,m=step_gradient(b,m,array(points),learning_rate)

    return[b,m]



def run():
    points=genfromtxt('data.csv',delimiter=',')
    x=[]
    y=[]

    for i in range(0,len(points)):
        x.append(points[i,0])
        y.append(points[i,1])

    x=np.array(x)
    y=np.array(y)
    #data=df.columns[1:]
    #x=list(data)
    learning_rate=0.0001   #hyperparameters
    #y=mx+b
    initial_b=1
    initial_m=1
    num_iterations=1000
    [b,m]=gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iterations)
    print(b)
    print(m)
    print(sum_of_errors(b,m,points))

    plt.scatter(x,y,color="b",marker="o",s=30)
    y_pred=b+m*x
    plt.plot(x,y_pred,color="r")
    plt.xlabel("hours studied")
    plt.ylabel("marks obtained")
    plt.show()


if __name__=='__main__':
    run()
    
