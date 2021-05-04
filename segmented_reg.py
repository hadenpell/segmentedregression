#Author: hadenpell
## Functions associated with implementing a segmented regression

from numpy.linalg import lstsq
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score 

ramp = lambda u: np.maximum( u, 0 )
step = lambda u: ( u > 0 ).astype(float)

def find_breakpoint( X, Y, breakpoints ):
    nIterationMax = 100

    breakpoints = np.sort( np.array(breakpoints) )

    dt = np.min( np.diff(X) )
    ones = np.ones_like(X)

    for i in range( nIterationMax ):
        # Linear regression:  solve A*p = Y
        Rk = [ramp( X - xk ) for xk in breakpoints ]
        Sk = [step( X - xk ) for xk in breakpoints ]
        A = np.array([ ones, X ] + Rk + Sk )
        p =  lstsq(A.transpose(), Y, rcond=None)[0] 

        # Parameters identification:
        a, b = p[0:2]
        ck = p[ 2:2+len(breakpoints) ]
        dk = p[ 2+len(breakpoints): ]

        # Estimation of the next break-points:
        newBreakpoints = breakpoints - dk/ck 

        # Stop condition
        if np.max(np.abs(newBreakpoints - breakpoints)) < dt/5:
            break

        breakpoints = newBreakpoints
    else:
        print('')

    # Compute the final segmented fit:
    Xsolution = np.insert( np.append( breakpoints, max(X) ), 0, min(X) )
    ones =  np.ones_like(Xsolution) 
    Rk = [ c*ramp( Xsolution - x0 ) for x0, c in zip(breakpoints, ck) ]

    Ysolution = a*ones + b*Xsolution + np.sum( Rk, axis=0 )

    return breakpoints[0]

#Predict consumption for a simple regression equation
def predict_simple(x, eq1, eq2, bkpt):
    #extract eq1 and eq2 coef and int
    eq1coef = eq1[0]
    eq1int = eq1[1]
    eq2coef = eq2[0]
    eq2int = eq2[1]
    
    #if temp <= breakpoint, use equation 1
    if(x <= bkpt):
        c = (eq1coef * x) + eq1int
        return c
    #otherwise use equation 2
    else:
        c = (eq2coef * x) + eq2int
        return c
    
#Predict consumption for a multiple regression equation with 2 variables
#Parameters:
#x- array of predictor variable 1
#y - array of predictor variable 2
#eq1 - the first equation as an array of 2 variables and 1 intercept
#eq2 - the second equation 
#bkpt - the breakpoint at which the other equation should be used
#Returns: an array of predicted values
def predict_multiple2(x, y, eq1, eq2, bkpt):
    
    if((len(eq1)!=3) | (len(eq2)!=3)):
        #if length = 2 this will be a simple regression
        raise Exception("Equations must have 2 variables and 1 intercept.")
    elif(len(x)!=len(y)):
        raise Exception("Length of X must equal length of Y")
    
    predicted = []
    
    #equation 1 coefs
    eq1coef1 = eq1[0]
    eq1coef2 = eq1[1]
    eq1int = eq1[2]
    
    #equation 2
    eq2coef1 = eq2[0] 
    eq2coef2 = eq2[1]
    eq2int = eq2[2]
    
    for i in range(len(x)):
        #if x <= breakpoint, use equation 1
        if(x[i] <= bkpt):
            c = (eq1coef1 * x[i]) + (eq1coef2 * y[i]) + eq1int
            predicted.append(c)
        #otherwise use equation 2
        else:
            c = (eq2coef1 * x[i]) + (eq2coef2 * y[i]) + eq2int
            predicted.append(c)
        
    return np.array(predicted)


#Returns 2 equations and a breakpoint
def segmented_equations(df,Xcol1,Xcol2,Ycol):
    #Get temp only, for use in finding breakpoint for electric data, and Y for predicted values for
    #use in breakpoint algorithm
    X_temp = np.array(df[Xcol1])
    median = df[Xcol1].median()
    Y_all = df[Ycol]

    breakpoint = find_breakpoint(np.array(df[Xcol1]),Y_all,[median])
    print(breakpoint)

    #separate df into 2
    df_1 = df[df[Xcol1]<=breakpoint]
    df_2 = df[df[Xcol1]>breakpoint]

    X_1 = df_1[[Xcol1,Xcol2]]
    Y_1 = df_1[Ycol]

    X_2 = df_2[[Xcol1,Xcol2]]
    Y_2 = df_2[Ycol]

    #build/fit the regression models
    linreg_1 = linear_model.LinearRegression().fit(X_1,Y_1)
    linreg_2 = linear_model.LinearRegression().fit(X_2,Y_2)

    #get equations
    equation1 = linreg_1.coef_[0],linreg_1.coef_[1],linreg_1.intercept_
    equation2 = linreg_2.coef_[0],linreg_2.coef_[1],linreg_2.intercept_
    
    return equation1,equation2,breakpoint

#Performs a multiple segmented regression and finds r2 scores
def segmentedreg(df,Xcol1,Xcol2,Ycol):
    eq1, eq2, bkpt = segmented_equations(df,Xcol1,Xcol2,Ycol)
    predicted = predict_consumption_multiple(np.array(df[Xcol1]),np.array(df[Xcol2]),eq1,eq2,bkpt)
    print('R2:',r2_score(df[Ycol],predicted))
    
    
#Predicts outcome variable for a multiple regression
#Parameters:
#X - a multidimensional array where each dimension is an array of that predictor variable.
#For example, if predictor variables are temperature and days, the array would look like this: [[temp1,temp2,...tempn][day1,day2,..dayn]]
#It is assumed that the FIRST dimension is the breakpoint determinant variable - so if the breakpoint is 40, this means that
#the breakpoint corresponds to temperature, and the predictions will be made according to equation based on temperature
#Eq1 is an array of coefficients + 1 intercept: ex - [5,3,1] for 5x+3y+1 (equation before breakpoint)
#Eq2 is an array like Eq1. (after breakpoint)
#Breakpoint is the breakpoint.
def predict_multiple(x,eq1,eq2,breakpoint):
    if(len(eq1)!=len(eq2)):
        raise Exception("Equations must be of equal size/number of variables.")   
    
    #first, figure out how many variables there are by finding how long the equations are
    equation = len(eq1)
    
    #assume that the first dimension in x array is the breakpoint variable (determines which equation to use)
    bkpt_var = x[0]
    
    #predicted array
    predicted = []
    
    #iterate through x and for each dimension in x, 
    #apply the appropriate equation 
    for i in range (len(x)):
        prediction = 0
        for j in range (equation):
            if(j==(equation-1)):
            #you're at the intercept, add intercept to final answer
                prediction = prediction + eq1[j]
                predicted.append(prediction) #loop breaks 
                break
            else:
                #get length of x dimension
                #assumes all dimensions are of equal length
                dim_length = len(x[0])
                #If you have more dimensions than actual numbers inside dimensions, eg [[2,2][1,1][2,3]] 
                #has 3 dimensions, but only 2 values per dimension, and you are now at i>=values - break, you are done.
                if(i>=dim_length):
                    break
                else:
                    coef = eq1[j]
                    prediction = prediction + (x[j][i] * coef)
                
    return predicted                  
