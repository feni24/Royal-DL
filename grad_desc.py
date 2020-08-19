import numpy as np
import time

X = [0.5, 2.5]
Y = [0.2, 0.9]

def f(w,b,x):
    return 1.0 / (1.0 + np.exp(-(w*x + b)))

def error(w,b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5 * (fx - y) ** 2
    return err

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx * (1 - fx)

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx * (1 - fx) * x

def do_gradient_descent():
    startT = time.time()
    nume = 0
    w, b, eta, max_epochs = 2,2,5,100
    for i in range(max_epochs):
        nume += 1
        dw, db = 0, 0
        for x,y in zip(X, Y): 
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        w = w - eta * dw
        b = b - eta *db
    endT = time.time()
    print("Vanilla Gradient Descent")
    #print("Value of w: ",w)
    #print("Value of b: ",b)
    print("Error: ",error(w,b))
    #print("Epochs required: ",nume)
    #print("Execution Time: ",endT - startT)
    
def do_momentum_grad():
    startT = time.time()
    nume = 0
    w, b, eta, max_epochs = 2,2,5,100
    prev_v_w,prev_v_b,gamma=0,0,0.9
    for i in range(max_epochs): 
        nume += 1
        dw,db=0,0
        for x,y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
     
        v_w=gamma*prev_v_w+eta*dw
        v_b=gamma*prev_v_b+eta*db
        w=w-v_w
        b=b-v_b
        prev_v_w=v_w
        prev_v_b= v_b
    endT = time.time()
    print("\n\nMomentum based Gradient Descent")
    #print("Value of w: ",w)
    #print("Value of b: ",b)
    print("Error: ",error(w,b))
    #print("Epochs required: ",nume)
    #print("Execution Time: ",endT - startT)
    

def do_nesterov_accelerated_gradient_descent():
    
    w, b, eta, max_epochs = 2,2,5,100
    prev_v_w,prev_v_b,gamma=0,0,0.9
    
    for i in range(max_epochs):
        dw,db=0,0
        
        v_w=gamma*prev_v_w
        v_b=gamma* prev_v_b
        
        for x,y in zip(X, Y):
            dw += grad_w(w - v_w, b - v_b, x, y)
            db += grad_b(w - v_w, b - v_b, x, y)
        
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w
        prev_v_b = v_b
    print("\nNesterov Error:", error(w,b))

do_gradient_descent()
do_momentum_grad()
do_nesterov_accelerated_gradient_descent()
        

























