import numpy as np

# houses sizes in hundred meters
x_train = np.array([0.8, 1.0, 0.6, 1.2, 2, 1.6 ])
# houses prices in thousands of  mad
y_train = np.array([45, 60, 40, 80, 120, 100])


def compute_cost(x, y, w, b) :
    # number of training examples
    m = x.shape[0];

    cost_sum = 0

    for i in range(m) : 
        # model function
        f_wb = w*x[i] + b

        cost = (f_wb - y[i])**2

        cost_sum = cost_sum + cost

    final_cost = (1 / (2 * m)) * cost_sum 

    return final_cost


def gradient_descent(x, y) :
    w=0
    b=0
    rate = 0.1
    m = x.shape[0];

    # Partial derivative of jwb by w
    def derivative_w(x,y,w,b,m):
        sigma_sum=0
        for i in range(m) :
            # model function
            f_wb = w*x[i] + b
            sum_formula = (f_wb - y[i])*x[i]
            sigma_sum = sigma_sum + sum_formula
        derivative_w = sigma_sum/m
        return derivative_w
    
    # Partial derivative of jwb by b
    def derivative_b(x,y,w,b,m):
        sigma_sum=0
        for i in range(m) :
            # model function
            f_wb = w*x[i] + b
            sum_formula = (f_wb - y[i])
            sigma_sum = sigma_sum + sum_formula
        derivative_b = sigma_sum/m
        return derivative_b


    iterations = 0
    max_iterations = 50000
    epsilon = 1e-5
    dw = derivative_w(x, y, w, b, m)
    db = derivative_b(x, y, w, b, m)

    while (abs(dw) > epsilon or abs(db) > epsilon) and iterations < max_iterations:
        
        w = w - rate * dw
        b = b - rate * db

        # 3. REFRESH dw and db for the next check!
        dw = derivative_w(x, y, w, b, m)
        db = derivative_b(x, y, w, b, m)
        iterations += 1

    if iterations >= max_iterations : print("Reached max iterations before converging : learning rate is too small")
    else : print(f"Converged succesfully in {iterations} steps")    

    final_cost = compute_cost(x, y, w, b)

    return w, b, final_cost


final_w, final_b, final_cost = gradient_descent(x_train, y_train)

print(f"Optimal w : {final_w:.2f}")
print(f"Optimal b : {final_b:.2f}")
print(f"Cost      : {final_cost:.4f}")