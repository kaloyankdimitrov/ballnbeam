def rk4_step(f, x0, dt, u):
    """
    Input:
        f(x): function to be integrated
        x0: current state
        dt: time step 
        u: control input
    Output:
        x1: x(n + dt)
    """
    k1 = f(x0, u)
    k2 = f(x0 + dt*k1/2, u)
    k3 = f(x0 + dt*k2/2, u)
    k4 = f(x0 + dt*k3, u)
    x1 = x0 + 1/6*(k1+2*k2+2*k3+k4)*dt
    return x1