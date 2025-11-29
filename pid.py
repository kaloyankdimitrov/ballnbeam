class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.setpoint = setpoint  # Desired value

        self.previous_error = 0
        self.integral = 0
        self.last_time = None

    def calculate(self, current_value, dt): 
        if dt == 0:  # Avoid division by zero if time hasn't advanced
            return 0

        error = self.setpoint - current_value

        # Proportional term
        proportional_term = self.Kp * error

        # Integral term
        self.integral += error * dt
        integral_term = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt
        derivative_term = self.Kd * derivative

        # Calculate total output
        output = proportional_term + integral_term + derivative_term

        # Update for next iteration
        self.previous_error = error

        return output
