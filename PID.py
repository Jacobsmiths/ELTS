import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.last_time = None

    def compute(self, process_variable):
        # calculate change in time
        current_time = time.perf_counter()
        
        # Calculate error
        error = self.setpoint - process_variable
        
        # handles the first call
        if self.last_time == None:
            self.last_time = current_time
            self.previous_error = error
            return 0
        
        dt = current_time - self.last_time
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out
        
        # Update previous error
        self.previous_error = error

        # update last time to be when we last checked the clock
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.previous_error = 0
        self.integral = 0
        self.last_time = None
    
    def set_setpoint(self, setpoint):
        """Update the setpoint"""
        self.setpoint = setpoint