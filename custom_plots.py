import matplotlib.pyplot as plt
import jax.numpy as jnp

def set_plot_limits(ax, x_min, x_max, y_min, y_max):
    """
    Sets the plot limits with 20% margin of the data
    Input:
        ax: plot to set limits for
        x_min: minimum x data point 
        x_max: maximum x data point 
        y_min: minimum y data point 
        y_max: maximum y data point 
    Ouput:
        None 
    """
    ax.set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

class BallPlot():
    def __init__(self):
        pass
        self.fig, self.ax = plt.subplots(
            figsize=(8,3)
        )
        self.fig.set_constrained_layout(True)   # allow dynamic spacing

        self.ax.set_title('Ball Position')
        self.ball_pos_setpoint_t, self.ball_pos_setpoint = [], []
        self.ball_pos_actual_t, self.ball_pos_actual = [], []
        (self.ball_setpoint_line,) = self.ax.plot([], [], lw=2, label='setpoint')
        (self.ball_actual_line,) = self.ax.plot([], [], lw=2, label='measurement')
        self.ax.legend()

    def update_ball_plot(self, t, new_ball_setpoint, new_ball_actual, live=False):
        # pos
        self.ball_pos_setpoint_t.append(t)
        self.ball_pos_setpoint.append(new_ball_setpoint)
        self.ball_pos_actual_t.append(t)
        self.ball_pos_actual.append(new_ball_actual)
        if live:
            self.show_ball_plot(t)
    def show_ball_plot(self, t):
        self.ball_setpoint_line.set_data(self.ball_pos_setpoint_t, self.ball_pos_setpoint)
        self.ball_actual_line.set_data(self.ball_pos_actual_t, self.ball_pos_actual)
        ball_data = jnp.array(self.ball_pos_setpoint + self.ball_pos_actual)
        set_plot_limits(self.ax, 0, t, jnp.min(ball_data), jnp.max(ball_data))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class MotorPlot():
    def __init__(self):
        self.fig, self.ax = plt.subplots(
            nrows=3, ncols=1,
            figsize=(8, 9),
            gridspec_kw={'height_ratios': [1, 1, 1]}
        )

        self.fig.set_constrained_layout(True)
        self.ax[0].set_title('Position')
        self.theta_x, self.theta_y = [], []
        self.theta_setpoint_x, self.theta_setpoint_y = [], []
        (self.theta_line,)=self.ax[0].plot(self.theta_x, self.theta_y, lw=2, label='measurement')
        (self.theta_setpoint_line,)=self.ax[0].plot(self.theta_x, self.theta_y, lw=2, label='setpoint')
        self.ax[0].legend()
        self.ax[1].set_title('Velocity')
        self.thetadot_x, self.thetadot_y = [], []
        self.thetadot_setpoint_x, self.thetadot_setpoint_y = [], []
        (self.thetadot_line,)=self.ax[1].plot(self.thetadot_x, self.thetadot_y, lw=2, label='measurement')
        (self.thetadot_setpoint_line,)=self.ax[1].plot(self.thetadot_setpoint_x, self.thetadot_setpoint_y, lw=2, label='setpoint')
        self.ax[2].set_title('Acceleration')
        self.thetaddot_x, self.thetaddot_y = [], []
        (self.thetaddot_line,)=self.ax[2].plot(self.thetaddot_x, self.thetaddot_y, lw=2, label='measurement')

    def update_plot(self, t, x, setpoints, live=True):
        # pos
        if setpoints and len(setpoints):
            self.theta_setpoint_x.append(t)
            self.theta_setpoint_y.append(setpoints[0])
        self.theta_x.append(t)
        self.theta_y.append(x[0])
        # vel
        self.thetadot_x.append(t)
        self.thetadot_y.append(x[1])
        if setpoints and len(setpoints) > 1:
            self.thetadot_setpoint_x.append(t)
            self.thetadot_setpoint_y.append(setpoints[1])
        # acc
        self.thetaddot_x.append(t)
        if self.thetadot_x and self.thetadot_y and len(self.thetadot_x) > 1 and len(self.thetadot_y) > 1:
            self.thetaddot_y.append((x[1] - self.thetadot_y[-2])/(t - self.thetadot_x[-2]))
        else:
            self.thetaddot_y.append(0)
        if live:
            self.show_plot(t)
    def show_plot(self, t, area_diff = None):
        self.theta_line.set_data(self.theta_x, self.theta_y)
        self.theta_setpoint_line.set_data(self.theta_setpoint_x, self.theta_setpoint_y)
        self.thetadot_line.set_data(self.thetadot_x, self.thetadot_y)
        self.thetadot_setpoint_line.set_data(self.thetadot_setpoint_x, self.thetadot_setpoint_y)
        self.thetaddot_line.set_data(self.thetaddot_x, self.thetaddot_y)
        theta_data = jnp.array(self.theta_y + self.theta_setpoint_y)
        pos_min = jnp.min(theta_data)
        pos_max = jnp.max(theta_data)
        set_plot_limits(self.ax[0], 0, t, pos_min, pos_max)
        thetadot_data = jnp.array(self.thetadot_y + self.thetadot_setpoint_y)
        vel_min = jnp.min(thetadot_data)
        vel_max = jnp.max(thetadot_data)
        set_plot_limits(self.ax[1], 0, t, vel_min, vel_max)
        acc_min = jnp.min(jnp.array(self.thetaddot_y))
        acc_max = jnp.max(jnp.array(self.thetaddot_y))
        set_plot_limits(self.ax[2], 0, t, acc_min, acc_max)