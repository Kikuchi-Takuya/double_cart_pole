"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class DoubleCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):

        print('test')

        self.gravity = 9.8

        self.masscart0 = 1.0
        self.masspole0 = 0.1
        self.total_mass0 = self.masspole0 + self.masscart0
        self.length0 = 0.5  # actually half the pole's length
        self.polemass_length0 = self.masspole0 * self.length0
        self.force_mag0 = 10.0
        self.tau0 = 0.02  # seconds between state updates
        self.kinematics_integrator0 = "euler"

        self.masscart1 = 1.0
        self.masspole1 = 0.1
        self.total_mass1 = self.masspole1 + self.masscart1
        self.length1 = 0.5  # actually half the pole's length
        self.polemass_length1 = self.masspole1 * self.length1
        self.force_mag1 = 10.0
        self.tau1 = 0.02  # seconds between state updates
        self.kinematics_integrator1 = "euler"


        # Angle at which to fail the episode
        #self.theta0_threshold_radians = 12 * 2 * math.pi / 360
        #self.x0_threshold = 2.4
        #self.theta1_threshold_radians = 12 * 2 * math.pi / 360
        #self.x1_threshold = 2.4

        self.theta0_threshold_radians = 12 * 2 * math.pi / 360
        self.x0_threshold = 10
        self.theta1_threshold_radians = 12 * 2 * math.pi / 360
        self.x1_threshold = 10

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high0 = np.array(
            [
                self.x0_threshold,
                np.finfo(np.float32).max,
                self.theta0_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        high1 = np.array(
            [
                self.x1_threshold,
                np.finfo(np.float32).max,
                self.theta1_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space0 = spaces.Box(-self.force_mag0, self.force_mag0, shape=(1,), dtype=np.float32)
        self.action_space1 = spaces.Box(-self.force_mag1, self.force_mag1, shape=(1,), dtype=np.float32)
        self.observation_space0 = spaces.Box(-high0, high0, dtype=np.float32)
        self.observation_space1 = spaces.Box(-high1, high1, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state0 = None
        self.state1 = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        #self.np_random0, seed = seeding.np_random(seed)
        #self.np_random1, seed = seeding.np_random(seed)
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def step(self, action):

        'step'

        #err_msg = "%r (%s) invalid" % (action[0], type(action[0]))
        #assert self.action_space.contains(action[0]), err_msg
        #err_msg2 = "%r (%s) invalid" % (action[1], type(action[1]))
        #assert self.action_space2.contains(action[1]), err_msg2

        x0, x0_dot, theta0, theta0_dot = self.state0
        x1, x1_dot, theta1, theta1_dot = self.state1

        force0 = np.clip(action[0], -self.force_mag0, self.force_mag0)[0]
        force1 = np.clip(action[1], -self.force_mag1, self.force_mag1)[0]
        #force = 20
        #force2 = -20
        #force0 = self.force_mag if action[0] == 1 else -self.force_mag
        #force2 = self.force_mag if action[1] == 1 else -self.force_mag
        #force = force0
        
        costheta0 = math.cos(theta0)
        sintheta0 = math.sin(theta0)
        costheta1 = math.cos(theta1)
        sintheta1 = math.sin(theta1)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp0 = (
            force0 + self.polemass_length0 * theta0_dot ** 2 * sintheta0
        ) / self.total_mass0
        thetaacc0 = (self.gravity * sintheta0 - costheta0 * temp0) / (
            self.length0 * (4.0 / 3.0 - self.masspole0 * costheta0 ** 2 / self.total_mass0)
        )
        xacc0 = temp0 - self.polemass_length0 * thetaacc0 * costheta0 / self.total_mass0

        if self.kinematics_integrator0 == "euler":
            x0 = x0 + self.tau0 * x0_dot
            x0_dot = x0_dot + self.tau0 * xacc0
            theta0 = theta0 + self.tau0 * theta0_dot
            theta0_dot = theta0_dot + self.tau0 * thetaacc0
        else:  # semi-implicit euler
            x0_dot = x0_dot + self.tau0 * xacc0
            x0 = x0 + self.tau0 * x0_dot
            theta0_dot = theta0_dot + self.tau0 * thetaacc0
            theta0 = theta0 + self.tau0 * theta0_dot

        self.state0 = (x0, x0_dot, theta0, theta0_dot)

        temp1 = (
            force1 + self.polemass_length1 * theta1_dot ** 2 * sintheta1
        ) / self.total_mass1
        thetaacc1 = (self.gravity * sintheta1 - costheta1 * temp1) / (
            self.length1 * (4.0 / 3.0 - self.masspole1 * costheta1 ** 2 / self.total_mass1)
        )
        xacc1 = temp1 - self.polemass_length1 * thetaacc1 * costheta1 / self.total_mass1

        if self.kinematics_integrator1 == "euler":
            x1 = x1 + self.tau1 * x1_dot
            x1_dot = x1_dot + self.tau1 * xacc1
            theta1 = theta1 + self.tau1 * theta1_dot
            theta1_dot = theta1_dot + self.tau1 * thetaacc1
        else:  # semi-implicit euler
            x1_dot = x1_dot + self.tau1 * xacc1
            x1 = x1 + self.tau1 * x1_dot
            theta1_dot = theta1_dot + self.tau1 * thetaacc1
            theta1 = theta1 + self.tau1 * theta1_dot

        self.state1 = (x1, x1_dot, theta1, theta1_dot)

        done0 = bool(
            #x0 < -self.x0_threshold
            #or x0 > self.x0_threshold
            theta0 < -self.theta0_threshold_radians
            or theta0 > self.theta0_threshold_radians
        )

        done1 = bool(
            #x1 < -self.x1_threshold
            #or x1 > self.x1_threshold
            theta1 < -self.theta1_threshold_radians
            or theta1 > self.theta1_threshold_radians
        )

        if (not done0) and (not done1):
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state0, dtype=np.float32), np.array(self.state1, dtype=np.float32), reward, done0, done1, {}

    def reset(self):
        #self.state0 = self.np_random0.uniform(low= -0.05, high= 0.05, size=(4,))
        #self.state1 = self.np_random1.uniform(low= -0.05, high= 0.05, size=(4,))
        self.state0 = self.np_random.uniform(low=[-8.00, -0.05, -0.05, -0.05], high=[-1.00, 0.05, 0.05, 0.05], size=(4,))
        self.state1 = self.np_random.uniform(low=[1.00, -0.05, -0.05, -0.05], high=[8.00, 0.05, 0.05, 0.05], size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state0, dtype=np.float32), np.array(self.state1, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 1200
        screen_height = 400

        world_width = self.x0_threshold * 2
        scale = screen_width / world_width 

        carty0 = 100  # TOP OF CART
        polewidth0 = 10.0
        polelen0 = scale * 2 * (2 * self.length0)
        cartwidth0 = 50.0
        cartheight0 = 30.0

        carty1 = 100  # TOP OF CART
        polewidth1 = 10.0
        polelen1 = scale * 2 * (2 * self.length1)
        cartwidth1 = 50.0
        cartheight1 = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l0, r0, t0, b0 = -cartwidth0 / 2, cartwidth0 / 2, cartheight0 / 2, -cartheight0 / 2
            axleoffset0 = cartheight0 / 4.0
            cart0 = rendering.FilledPolygon([(l0, b0), (l0, t0), (r0, t0), (r0, b0)])
            self.carttrans0 = rendering.Transform()
            cart0.add_attr(self.carttrans0)
            self.viewer.add_geom(cart0)
            l0, r0, t0, b0 = (
                -polewidth0 / 2,
                polewidth0 / 2,
                polelen0 - polewidth0 / 2,
                -polewidth0 / 2,
            )
            l1, r1, t1, b1 = -cartwidth1 / 2, cartwidth1 / 2, cartheight1 / 2, -cartheight1 / 2
            axleoffset1 = cartheight1 / 4.0
            cart1 = rendering.FilledPolygon([(l1, b1), (l1, t1), (r1, t1), (r1, b1)])
            self.carttrans1 = rendering.Transform()
            cart1.add_attr(self.carttrans1)
            self.viewer.add_geom(cart1)
            l1, r1, t1, b1 = (
                -polewidth1 / 2,
                polewidth1 / 2,
                polelen1 - polewidth1 / 2,
                -polewidth1 / 2,
            )

            pole0 = rendering.FilledPolygon([(l0, b0), (l0, t0), (r0, t0), (r0, b0)])
            pole0.set_color(0.8, 0.6, 0.4)
            self.poletrans0 = rendering.Transform(translation=(0, axleoffset0))
            pole0.add_attr(self.poletrans0)
            pole0.add_attr(self.carttrans0)
            self.viewer.add_geom(pole0)
            self.axle0 = rendering.make_circle(polewidth0 / 2)
            self.axle0.add_attr(self.poletrans0)
            self.axle0.add_attr(self.carttrans0)
            self.axle0.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle0)

            pole1 = rendering.FilledPolygon([(l1, b1), (l1, t1), (r1, t1), (r1, b1)])
            pole1.set_color(0.9, 0.7, 0.5)
            self.poletrans1 = rendering.Transform(translation=(0, axleoffset1))
            pole1.add_attr(self.poletrans1)
            pole1.add_attr(self.carttrans1)
            self.viewer.add_geom(pole1)
            self.axle1 = rendering.make_circle(polewidth1 / 2)
            self.axle1.add_attr(self.poletrans1)
            self.axle1.add_attr(self.carttrans1)
            self.axle1.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle1)

            self.track0 = rendering.Line((0, carty0), (screen_width, carty0))
            self.track0.set_color(0, 0, 0)
            self.viewer.add_geom(self.track0)
            self.track1 = rendering.Line((0, carty1), (screen_width, carty1))
            self.track1.set_color(0, 0, 0)
            self.viewer.add_geom(self.track1)

            self._pole0_geom = pole0
            self._pole1_geom = pole1

        if self.state0 is None:
            return None

        # Edit the pole polygon vertex
        pole0 = self._pole0_geom
        l0, r0, t0, b0 = (
            -polewidth0 / 2,
            polewidth0 / 2,
            polelen0 - polewidth0 / 2,
            -polewidth0 / 2,
        )
        pole0.v = [(l0, b0), (l0, t0), (r0, t0), (r0, b0)]

        pole1 = self._pole1_geom
        l1, r1, t1, b1 = (
            -polewidth1 / 2,
            polewidth1 / 2,
            polelen1 - polewidth1 / 2,
            -polewidth1 / 2,
        )
        pole1.v = [(l1, b1), (l1, t1), (r1, t1), (r1, b1)]

        x0 = self.state0
        cartx0 = x0[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans0.set_translation(cartx0, carty0)
        self.poletrans0.set_rotation(-x0[2])

        x1 = self.state1
        cartx1 = x1[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans1.set_translation(cartx1, carty1)
        self.poletrans1.set_rotation(-x1[2])
        

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

