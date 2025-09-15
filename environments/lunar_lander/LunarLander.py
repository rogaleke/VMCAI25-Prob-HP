import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, FPS

class CustomLunarLander(LunarLander):
    def set_state(self, state=None):
        if state is None:
            return
        self.lander.position.x = state[0]
        self.lander.position.y = state[1]
        self.lander.linearVelocity.x = state[2]
        self.lander.linearVelocity.y = state[3]
        self.lander.angle = state[4]
        self.lander.angularVelocity = state[5]
        self.legs[0].ground_contact = state[6]
        self.legs[1].ground_contact = state[7]

    def get_state(self):
       
        state = []
        state.append(self.lander.position.x)
        state.append(self.lander.position.y)
        state.append(self.lander.linearVelocity.x)
        state.append(self.lander.linearVelocity.y)
        state.append(self.lander.angle)
        state.append(self.lander.angularVelocity)
        state.append(self.legs[0].ground_contact)
        state.append(self.legs[1].ground_contact)
        return state

    def get_normalized_state(self):
       
        pos = self.lander.position
        vel = self.lander.linearVelocity

        half_world_x = (VIEWPORT_W / SCALE) / 2.0  
        half_world_y = (VIEWPORT_H / SCALE) / 2.0 

        helipad_y = getattr(self, 'helipad_y', 0)

        norm_x = (pos.x - half_world_x) / half_world_x
        norm_y = (pos.y - (helipad_y + LEG_DOWN / SCALE)) / half_world_y
        norm_vx = vel.x * (half_world_x) / FPS
        norm_vy = vel.y * (half_world_y) / FPS
        norm_angle = self.lander.angle
        norm_ang_vel = 20.0 * self.lander.angularVelocity / FPS
        norm_leg1 = 1.0 if self.legs[0].ground_contact else 0.0
        norm_leg2 = 1.0 if self.legs[1].ground_contact else 0.0

        return np.array([norm_x, norm_y, norm_vx, norm_vy, norm_angle, norm_ang_vel, norm_leg1, norm_leg2], dtype=np.float32)
    
    def normalize_state(self, raw_state):
        
        half_world_x = (VIEWPORT_W / SCALE) / 2.0  
        half_world_y = (VIEWPORT_H / SCALE) / 2.0  
        helipad_y = getattr(self, 'helipad_y', 0)
        
        x_raw, y_raw, vx_raw, vy_raw, angle_raw, ang_vel_raw, leg1_contact, leg2_contact = raw_state
        

        norm_x      = (x_raw - half_world_x) / half_world_x
        norm_y      = (y_raw - (helipad_y + LEG_DOWN / SCALE)) / half_world_y
        norm_vx     = vx_raw * (half_world_x) / FPS
        norm_vy     = vy_raw * (half_world_y) / FPS
        norm_angle  = angle_raw
        norm_ang_vel = 20.0 * ang_vel_raw / FPS
        norm_leg1   = 1.0 if leg1_contact else 0.0
        norm_leg2   = 1.0 if leg2_contact else 0.0

        return [norm_x, norm_y, norm_vx, norm_vy, norm_angle, norm_ang_vel, norm_leg1, norm_leg2]
    
    def helper(self, xy):
        
        half_world_x = (VIEWPORT_W / SCALE) / 2.0  
        half_world_y = (VIEWPORT_H / SCALE) / 2.0  
        helipad_y = getattr(self, 'helipad_y', 0)
        
        # x_raw, y_raw, vx_raw, vy_raw, angle_raw, ang_vel_raw, leg1_contact, leg2_contact = raw_state
        
        x_raw, y_raw = xy

        norm_x      = (x_raw - half_world_x) / half_world_x
        norm_y      = (y_raw - (helipad_y + LEG_DOWN / SCALE)) / half_world_y
        # norm_vx     = vx_raw * (half_world_x) / FPS
        # norm_vy     = vy_raw * (half_world_y) / FPS
        # norm_angle  = angle_raw
        # norm_ang_vel = 20.0 * ang_vel_raw / FPS
        # norm_leg1   = 1.0 if leg1_contact else 0.0
        # norm_leg2   = 1.0 if leg2_contact else 0.0

        return norm_x, norm_y




if __name__ == "__main__":
    env = CustomLunarLander(render_mode='human')

    print(env.helper([8.8,4.13]))


    