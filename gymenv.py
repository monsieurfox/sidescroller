from os import environ
import numpy as np
import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from engine import GameEngine
from controller import GameController

# Not required unless we want to provide traditional gym.make capabilities
register(id='Sidescroller-v0',
         entry_point='gymenv:ShooterEnv',
         max_episode_steps=5000)


class ShooterEnv(gym.Env):
    '''
    Wrapper class that creates a gym interface to the original game engine.
    '''

    # Hints for registered environments; ignored otherwise
    metadata = {
        'render_modes': ['human'],
        'render_fps': 60
    }

    def __init__(self, render_mode=None):
        '''
        Initializes a new Gymnasium environment for the Shooter Game. Loads the
        game engine into the background and defines the action space and the
        observation space.
        '''

        super().__init__()
        self.render_mode = render_mode
        pygame.init()
        pygame.display.init()
        if self.render_mode != 'human':
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
        else:
            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Shooter')
            pygame.mixer.init()
        self.screen = pygame.display.get_surface()
        self.game = GameEngine(self.screen)

        # Discrete action space: 7 possible moves
        self.action_space = spaces.Discrete(7)

        # Observation: [dx, dy, health, exit_dx, exit_dy, ammo, grenades]
        low = np.array([-10000, -1000, 0, -10000, -10000, 0, 0], dtype=np.float32)
        high = np.array([10000, 1000, 100, 10000, 10000, 50, 20], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)


    def reset(self, seed=None, options=None):
        '''
        Resets the game environment for the beginning of another episode.
        '''
        self.step_count = 0
        self.game.reset_world()
        self.game.load_current_level()

        # Tracks observation and reward values across steps
        self.start_x = self.game.player.rect.centerx
        self.start_y = self.game.player.rect.centery

        # Initialize the variables I decided were important for debugging
        debug_info = {
            'player_health': self.game.player.health,
            'player_distance': (0, 0),
            'exit_distance': self._get_exit_offset(self.game.player)
        }

        # Return the initial game state
        observation, debug_info = self._get_observation()
        return observation, debug_info


    def step(self, action):
        '''
        Agent performs a single action in the game environemnt.
        '''
        controller = self._action_to_controller(action)
        self.game.update(controller)
        self.step_count += 1

        observation, debug_info = self._get_observation()
        reward = self._get_reward()
        terminated = not self.game.player.alive or self.game.level_complete
        truncated = self.step_count >= 1000

        return observation, reward, terminated, truncated, debug_info


    def render(self):
        ''' 
        Visually renders the game so that viewers can watch the agent play. The
        first time this function is called, it initializes the PyGame display
        and mixer (just like a real game). If the self. Every time that it is called, this
        function draws the game.
        '''
        # Do nothing if rendering is disabled
        if self.render_mode != "human":
            return

        # Draw the screen        
        self.game.draw()
        pygame.display.update()


    def _get_observation(self):
        p = self.game.player

        # Distance from start
        p_dx = p.rect.centerx - self.start_x
        p_dy = p.rect.centery - self.start_y

        # Exit distance
        exit_dx, exit_dy = self._get_exit_offset(p)

        # Create an observation (7 values)
        obs = [
            p_dx,
            p_dy,
            p.health,
            exit_dx,
            exit_dy,
            p.ammo,
            p.grenades
        ]

        # Create debug information
        debug_info = {
            'player_health': p.health,
            'player_distance': (p_dx, p_dy),
            'exit_distance': (exit_dx, exit_dy),
        }

        return np.array(obs, dtype=np.float32), debug_info
    

    def _get_exit_offset(self, player):
        min_dist = float('inf')
        closest_dx, closest_dy = 9999, 9999

        for tile in self.game.groups['exit']:
            dx = tile.rect.centerx - player.rect.centerx
            dy = tile.rect.centery - player.rect.centery
            dist = abs(dx) + abs(dy)
            if dist < min_dist:
                min_dist = dist
                closest_dx = dx
                closest_dy = dy

        return closest_dx, closest_dy


    def _get_reward(self):
        if not self.game.player.alive:
            return -100

        reward = 0.1 * (self.game.player.rect.centerx - self.start_x)
        if self.game.level_complete:
            reward += 100
        
        return reward


    def _action_to_controller(self, action):
        '''
        Converts an action (just an integer) to a game controller object.
        '''
        ctrl = GameController()
        if action == 0: ctrl.mleft = True
        elif action == 1: ctrl.mright = True
        elif action == 2: ctrl.jump = True
        elif action == 3: ctrl.jump = ctrl.mleft = True
        elif action == 4: ctrl.jump = ctrl.mright = True
        elif action == 5: ctrl.shoot = True
        elif action == 6: ctrl.throw = True
        return ctrl
