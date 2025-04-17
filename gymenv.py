from os import environ
import numpy as np
import pygame
from settings import SCREEN_WIDTH, SCREEN_HEIGHT
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from engine import GameEngine
from controller import GameController
from settings import TILEMAP

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

    def __init__(self, render_mode=None, start_at=1):
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
        self.game = GameEngine(self.screen, start_level=start_at)

        # Discrete action space: 7 possible moves
        self.action_space = spaces.Discrete(7)

        # self.obstacle_active = False
        # self.obstacles_passed = 0

        # Observation: [dx, dy, health, exit_dx, exit_dy, ammo, grenades]
        self.low = np.array([-10000, -1000, 0, 0, -10000, 0, 0, 0, 0, -1], dtype=np.float32)
        self.high = np.array([10000, 1000, 100, 10000, 10000, 50, 20, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)


    def reset(self, seed=None, options=None):
        '''
        Resets the game environment for the beginning of another episode.
        '''
        self.step_count = 0
        self.game.reset_world()
        self.game.load_current_level()

        self.obstacle_active = False
        self.obstacles_passed = 0
        self.prev_obstacles_passed = self.obstacles_passed
        self.prev_x = self.game.player.rect.centerx
        self.still_steps = 0
        self.prev_in_air = self.game.player.in_air
        
        self.p_direction = self.game.player.direction
        self.previous_p_direction = self.p_direction
        self.direction_swap_ct = 0

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

        return observation, reward, terminated, truncated, debug_info, self.step_count


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

        world_map = self.game.get_world_data()
        tile_size = TILEMAP.TILE_SIZE

        # Distance from start
        p_dx = p.rect.centerx - self.start_x
        p_dy = p.rect.centery - self.start_y

        # Exit distance
        exit_dx, exit_dy = self._get_exit_offset(p)

        obstacle = self._is_obstacle_ahead(p, world_map, tile_size)

        obstacle = obstacle if not p.in_air else False

        if obstacle:
            self.current_obstacle_x = p.rect.centerx // tile_size + 1
            self.obstacle_active = True
        
        if self.obstacle_active and self._did_pass_obstacle(p, self.current_obstacle_x, tile_size):
            self.obstacle_active = False  # Reset tracking
            self.obstacles_passed += 1

        exit_dir = 1 if exit_dx > 0 else (-1 if exit_dx < 0 else 0)  # -1 = left, 1 = right

        # Create an observation (10 values)
        obs = [
            p_dx,
            p_dy,
            p.health,
            abs(exit_dx),
            exit_dy,
            p.ammo,
            p.grenades,
            obstacle,
            p.in_air,
            exit_dir
        ]

        # Min-Max normalization
        normalized_obs = (np.array(obs, dtype=np.float32) - self.low) / (self.high - self.low)

        # Create debug information
        debug_info = {
            'player_health': p.health,
            'player_distance': (p_dx, p_dy),
            'exit_distance': (exit_dx, exit_dy),
            'obstacle_ahead': obstacle,
            'obstacles_passed': self.obstacles_passed,
            'in_air': p.in_air,
            'direction': p.direction
        }

        return np.array(normalized_obs, dtype=np.float32), debug_info
    
    def _is_obstacle_ahead(self, player, world_map, tile_size):
        num_rows = len(world_map)
        num_cols = len(world_map[0])

        tile_x = player.rect.centerx // tile_size
        tile_y = player.rect.centery // tile_size

        # Always check one tile to the right (assuming movement is right-facing)
        # or in the direction the bot was last facing
        direction = player.direction or 1  # Default to right if standing still

        tile_ahead_x = tile_x + direction
        tile_ahead_y = tile_y
        tile_below_y = tile_y + 1

        # Treat out-of-bounds as obstacles or pits
        if (
            tile_ahead_x < 0 or tile_ahead_x >= num_cols or
            tile_ahead_y < 0 or tile_ahead_y >= num_rows or
            tile_below_y < 0 or tile_below_y >= num_rows
        ):
            return True

        tile_ahead = world_map[tile_ahead_y][tile_ahead_x]
        tile_below = world_map[tile_below_y][tile_ahead_x]

        # Define what is considered empty and what is an obstacle
        pit = tile_below == TILEMAP.EMPTY_TILE
        obstacle = tile_ahead >= TILEMAP.DIRT_TILE_FIRST and tile_ahead <= TILEMAP.DIRT_TILE_LAST

        return pit or obstacle

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
        player = self.game.player
        if not player.alive:
            return -100

        reward = 0
        tile_size = TILEMAP.TILE_SIZE
        current_tile = player.rect.centerx // tile_size
        prev_tile = self.prev_x // tile_size
        tile_diff = current_tile - prev_tile

        # Movement reward: based on number of tiles moved
        if tile_diff > 0:
            reward += 2 * tile_diff
            self.still_steps = 0
        elif tile_diff < 0:
            reward -= 1.5 * abs(tile_diff)
            self.still_steps = 0
        else:
            self.still_steps += 1

        self.prev_x = player.rect.centerx

        # Penalize being still too long
        if self.still_steps > 20:
            penalty = min(5, 1 + 0.03 * self.still_steps)
            reward -= penalty

        # Reward passing obstacles
        if self.obstacles_passed > self.prev_obstacles_passed:
            reward += 75 * (self.obstacles_passed - self.prev_obstacles_passed)

        self.prev_obstacles_passed = self.obstacles_passed
        
        # initialize if object ahead
        obstacle_ahead = self._is_obstacle_ahead(player, self.game.get_world_data(), tile_size)
        obstacle_ahead = obstacle_ahead if not player.in_air else False

        # penalize for swapping directions too often
        if self.previous_p_direction != player.direction:
            self.direction_swap_ct += 1
            if self.direction_swap_ct > 5:
                reward -= min(3.0, 0.5 * self.direction_swap_ct)
        else:
            self.direction_swap_ct = 0

        self.previous_p_direction = player.direction

        # Movement heuristics
        if not player.in_air:
            reward += 0.05 if not obstacle_ahead else -1.0
        else:
            reward += 0.05 if obstacle_ahead else -0.1

        # Encourage jumping over obstacles at right time
        if not self.prev_in_air and player.in_air and obstacle_ahead:
            reward += 0.2

        if self.still_steps > 100:
            player.alive = False

        if self.game.level_complete:
            reward += 500

        return reward


    def _did_pass_obstacle(self, player, obstacle_tile_x, tile_size):
        # Get current tile position
        tile_x = player.rect.centerx // tile_size
        
        # If the bot has moved past the obstacle's x-tile
        return tile_x > obstacle_tile_x

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
