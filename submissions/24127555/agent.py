"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
import random
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First

    [DFS]
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = 2 # self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "DFS Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.last_position = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    self.last_position = my_position
                    return (move, 1)
            return (Move.STAY, 1)

        full_path = self._dfs_find_path(my_position, target, map_state)

        if full_path:
            # first step
            step1_pos = full_path[0]

            # checking last step
            if self.last_position is not None and step1_pos == self.last_position:
                alternative_move = self._find_non_repeating_move(my_position, map_state)
                if alternative_move:
                    self.last_position = my_position
                    return (alternative_move, 1)
                else:
                    self.last_position = None

            delta_row1 = step1_pos[0] - my_position[0]
            delta_col1 = step1_pos[1] - my_position[1]

            chosen_move = Move.STAY
            if delta_row1 == -1:
                chosen_move = Move.UP
            elif delta_row1 == 1:
                chosen_move = Move.DOWN
            elif delta_col1 == -1:
                chosen_move = Move.LEFT
            elif delta_col1 == 1:
                chosen_move = Move.RIGHT

            # second step
            if len(full_path) >= 2 and self.pacman_speed >= 2:
                step2_pos = full_path[1]
                delta_row2 = step2_pos[0] - step1_pos[0]
                delta_col2 = step2_pos[1] - step1_pos[1]

                if delta_row1 == delta_row2 and delta_col1 == delta_col2:
                    self.last_position = my_position
                    return (chosen_move, 2)

            if chosen_move != Move.STAY:
                self.last_position = my_position
                return (chosen_move, 1)

        fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        action = self._choose_action(my_position, fallback_moves, map_state, self.pacman_speed)
        if action:
            self.last_position = my_position
            return action

        return (Move.STAY, 1)
    
    # Helper methods (you can add more)
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _dfs_find_path(self, start: tuple, target: tuple, map_state: np.ndarray) -> list:
        if start == target:
            return []
        stack = [[start]]
        visited = {start}
        while stack:
            path = stack.pop()
            current_pos = path[-1]
            if current_pos == target:
                return path[1:]
            row, col = current_pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (row + dr, col + dc)
                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    new_path = list(path)
                    new_path.append(next_pos)
                    stack.append(new_path)
        return []

    def _find_non_repeating_move(self, current_pos: tuple, map_state: np.ndarray) -> Move:
        valid_alternatives = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)

            if self._is_valid_position(next_pos, map_state):
                if self.last_position is None or next_pos != self.last_position:
                    valid_alternatives.append(move)

        if valid_alternatives:
            return random.choice(valid_alternatives)
        return None

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        if enemy_position is None:
            valid_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    valid_moves.append(move)
            chosen_move = random.choice(valid_moves) if valid_moves else Move.STAY
            return chosen_move

        pacman_pos = enemy_position
        current_distance = self._calculate_distance(my_position, pacman_pos)
        scored_moves = {}

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

            if self._is_valid_position(next_pos, map_state):
                score = 0

                # 1
                next_dist = self._calculate_distance(next_pos, pacman_pos)
                if next_dist > current_distance:
                    score += 1

                # 2
                if next_pos[0] != pacman_pos[0] and next_pos[1] != pacman_pos[1]:
                    score += 2

                # 3
                if not self._is_dead_end(next_pos, map_state):
                    score += 5

                if score not in scored_moves:
                    scored_moves[score] = []
                scored_moves[score].append(move)

        if scored_moves:
            max_score = max(scored_moves.keys())
            best_moves = scored_moves[max_score]
            chosen_move = random.choice(best_moves)

            return chosen_move

        return Move.STAY
    
    # Helper methods (you can add more)
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _is_dead_end(self, pos: tuple, map_state: np.ndarray) -> bool:
        """"""
        row, col = pos
        valid_neighbors = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (row + dr, col + dc)
            if self._is_valid_position(next_pos, map_state):
                valid_neighbors += 1

        return valid_neighbors <= 1

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> int:
        """"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])