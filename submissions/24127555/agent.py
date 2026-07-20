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
import heapq
from pathlib import Path
from collections import deque

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
        self.pacman_speed = 2  # self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "A* Pacman"
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

        # find the shortest path by BFS
        full_path = self._astar_find_path(my_position, target, map_state)

        if full_path:
            step1_pos = full_path[0]
            delta_row1 = step1_pos[0] - my_position[0]
            delta_col1 = step1_pos[1] - my_position[1]
            chosen_move = self._get_move_from_delta(delta_row1, delta_col1)

            if len(full_path) >= 2:
                step2_pos = full_path[1]
                delta_row2 = step2_pos[0] - step1_pos[0]
                delta_col2 = step2_pos[1] - step1_pos[1]
                next_move = self._get_move_from_delta(delta_row2, delta_col2)

                if next_move == chosen_move:
                    return (chosen_move, 2)

            if chosen_move != Move.STAY:
                return (chosen_move, 1)

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

    def _get_move_from_delta(self, delta_row: int, delta_col: int) -> Move:
        """Converts a coordinate offset into its corresponding Move enum direction."""
        for move in Move:
            if move.value == (delta_row, delta_col):
                return move
        return Move.STAY

    def _astar_find_path(self, start: tuple, target: tuple, map_state: np.ndarray) -> list:
        """Finds the shortest path from start to target using the A* search algorithm."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, target), 0, start, []))
        visited = {start: 0}

        while open_set:
            _, g, current_pos, path = heapq.heappop(open_set)

            if current_pos == target:
                return path

            row, col = current_pos
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (row + dr, col + dc)

                if self._is_valid_position(next_pos, map_state):
                    new_g = g + 1
                    if next_pos not in visited or new_g < visited[next_pos]:
                        visited[next_pos] = new_g
                        f = new_g + heuristic(next_pos, target)
                        heapq.heappush(open_set, (f, new_g, next_pos, path + [next_pos]))
        return []


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught

    Implement search algorithm to evade Pacman as long as possible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A* Ghost"
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move for Ghost.
        """
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos

        if threat is None:
            # No information about enemy - move randomly
            return self._random_move(my_position, map_state)

        current_steps_to_catch = self._astar_pacman_steps_to_catch(threat, my_position, map_state)

        normal_moves = []
        dead_end_moves = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])

            if self._is_valid_position(next_pos, map_state):
                exits = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if self._is_valid_position((next_pos[0] + dr, next_pos[1] + dc), map_state))

                next_steps_to_catch = self._astar_pacman_steps_to_catch(threat, next_pos, map_state)

                if exits <= 1:
                    dead_end_moves.append((move, next_pos, next_steps_to_catch))
                else:
                    normal_moves.append((move, next_pos, next_steps_to_catch))

        is_trapped = (current_steps_to_catch <= 2) and (
            all(steps <= 1 for _, _, steps in normal_moves) if normal_moves else True)

        if is_trapped:
            all_possible_moves = normal_moves + dead_end_moves
            best_survival_move = Move.STAY
            max_survival = -1

            for move, next_pos, next_steps in all_possible_moves:
                if (move, next_pos) in [(m, p) for m, p, _ in dead_end_moves]:
                    survival = self._calculate_dead_end_depth(next_pos, my_position, map_state)
                else:
                    survival = next_steps

                if survival > max_survival:
                    max_survival = survival
                    best_survival_move = move

            if best_survival_move != Move.STAY:
                return best_survival_move

        safe_normal_moves = [(m, steps) for m, _, steps in normal_moves if steps > 1]

        if safe_normal_moves:
            safe_normal_moves.sort(key=lambda x: x[1], reverse=True)
            return safe_normal_moves[0][0]

        if normal_moves: return normal_moves[0][0]
        if dead_end_moves: return dead_end_moves[0][0]

        return Move.STAY

    # --- Các hàm bổ trợ (Helper Methods) ---

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Checks if a move from the current position is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _random_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """Random movement when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)

        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move

        return Move.STAY

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Checks if a position is within boundaries and not a wall."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0 or map_state[row, col] == -1

    def _is_dead_end(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Determines if a position is a dead end with at most one exit."""
        row, col = pos
        valid_neighbors = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (row + dr, col + dc)
            if self._is_valid_position(next_pos, map_state):
                valid_neighbors += 1
        return valid_neighbors <= 1

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculates the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _calculate_dead_end_depth(self, start_pos: tuple, came_from: tuple, map_state: np.ndarray) -> int:
        """Calculates the depth of a dead end to measure survival turns."""
        depth = 1
        curr = start_pos
        prev = came_from
        while True:
            next_options = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                np_pos = (curr[0] + dr, curr[1] + dc)
                if self._is_valid_position(np_pos, map_state) and np_pos != prev:
                    next_options.append(np_pos)

            if len(next_options) == 1:
                depth += 1
                prev = curr
                curr = next_options[0]
            else:
                break

        return depth

    def _astar_pacman_steps_to_catch(self, start_pacman: tuple, target_ghost: tuple, map_state: np.ndarray) -> int:
        """Simulates the minimum game steps Pacman needs to catch Ghost using A*."""

        def heuristic(a, b):
            return (abs(a[0] - b[0]) + abs(a[1] - b[1])) // 2

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start_pacman, target_ghost), 0, start_pacman))
        visited = {start_pacman: 0}

        while open_set:
            _, steps, curr_pos = heapq.heappop(open_set)

            if (abs(curr_pos[0] - target_ghost[0]) + abs(curr_pos[1] - target_ghost[1])) < 2:
                return steps

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                for speed in [1, 2]:
                    next_pos = (curr_pos[0] + dr * speed, curr_pos[1] + dc * speed)
                    if speed == 2:
                        mid_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                        if not self._is_valid_position(mid_pos, map_state):
                            continue

                    if self._is_valid_position(next_pos, map_state):
                        new_steps = steps + 1
                        if next_pos not in visited or new_steps < visited[next_pos]:
                            visited[next_pos] = new_steps
                            f = new_steps + heuristic(next_pos, target_ghost)
                            heapq.heappush(open_set, (f, new_steps, next_pos))

        return 999