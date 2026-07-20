import sys
import random
from pathlib import Path
from collections import deque
import heapq
import numpy as np
import math

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
        self.name = "BFS Pacman"
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
        full_path = self._bfs_find_path(my_position, target, map_state)

        if full_path:
            step1_pos = full_path[0]

            if self.last_position is not None and step1_pos == self.last_position:
                alternative_move = self._find_non_repeating_move(my_position, map_state)
                if alternative_move:
                    self.last_position = my_position
                    return (alternative_move, 1)
                else:
                    self.last_position = None

            delta_row1 = step1_pos[0] - my_position[0]
            delta_col1 = step1_pos[1] - my_position[1]

            chosen_move = self._get_move_from_delta(delta_row1, delta_col1)

            if len(full_path) >= 2 and self.pacman_speed >= 2:
                step2_pos = full_path[1]
                delta_row2 = step2_pos[0] - step1_pos[0]
                delta_col2 = step2_pos[1] - step1_pos[1]
                next_move = self._get_move_from_delta(delta_row2, delta_col2)

                if next_move == chosen_move:
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

    def _get_move_from_delta(self, delta_row: int, delta_col: int) -> Move:
        for move in Move:
            if move.value == (delta_row, delta_col):
                return move
        return Move.STAY

    def _bfs_find_path(self, start: tuple, target: tuple, map_state: np.ndarray) -> list:
        if start == target:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current_pos, path = queue.popleft()

            if current_pos == target:
                return path

            row, col = current_pos
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (row + dr, col + dc)

                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

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
    - Đo chiều sâu ngõ cụt để sinh tồn (rất nhẹ CPU).
    - Ưu tiên bám trung tâm bản đồ để giữ thế chủ động (Center Control).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost 02 team"
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target_pacman = enemy_position or self.last_known_enemy_pos
        h, w = map_state.shape
        center = (h // 2, w // 2)

        # TRƯỜNG HỢP 1: Không rõ Pacman ở đâu -> Đi dạo nhưng hướng về trung tâm
        if target_pacman is None:
            move_choices = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    next_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
                    depth = self._get_corridor_depth(next_pos, my_position, map_state)
                    
                    # Tính khoảng cách từ ô tiếp theo đến trung tâm
                    dist_to_center = abs(next_pos[0] - center[0]) + abs(next_pos[1] - center[1])
                    move_choices.append((move, depth, dist_to_center))
            
            if move_choices:
                # Bước 1: Lọc ra các hướng an toàn nhất (độ sâu lớn nhất)
                max_depth = max([d for m, d, c in move_choices])
                safe_moves = [(m, c) for m, d, c in move_choices if d == max_depth]
                
                # Bước 2: Trong các hướng an toàn, chọn hướng gần trung tâm nhất
                min_center_dist = min([c for m, c in safe_moves])
                best_moves = [m for m, c in safe_moves if c == min_center_dist]
                
                return random.choice(best_moves)

            return Move.STAY

        # TRƯỜNG HỢP 2: Đang bị rượt đuổi
        current_dist = self._calculate_maze_distance(my_position, target_pacman, map_state)
        scored_moves = {}

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

            if self._is_valid_position(next_pos, map_state):
                score = 0
                next_dist = self._calculate_maze_distance(next_pos, target_pacman, map_state)

                # 1. Tiêu chí khoảng cách tránh Pacman (Quan trọng nhất)
                if next_dist > current_dist:
                    score += 100  
                elif next_dist == current_dist:
                    score += 30   
                else:
                    score -= 100

                # 2. Tránh đi cùng trục ngang/dọc với Pacman (Tránh tầm nhìn)
                if next_pos[0] != target_pacman[0] and next_pos[1] != target_pacman[1]:
                    score += 15

                # 3. Đếm ngã rẽ phụ (Sự linh hoạt cục bộ)
                escape_ways = self._count_valid_neighbors(next_pos, map_state)
                score += escape_ways * 5  

                # 4. ĐO CHIỀU SÂU NGÕ CỤT (Chống bẫy)
                corridor_depth = self._get_corridor_depth(next_pos, my_position, map_state)
                if corridor_depth == float('inf'):
                    score += 50  # Đường thông thoáng -> Thưởng lớn
                else:
                    # Phạt nặng ngõ cụt, nhưng bù lại bằng độ sâu (sâu hơn thì đỡ trừ hơn)
                    score -= 1000
                    score += corridor_depth * 5

                # 5. TIÊU CHÍ HƯỚNG TÂM (Bổ sung từ Phiên bản 2)
                dist_to_center = abs(next_pos[0] - center[0]) + abs(next_pos[1] - center[1])
                # Trừ điểm nhẹ nếu đi ra xa trung tâm.
                # Trọng số (-2) đảm bảo nó không phá hỏng việc chạy trốn (100) hay né ngõ cụt (-1000)
                score -= dist_to_center * 2 

                if score not in scored_moves:
                    scored_moves[score] = []
                scored_moves[score].append(move)

        if scored_moves:
            max_score = max(scored_moves.keys())
            best_moves = scored_moves[max_score]
            return random.choice(best_moves)

        return Move.STAY
    
    # --- CÁC HÀM BÊN DƯỚI GIỮ NGUYÊN HOÀN TOÀN TỪ PHIÊN BẢN 1 ---
    
    def _get_corridor_depth(self, next_pos: tuple, current_pos: tuple, map_state: np.ndarray) -> float:
        prev = current_pos
        curr = next_pos
        visited = {current_pos, next_pos}
        depth = 1 
        
        while True:
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n_pos = (curr[0] + dr, curr[1] + dc)
                if self._is_valid_position(n_pos, map_state):
                    neighbors.append(n_pos)
            
            if len(neighbors) <= 1:
                return depth
            
            if len(neighbors) > 2:
                return float('inf')
            
            next_step = neighbors[0] if neighbors[0] != prev else neighbors[1]
            
            if next_step in visited:
                return float('inf')
                
            visited.add(next_step)
            prev = curr
            curr = next_step
            depth += 1

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0 or map_state[row, col] == -1

    def _count_valid_neighbors(self, pos: tuple, map_state: np.ndarray) -> int:
        row, col = pos
        valid_count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (row + dr, col + dc)
            if self._is_valid_position(next_pos, map_state):
                valid_count += 1
        return valid_count

    def _calculate_maze_distance(self, start: tuple, target: tuple, map_state: np.ndarray) -> int:
        if start == target:
            return 0

        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        open_set = []
        heapq.heappush(open_set, (heuristic(start, target), 0, start))
        g_scores = {start: 0}

        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)

            if current == target:
                return current_g

            if current_g > g_scores.get(current, float('inf')):
                continue

            row, col = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (row + dr, col + dc)
                
                if self._is_valid_position(next_pos, map_state):
                    tentative_g_score = current_g + 1
                    
                    if tentative_g_score < g_scores.get(next_pos, float('inf')):
                        g_scores[next_pos] = tentative_g_score
                        f_score = tentative_g_score + heuristic(next_pos, target)
                        heapq.heappush(open_set, (f_score, tentative_g_score, next_pos))

        return heuristic(start, target)