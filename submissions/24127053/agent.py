"""
Mã số sinh viên: 24127053
Đội thi: Gr03_HideSeek
Thuật toán sử dụng: 
- Pacman (Seeker): A* Search tối ưu hóa đường đi ngắn nhất kết hợp bộ kích tốc (Speed Multiplier).
- Ghost (Hider): Khai thác cơ chế giới hạn tốc độ khúc cua của Pacman bằng cách tối đa hóa 
                 số lượt di chuyển thực tế của Pacman (Pacman-Steps Metric) để kéo dài thời gian sống sót.
"""

import sys
from pathlib import Path
import heapq
import random
import numpy as np

# Thêm thư mục src vào hệ thống để import chính xác các class giao diện từ framework
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman Agent (Seeker) - Đi săn Ghost dùng thuật toán A* và tối ưu tốc độ đường thẳng.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gr03_Seeker_AStar_24127053"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2))) 
        self.last_known_enemy_pos = None 
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                           if self._is_valid(self._apply_move(my_position, m), map_state)]
            return (valid_moves[0], 1) if valid_moves else (Move.STAY, 1)

        path = self._astar_find_path(my_position, target, map_state)
        
        if path:
            first_move = path[0]
            consecutive_steps = 1
            
            for next_move in path[1:self.pacman_speed]:
                if next_move == first_move:
                    consecutive_steps += 1
                else:
                    break
                    
            return (first_move, consecutive_steps)
            
        return (Move.STAY, 1)

    def _astar_find_path(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        if start == goal:
            return []
            
        h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        count = 0  
        open_set = [(h_start, count, 0, start, [])]
        visited = {start: 0}
        h_max, w_max = map_state.shape
        
        while open_set:
            _, _, g, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path 
                
            if g > visited.get(current, float('inf')):
                continue
                
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = current[0] + dr, current[1] + dc
                
                if 0 <= nr < h_max and 0 <= nc < w_max and map_state[nr, nc] == 0:
                    next_pos = (nr, nc)
                    new_g = g + 1
                    
                    if new_g < visited.get(next_pos, float('inf')):
                        visited[next_pos] = new_g
                        h_val = abs(nr - goal[0]) + abs(nc - goal[1])
                        count += 1
                        heapq.heappush(open_set, (new_g + h_val, count, new_g, next_pos, path + [move]))
                        
        return []

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        return (pos[0] + move.value[0], pos[1] + move.value[1])

    def _is_valid(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent (Ghost/Hider) - Chạy trốn thông minh, bắt Pacman tốn nhiều lượt đi nhất có thể.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gr03_Ghost_Evasive_24127053"
        self.last_known_pacman_pos = None
        self.path_cache = {} 
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            self.last_known_pacman_pos = enemy_position
            
        pacman_pos = enemy_position or self.last_known_pacman_pos
        
        # Thiết lập thứ tự ưu tiên cố định cho các nước đi để triệt tiêu tính ngẫu nhiên
        move_priority = {Move.UP: 0, Move.RIGHT: 1, Move.DOWN: 2, Move.LEFT: 3, Move.STAY: 4}
        
        if pacman_pos is None:
            # Chọn hướng đi hợp lệ đầu tiên theo độ ưu tiên nếu mất dấu Pacman
            for m in sorted(move_priority.keys(), key=lambda x: move_priority[x]):
                next_pos = (my_position[0] + m.value[0], my_position[1] + m.value[1])
                if self._is_valid_position(next_pos, map_state):
                    return m
            return Move.STAY
            
        r_g, c_g = my_position
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        
        # Reset cache đường đi ở mỗi lượt tính toán mới để tránh sai lệch dữ liệu
        self.path_cache = {}
        evaluated_moves = []
        
        for move in possible_moves:
            next_pos = (r_g + move.value[0], c_g + move.value[1])
            
            if not self._is_valid_position(next_pos, map_state):
                continue
                
            # Tìm đường đi ngắn nhất từ ô dự kiến của Ghost đến Pacman
            shortest_path = self._get_shortest_path(next_pos, pacman_pos, map_state)
            
            # Tính toán xem Pacman mất thực tế bao nhiêu LƯỢT ĐI (turns) để vượt qua quãng đường này
            pacman_turns_needed = self._calculate_pacman_turns(shortest_path)
            
            if pacman_turns_needed <= 1:
                # Vùng nguy hiểm chí mạng! Pacman có thể tóm gọn Ghost ngay trong lượt sau
                score = -100000.0 + pacman_turns_needed * 10
            else:
                # Điểm số cốt lõi: Càng khiến Pacman mất nhiều lượt tiếp cận thì điểm càng cực kỳ cao!
                score = pacman_turns_needed * 1000.0
                
                # Tránh các ngõ cụt nguy hiểm
                escape_routes = self._count_walkable_neighbors(next_pos, map_state)
                if escape_routes <= 1:
                    score -= 5000.0  # Phạt nặng ngõ cụt
                elif escape_routes >= 3:
                    score += 150.0   # Cộng thưởng ngã rẽ rộng mở để dễ luồn lách
                    
                # Phạt hành động đứng yên (chỉ chọn STAY khi các hướng khác cực kỳ nguy hiểm)
                if move == Move.STAY:
                    score -= 300.0
            
            # Lưu lại điểm số và độ ưu tiên để sắp xếp
            evaluated_moves.append((score, -move_priority[move], move))
            
        # Sắp xếp giảm dần theo điểm số, nếu bằng điểm thì ưu tiên nước đi có độ ưu tiên cao hơn (Deterministic)
        evaluated_moves.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        return evaluated_moves[0][2] if evaluated_moves else Move.STAY

    def _get_shortest_path(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """ Trả về toàn bộ danh sách các bước di chuyển ngắn nhất từ start đến goal """
        if start == goal:
            return []
            
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
            
        h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        count = 0
        open_set = [(h_start, count, 0, start, [])]
        visited = {start: 0}
        h_max, w_max = map_state.shape
        
        while open_set:
            _, _, g, current, path = heapq.heappop(open_set)
            
            if current == goal:
                self.path_cache[cache_key] = path
                return path
                
            if g > visited.get(current, float('inf')):
                continue
                
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < h_max and 0 <= nc < w_max and map_state[nr, nc] == 0:
                    next_pos = (nr, nc)
                    new_g = g + 1
                    if new_g < visited.get(next_pos, float('inf')):
                        visited[next_pos] = new_g
                        h_val = abs(nr - goal[0]) + abs(nc - goal[1])
                        count += 1
                        heapq.heappush(open_set, (new_g + h_val, count, new_g, next_pos, path + [move]))
                        
        self.path_cache[cache_key] = []
        return []

    def _calculate_pacman_turns(self, path: list) -> int:
        """
        Dựa vào danh sách bước đi, tính toán chính xác số LƯỢT DI CHUYỂN của Pacman.
        Pacman đi thẳng: 2 ô/lượt. Gặp khúc cua: chỉ đi được 1 ô/lượt.
        """
        if not path:
            return 0
            
        turns = 0
        i = 0
        n = len(path)
        
        while i < n:
            current_direction = path[i]
            segment_len = 0
            # Pacman chỉ có thể đi tối đa 2 ô cùng chiều trong 1 lượt di chuyển
            while i < n and path[i] == current_direction and segment_len < 2:
                segment_len += 1
                i += 1
            turns += 1
            
        return turns

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0

    def _count_walkable_neighbors(self, pos: tuple, map_state: np.ndarray) -> int:
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_valid_position((pos[0] + dr, pos[1] + dc), map_state):
                count += 1
        return count