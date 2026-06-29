"""
Mã nguồn Agent chính thức của Nhóm 03 - Phân hệ GhostAgent (Hide Agent)
Đáp ứng các tiêu chí: Tối đa khoảng cách, Né trục thẳng hàng/cột, Tránh ngõ cụt.
"""

import sys
from pathlib import Path

# Cấu hình tự động trỏ đường dẫn về thư mục src để import framework
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random


class PacmanAgent(BasePacmanAgent):
    """
    Placeholder cho Pacman Agent (Lấy thuật toán Greedy mẫu của trường).
    Nhóm có thể tối ưu thêm class này sau ở phân hệ Seeker.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gr03 Seeker Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        target = enemy_position or self.last_known_enemy_pos
        if target is None:
            return (Move.RIGHT, 1)
        row_diff, col_diff = target[0] - my_position[0], target[1] - my_position[1]
        move = Move.STAY
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Kiểm tra tính hợp lệ tối thiểu
        dr, dc = move.value
        np_pos = (my_position[0] + dr, my_position[1] + dc)
        if 0 <= np_pos[0] < map_state.shape[0] and 0 <= np_pos[1] < map_state.shape[1] and map_state[np_pos] == 0:
            return (move, 1)
        return (Move.STAY, 1)


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent (Hide Agent) được tối ưu hóa bởi Nhóm 03.
    Chiến thuật: Tham lam thông minh kết hợp ma trận phạt rủi ro.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gr03 Smart Evasive Ghost"
        # Bộ nhớ lưu vị trí Pacman phòng trường hợp bật chế độ sương mù (Fog of war)
        self.last_known_pacman_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple, 
             step_number: int) -> Move:
        """
        Xử lý từng bước di chuyển của Ghost.
        """
        # 1. Cập nhật thông tin vị trí của Pacman vào bộ nhớ ngắn hạn
        if enemy_position is not None:
            self.last_known_pacman_pos = enemy_position
            
        pacman_pos = enemy_position or self.last_known_pacman_pos
        
        # Nếu hoàn toàn không biết Pacman ở đâu (đầu game), di chuyển ngẫu nhiên an toàn
        if pacman_pos is None:
            return self._get_random_safe_move(my_position, map_state)
            
        r_p, c_p = pacman_pos      # Tọa độ Pacman (Hàng, Cột)
        r_g, c_g = my_position     # Tọa độ Ghost hiện tại (Hàng, Cột)
        
        # Danh sách 5 nước đi có thể thực hiện
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        
        best_moves = []
        best_score = -float('inf')
        
        # 2. Duyệt qua và chấm điểm toán học cho từng nước đi
        for move in possible_moves:
            dr, dc = move.value
            next_pos = (r_g + dr, c_g + dc)
            
            # Kiểm tra xem nước đi này có hợp lệ không (Không đâm vào tường, không ra ngoài map)
            if not self._is_valid_position(next_pos, map_state):
                continue
                
            # Khởi tạo điểm số cơ bản cho vị trí mới
            score = 0
            
            # --- YÊU CẦU 1: TỐI ĐA HÓA KHOẢNG CÁCH MANHATTAN VỚI PACMAN ---
            manhattan_dist = abs(next_pos[0] - r_p) + abs(next_pos[1] - c_p)
            score += manhattan_dist * 2.0  # Nhân hệ số trọng số cho khoảng cách
            
            # --- YÊU CẦU 2: TRÁNH CHUNG HÀNG HOẶC CHUNG CỘT VỚI PACMAN ---
            # Pacman đi thẳng được 2 ô, đứng chung trục dọc/ngang cực kỳ nguy hiểm nếu khoảng cách gần
            if next_pos[0] == r_p or next_pos[1] == c_p:
                if manhattan_dist <= 4:
                    score -= 40  # Phạt cực nặng nếu đang ở gần mà lại lao vào trục thẳng
                else:
                    score -= 15  # Phạt vừa phải nếu đang ở xa
                    
            # --- YÊU CẦU 3: TUYỆT ĐỐI NÉ NGÕ CỤT (DEAD-ENDS) ---
            # Chỉ kiểm tra nếu Ghost thực sự di chuyển đến vị trí mới
            if move != Move.STAY:
                escape_routes = self._count_walkable_neighbors(next_pos, map_state)
                
                # Nếu ô sắp tới chỉ có 1 đường ra (chính là đường mình vừa đi vào) -> Ngõ cụt!
                if escape_routes <= 1:
                    score -= 60  # Trừ điểm rất nặng để chặn không cho Ghost tự chui đầu vào góc kẹt
                elif escape_routes == 2:
                    score -= 5   # Phạt nhẹ vì đây là đường hành lang đơn, dễ bị Pacman dồn ép
            
            # --- ĐẶC BIỆT: PHẠT NƯỚC ĐI ĐỨNG YÊN (STAY) NẾU PACMAN ĐANG ĐẾN GẦN ---
            if move == Move.STAY and manhattan_dist <= 5:
                score -= 25
                
            # 3. Tìm nước đi có điểm số cao nhất
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move) # Lưu lại nếu có nhiều nước đi tốt bằng nhau để tránh bị bắt bài
                
        # Nếu tìm được nước đi tốt, chọn ngẫu nhiên trong nhóm tốt nhất để tăng tính bất ngờ
        if best_moves:
            return random.choice(best_moves)
            
        return Move.STAY

    # Các hàm bổ trợ (Helper Methods)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Kiểm tra tọa độ có nằm trong bản đồ và không phải là tường (1)."""
        r, c = pos
        h, w = map_state.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return map_state[r, c] == 0  # 0 là ô trống đi được

    def _count_walkable_neighbors(self, pos: tuple, map_state: np.ndarray) -> int:
        """Đếm số ô trống bao quanh một tọa độ để phát hiện ngõ cụt."""
        r, c = pos
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)
            if self._is_valid_position(neighbor, map_state):
                count += 1
        return count

    def _get_random_safe_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """Di chuyển ngẫu nhiên an toàn khi mất dấu Pacman."""
        safe_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            dr, dc = move.value
            next_pos = (my_position[0] + dr, my_position[1] + dc)
            if self._is_valid_position(next_pos, map_state):
                safe_moves.append(move)
        return random.choice(safe_moves) if safe_moves else Move.STAY