"""
NHẬT KÝ SỬA CODE SO VỚI BẢN CŨ:
- Pacman: 
  + CŨ: Dùng BFS loang đều để tìm đường, gặp map to là chạy bị lag với dễ bị timeout
  + MỚI: Đổi sang thuật toán A* dùng heapq, hướng thẳng tới Ghost nên chại nhanh hơn
- Ghost:
  + CŨ: Tính khoảng cách Manhattan (đường chim bay) nên hay bị vô tường
        Phạt ngõ cụt -60 nhẹ nên vẫn hay tự chui đầu vào tường
  + MỚI: Tính khoảng cách đi bộ thực tế (True Distance) bằng A*
        Thêm cái cache lưu khoảng cách để k bị tính đi tính lại
        Thêm hàm lookahead check xem lượt tới Pacman có chạy thẳng 2 ô tới mình k
        Tăng phạt ngõ cụt lên -8000 cho nó sợ k dám chui vào
"""

import sys
from pathlib import Path
import heapq  # dùng cái này làm hàng đợi ưu tiên cho A*
import random
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Con Pacman đi săn Ghost (đội Seeker)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gr03_Seeker_AStar_24127053"
        # lấy tốc độ tối đa được cấu hình, nếu lỗi thù mặc định cho bằng 2
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2))) 
        # biến tạm để nhớ vị trí cuối cùng của ghost khi bị mất dấu trong sương mù
        self.last_known_enemy_pos = None 
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        """
        Hàm xử lý chính mỗi lượt của pacman, trả về hướng đi với số bước
        """
        # thấy ghost thỳ phải note lại toạ độ liền
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        # ưu tiên đuổi theo vị trí mới nhất, ko thấy thì chạy tới chỗ cũ vừa lưu
        target = enemy_position or self.last_known_enemy_pos
        
        # nếu mất dấu hoàn toàn cho chạy bừa vô mấy ô trống để dò đường :v
        if target is None:
            # lọc mấy hướng đi hợp lệ ko bị đụng tường
            valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                           if self._is_valid(self._apply_move(my_position, m), map_state)]
            # có đường thì bốc đại 1 hướng, ko có thù đứng im
            return (random.choice(valid_moves), 1) if valid_moves else (Move.STAY, 1)

        # gọi thuật toán A* để tìm đường ngắn nhất tới con Ghost
        path = self._astar_find_path(my_position, target, map_state)
        
        # nếu A* tìm đc đường đi
        if path:
            first_move = path[0] # lấy nước đi đầu tiên để chạy
            consecutive_steps = 1 # mặc định đi 1 ô trc đã
            
            # khúc này tận dụng speed multiplier để check xem đường phía trc có thẳng ko để phóng 2 lượt
            for next_move in path[1:self.pacman_speed]:
                if next_move == first_move:
                    consecutive_steps += 1 # đường thẳng băng thì cộng thêm bước chạy cho nhanh
                else:
                    break # gặp ngã rẽ hay cua quẹo thì dừng lại
                    
            return (first_move, consecutive_steps)
            
        # bí đường quá k biết đi đâu thì đứng im
        return (Move.STAY, 1)

    def _astar_find_path(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """ Thuật toán A* tìm đường ngắn nhất, tránh ngõ cụt ôn hơn cái BFS """
        if start == goal:
            return [] # đứng trùng chỗ r thì khỏi đi mất công
            
        # open_set dùng heapq lưu: (điểm_f, chi_phí_g, tọa_độ, danh_sách_nước_đi)
        # hàm heuristic tính bằng khoảng cách Manhattan
        h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        open_set = [(h_start, 0, start, [])]
        
        # dùng dict lưu mấy ô đã đi qua kèm chi phí g để tránh bị chạy vòng lặp vô tận
        visited = {start: 0}
        
        h_max, w_max = map_state.shape # lấy kích thước map
        
        while open_set:
            # bốc cái ô có điểm f nhỏ nhất ra duyệt trc (ưu tiên ô hướng về phía mục tiêu)
            _, g, current, path = heapq.heappop(open_set)
            
            # chạm đc vào người Ghost r thì trả về đường đi luôn
            if current == goal:
                return path 
                
            # nếu đường này tốn sức hơn đường khác từng đi qua ô này thì bỏ qua
            if g > visited.get(current, float('inf')):
                continue
                
            # quét 4 hướng lên xuống trái phải xem đi hướng nào ổn nhất
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = current[0] + dr, current[1] + dc
                
                # check xem ô tiếp theo có nằm trong map với là ô trống (0) ko, tường (1) thì bỏ
                if 0 <= nr < h_max and 0 <= nc < w_max and map_state[nr, nc] == 0:
                    next_pos = (nr, nc)
                    new_g = g + 1 # đi thêm 1 bước thì tốn thêm 1 công
                    
                    # nếu tìm ra đường tối ưu hơn cho ô next_pos này
                    if new_g < visited.get(next_pos, float('inf')):
                        visited[next_pos] = new_g # cập nhật kỷ lục mới cho ô này
                        h_val = abs(nr - goal[0]) + abs(nc - goal[1]) # tính heuristic mới
                        # nhét vào heapq tí so sánh tiếp
                        heapq.heappush(open_set, (new_g + h_val, new_g, next_pos, path + [move]))
                        
        return [] # đi hết map k thấy đường thì trả về rỗng

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """ Hàm tính nhanh tọa độ mới khi đi theo một hướng nào đó """
        return (pos[0] + move.value[0], pos[1] + move.value[1])

    def _is_valid(self, pos: tuple, map_state: np.ndarray) -> bool:
        """ hàm check xem ô đó có hợp pháp để bước chân vào k """
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0


class GhostAgent(BaseGhostAgent):
    """
    Con Ghost né Pacman (đội Hider)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Gr03_Ghost_Evasive_24127053"
        self.last_known_pacman_pos = None
        # BỘ NHỚ ĐỆM CACHE: lưu khoảng cách mấy cặp ô, đỡ phải tính lại A* liên tục
        self.distance_cache = {} 
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        """
        Hàm xử lý chính mỗi lượt của Ghost để tìm hướng chạy trốn an toàn nhất
        """
        # thấy pacman thì ghi vô sổ tay toạ độ liền
        if enemy_position is not None:
            self.last_known_pacman_pos = enemy_position
            
        pacman_pos = enemy_position or self.last_known_pacman_pos
        
        # mất dấu pacman hoàn toàn thì chạy đại vô ô trống nào an toàn cho đỡ đứng im chịu trận
        if pacman_pos is None:
            return self._get_random_safe_move(my_position, map_state)
            
        r_g, c_g = my_position
        # mảng 5 hành động có thể làm
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        
        best_moves = []
        best_score = -float('inf') # đặt điểm tốt nhất ban đầu âm vô cùng để tí so sánh
        
        # vòng lặp chấm điểm từng nước đi để chọn ra cái tốt nhất
        for move in possible_moves:
            next_pos = (r_g + move.value[0], c_g + move.value[1])
            
            # điểm đến k hợp lệ (đâm tường hoặc ngoài rìa) thì dẹp luôn k xét
            if not self._is_valid_position(next_pos, map_state):
                continue
                
            # ĐOẠN NHÌN TRƯỚC: check trc tương lai xem bước vô ô này có bị pacman phóng thẳng đớp luôn ko
            if self._can_pacman_snipe_me(next_pos, pacman_pos, map_state):
                score = -999999 # ô tử thần, phạt điểm cực nặng để k bao giờ đi vào!!
            else:
                # CẢI TIẾN: tính khoảng cách đi bộ thực tế né tường bằng A*
                true_dist = self._get_true_distance(next_pos, pacman_pos, map_state)
                score = true_dist * 150.0 # càng xa pacman điểm càng to, nhân thêm 150 cho máu
                
                # đếm số đường lui xung quanh xem có bị dồn vô ngõ cụt ko
                escape_routes = self._count_walkable_neighbors(next_pos, map_state)
                if move != Move.STAY:
                    if escape_routes <= 1:
                        score -= 8000  # ngõ cụt là trừ 8000 điểm cho nó đỡ chui vào
                    elif escape_routes == 2:
                        score -= 400   # đường hành lang đơn hẹp, hạn chế đi kẻo bị ép góc
                else:
                    # nếu chọn đứng im (STAY) mà pacman đang ở quá gần (khoảng cách thực <= 4)
                    if true_dist <= 4:
                        score -= 2000  # địch đến gần thì trừ điểm nặng cho biết đường chạy
            
            # thuật toán tìm max điểm cơ bản tự viết
            if score > best_score:
                best_score = score
                best_moves = [move] # tìm đc đuòng tốt hơn thì reset lại danh sách
            elif score == best_score:
                best_moves.append(move) # bằng nhau thì nhét chung vào tí chọn ngẫu nhiên
                
        # bốc ngẫu nhiên 1 nước trong đám ngon nhất để tạo tính bất ngờ, tránh bị bắt bài
        return random.choice(best_moves) if best_moves else Move.STAY

    def _get_true_distance(self, start: tuple, goal: tuple, map_state: np.ndarray) -> int:
        """ Đo khoảng cách thực tế đi trong mê cung chứ k chơi đường chim bay nha """
        if start == goal:
            return 0
            
        # chuẩn hóa key (ô nhỏ đứng trước) để đi từ A->B hay B->A cũng tính là 1 cặp, đỡ tốn cache
        pair = (start, goal) if start < goal else (goal, start)
        if pair in self.distance_cache:
            return self.distance_cache[pair] # có sẵn kết quả r thì bốc ra xài luôn
            
        # nếu cache chưa có, chạy A* thu gọn để tính khoảng cách thực tế giữa 2 ô
        h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        open_set = [(h_start, 0, start)]
        visited = {start: 0}
        h_max, w_max = map_state.shape
        result_dist = 999 # mặc định nếu bị cô lập hoàn toàn k có đường thông nhau
        
        while open_set:
            _, g, current = heapq.heappop(open_set)
            
            if current == goal:
                result_dist = g # tìm thấy đường r, lấy chi phí bước đi g làm khoảng cách luôn
                break
                
            if g > visited.get(current, float('inf')):
                continue
                
            # quét nhanh 4 ô xung quanh lân cận
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < h_max and 0 <= nc < w_max and map_state[nr, nc] == 0:
                    next_pos = (nr, nc)
                    new_g = g + 1
                    if new_g < visited.get(next_pos, float('inf')):
                        visited[next_pos] = new_g
                        h_val = abs(nr - goal[0]) + abs(nc - goal[1])
                        heapq.heappush(open_set, (new_g + h_val, new_g, next_pos))
                        
        self.distance_cache[pair] = result_dist # nạp vào bộ nhớ cache để lượt sau khỏi mất công tính lại
        return result_dist

    def _can_pacman_snipe_me(self, ghost_pos: tuple, pacman_pos: tuple, map_state: np.ndarray) -> bool:
        """ Hàm nhìn trước tương lai: giả lập xem lượt tới Pacman có kích tốc phóng thẳng tóm mình ko """
        h_max, w_max = map_state.shape
        
        # thử duyệt qua mấy kịch bản hành động của pacman địch
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            curr_p = pacman_pos
            
            # kịch bản pacman đứng im: check xem khoảng cách kề bên có chạm nhau chưa
            if move == Move.STAY:
                if abs(curr_p[0] - ghost_pos[0]) + abs(curr_p[1] - ghost_pos[1]) < 2:
                    return True # sát nhau quá, nguy hiểm vcl!
            else:
                # kịch bản pacman phóng đường thẳng: giả lập pacman chại thẳng tối đa 2 ô liên tiếp
                for _ in range(2): 
                    nr, nc = curr_p[0] + move.value[0], curr_p[1] + move.value[1]
                    # nếu đường thẳng đó trống trải đi đc
                    if 0 <= nr < h_max and 0 <= nc < w_max and map_state[nr, nc] == 0:
                        curr_p = (nr, nc) # pacman tiến lên 1 ô giả lập
                        # nếu ô giả lập này nằm đè hoặc sát sạt ô Ghost định đi tới
                        if abs(curr_p[0] - ghost_pos[0]) + abs(curr_p[1] - ghost_pos[1]) < 2:
                            return True # kích hoạt cảnh báo nguy hiểm
                    else:
                        break # đụng tường chặn đứng hướng chạy thẳng thì dừng giả lập hướng này
        return False # ô này check xong thấy an toàn

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """ hàm check tọa độ xem có nằm trong map với ko phải tường ko """
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0

    def _count_walkable_neighbors(self, pos: tuple, map_state: np.ndarray) -> int:
        """ đếm xem xung quanh ô này có bao nhiêu lối đi trống để phát hiện ngõ cụt hiểm trở """
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self._is_valid_position((pos[0] + dr, pos[1] + dc), map_state):
                count += 1
        return count

    def _get_random_safe_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """ hàm cứu cánh lúc mất dấu địch: lựa đại 1 ô trống xung quanh để chại """
        safe_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
                      if self._is_valid_position((my_position[0] + m.value[0], my_position[1] + m.value[1]), map_state)]
        return random.choice(safe_moves) if safe_moves else Move.STAY