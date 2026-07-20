# NKT - HideSeek

## 1. Giới thiệu
* Tên dự án: CSC14003 - Hide and Seek Arena 
* Nhóm 2, thành viên nhóm: 
  * Chung Nguyễn Hữu Tiến - 24127555 
  * Nguyễn Phạm Minh Ngọc - 24127464
  * Trần Hoàng Minh Khang - 24127053
* Tổng quan Dự án: Xây dựng hai tác tử AI trong môi trường bản đồ $21 \times 21$: Pacman (Seek Agent - có nhiệm vụ truy tìm và bắt mục tiêu nhanh nhất) và Ghost (Hide Agent - có nhiệm vụ lẩn trốn và sinh tồn lâu nhất).
---

## 2. Nhật ký phát triển & Các lần Submit
### 2.1 Initial Submit


## a. `PacmanAgent`

Thuật toán cốt lõi được sử dụng cho Pacman là **Tìm kiếm theo chiều rộng (Breadth-First Search - BFS)** kết hợp với cơ chế **ghi nhớ trạng thái** để đối phó với sương mù và tối ưu hóa tốc độ di chuyển.

**Các bước hoạt động chính:**

*   **Xử lý tầm nhìn hạn chế (Fog of War):** Thuật toán liên tục cập nhật và ghi nhớ vị trí của Ghost (`last_known_enemy_pos`). Nếu Ghost biến mất vào vùng sương mù, Pacman sẽ đi đến vị trí cuối cùng mà nó nhìn thấy mục tiêu.
*   **Tìm đường ngắn nhất (BFS):** Hàm `_bfs_find_path` sử dụng hàng đợi (queue) để loang ra các ô lân cận. BFS đảm bảo rằng con đường tìm được từ Pacman đến Ghost luôn là con đường ngắn nhất và không xuyên qua tường.
*   **Chống lặp bước (Anti-Oscillation):** Thuật toán lưu lại vị trí của bước đi liền trước (`last_position`). Nếu đường đi BFS chỉ đạo Pacman quay lại chính ô nó vừa đứng (có thể do cập nhật lại vị trí Ghost), nó sẽ gọi hàm `_find_non_repeating_move` để tìm một hướng đi thay thế, giúp tác tử không bị mắc kẹt trong vòng lặp tiến-lùi liên tục.
*   **Tối ưu hóa tốc độ (Speed Gomming):** Tận dụng lợi thế về tốc độ (`pacman_speed >= 2`), nếu hai bước đầu tiên trong đường đi BFS đều chỉ về cùng một hướng, thuật toán sẽ gộp lại và trả về lệnh đi 2 bước `(chosen_move, 2)`, giúp Pacman áp sát nhanh hơn. 

---

## b. `GhostAgent`

Khác với Pacman sử dụng thuật toán dò đường tuyệt đối, Ghost sử dụng **Tìm kiếm Tham lam / Heuristic (Greedy/Heuristic Search)** dựa trên một hàm đánh giá điểm số (scoring function) để quyết định bước đi tốt nhất nhằm sống sót lâu nhất.

**Các bước hoạt động chính:**

*   **Trường hợp mất dấu Pacman:**
    *   Ghost không đứng im mà di chuyển ngẫu nhiên để tránh bị phục kích. 
    *   Thuật toán quét các hướng đi hợp lệ và sử dụng hàm `_is_dead_end` (kiểm tra ngõ cụt - nơi chỉ có 1 lối ra/vào). Nó ưu tiên lọc bỏ các ngõ cụt khỏi danh sách đi ngẫu nhiên, chỉ đi vào ngõ cụt nếu không còn đường nào khác.
*   **Trường hợp nhìn thấy (hoặc nhớ) vị trí Pacman:**
    Ghost sẽ đánh giá 4 hướng di chuyển có thể (Lên, Xuống, Trái, Phải) dựa trên **Khoảng cách Manhattan** và chấm điểm từng hướng đi. Điểm càng cao, nước đi càng an toàn.
    *   **Tiêu chí 1 - Khoảng cách (Tối đa +10 điểm):** Tính khoảng cách Manhattan từ vị trí dự kiến đến Pacman. Nếu khoảng cách tăng lên (đi xa ra), cộng 10 điểm. Nếu khoảng cách giữ nguyên (đi ngang), cộng 2 điểm.
    *   **Tiêu chí 2 - Tránh bị truy đuổi thẳng (Tối đa +15 điểm):** Ưu tiên các bước đi khiến Ghost không nằm cùng hàng (row) hoặc cùng cột (col) với Pacman. Điều này khiến Pacman phải rẽ ngoặt liên tục để đuổi theo, đồng thời giúp Ghost dễ lẩn khuất vào sương mù hơn.
    *   **Tiêu chí 3 - Tuyệt đối tránh ngõ cụt (Tối đa +30 điểm):** Đây là tiêu chí quan trọng nhất. Hàm sẽ cộng 30 điểm cho bất kỳ nước đi nào không dẫn vào ngõ cụt.
*   **Ra quyết định:** Sau khi chấm điểm, Ghost sẽ tìm ra mức điểm cao nhất. Nếu có nhiều hướng cùng đạt mức điểm tối đa, nó sẽ chọn ngẫu nhiên một trong số đó để tạo tính khó đoán (unpredictability), khiến Pacman khó bắt bài hơn.
### 2.2 Final Submit


### a. `PacmanAgent`
Mục tiêu của Pacman là tìm và bắt được Ghost nhanh nhất có thể.

* **Thuật toán tìm đường chính**: **Tìm kiếm theo chiều rộng (BFS - Breadth-First Search)**
    * Hàm `_bfs_find_path` sử dụng cấu trúc dữ liệu hàng đợi (`deque`) để duyệt qua các ô trống kề nhau trên bản đồ. Thuật toán này đảm bảo tìm ra con đường ngắn nhất (ít bước di chuyển nhất) từ vị trí hiện tại của Pacman đến vị trí của Ghost.
* **Chiến thuật xử lý sương mù (Fog of War)**:
    * Agent có bộ nhớ lưu lại vị trí cuối cùng nhìn thấy Ghost (`self.last_known_enemy_pos`). Nếu Ghost biến mất khỏi tầm nhìn, Pacman vẫn sẽ đi đến điểm cuối cùng đó.
    * Nếu chưa từng nhìn thấy Ghost, Pacman sẽ chọn di chuyển hợp lệ ngẫu nhiên để dò đường.
* **Chống kẹt (Anti-loop / Oscillation)**:
    * Lưu lại vị trí ở bước đi trước đó (`self.last_position`).
    * Nếu thuật toán tính toán ra bước đi tiếp theo lại quay về đúng vị trí cũ (dấu hiệu của việc bị kẹt hoặc lặp vòng), nó sẽ kích hoạt hàm `_find_non_repeating_move` để tìm một hướng đi hợp lệ khác nhằm bẻ gãy vòng lặp.
* **Tận dụng tốc độ**:
    * Nếu Pacman có tốc độ $\ge 2$, agent sẽ cố gắng lấy ra 2 bước đi liên tiếp từ đường dẫn BFS. Nếu cả 2 bước này đều đi trên cùng một đường thẳng (ví dụ: cùng tiến lên), nó sẽ trả về nước đi 2 bước để tăng tốc độ rút ngắn khoảng cách.

---

### b. `GhostAgent` 
Mục tiêu của Ghost là sinh tồn càng lâu càng tốt. Thay vì chỉ chạy trốn một cách mù quáng, Ghost sử dụng một **Hệ thống đánh giá Heuristic (Heuristic Evaluation)** phức tạp kết hợp với thuật toán **A* (A-Star)**.

Thuật toán của Ghost được chia làm hai trường hợp rõ rệt:

#### Trường hợp 1: Không bị rượt đuổi (Không thấy Pacman)
Thay vì đứng im, Ghost chủ động chiếm lợi thế địa hình bằng chiến thuật **Kiểm soát trung tâm (Center Control)**.
* Đánh giá tất cả các hướng đi có thể.
* Sử dụng hàm `_get_corridor_depth` để loại bỏ các hướng đi dẫn vào ngõ cụt.
* Trong số những hướng đi an toàn, Ghost sẽ chọn hướng đi có **khoảng cách Manhattan gần với tâm bản đồ nhất**. Việc đứng ở giữa bản đồ giúp Ghost có nhiều không gian để né tránh khi Pacman bất ngờ xuất hiện.

#### Trường hợp 2: Bị rượt đuổi (Thấy Pacman)
Ghost dùng hệ thống "chấm điểm" (Scoring System) cho từng hướng đi xung quanh (LÊN, XUỐNG, TRÁI, PHẢI). Nước đi nào có tổng điểm cao nhất sẽ được chọn. Điểm được cộng/trừ dựa trên 5 tiêu chí:

1. **Khoảng cách thực tế tới Pacman (Quan trọng nhất)**:
    * Sử dụng thuật toán **A*** (`_calculate_maze_distance`) để tính độ dài đoạn đường thực tế trong mê cung giữa vị trí dự kiến và Pacman (không tính xuyên tường).
    * *Điểm*: **+100** nếu bước đi giúp tăng khoảng cách, **+30** nếu giữ nguyên khoảng cách, và phạt **-100** nếu bước đi đó làm khoảng cách bị thu hẹp lại.
2. **Tránh trục nhìn (Line of sight)**:
    * *Điểm*: **+15** nếu bước đi đưa Ghost ra khỏi cùng trục ngang hoặc dọc với vị trí của Pacman, giảm thiểu nguy cơ bị ép góc.
3. **Độ linh hoạt cục bộ**:
    * Kiểm tra xem từ vị trí dự kiến đi tới có bao nhiêu ngã rẽ có thể đi được (`_count_valid_neighbors`).
    * *Điểm*: **+5** cho mỗi ngã rẽ hợp lệ. Càng nhiều đường thoát, vị trí đó càng an toàn.
4. **Đo chiều sâu ngõ cụt (Chống bẫy)**:
    * Sử dụng thuật toán dò hành lang (`_get_corridor_depth`). Thuật toán này mô phỏng việc đi dọc theo một con đường hẹp xem có bị tắc ở cuối không.
    * *Điểm*: **+50** nếu đường đi thông thoáng (trả về vô cực). Phạt cực nặng **-1000** nếu phát hiện đó là ngõ cụt không lối thoát, tuyệt đối tránh tự đưa mình vào ngõ cụt.
5. **Tiêu chí hướng tâm**:
    * *Điểm*: Phạt nhẹ **-2 $\times$ (Khoảng cách tới tâm)** để Ghost không có xu hướng dạt ra rìa bản đồ trừ khi bắt buộc phải làm thế để né ngõ cụt hoặc né Pacman. Trọng số này rất nhỏ để không phá hỏng ưu tiên chạy trốn.

---
## 3. Kết quả Thực nghiệm & Giải quyết Khó khăn
### Đánh giá chung
* Các tác tử đã hoàn thành vai trò. Thuật toán cân bằng tốt giữa thời gian chạy (tối đa 1 giây) và tài nguyên bộ nhớ (tối đa 128 MB), tối ưu hóa được độ chênh lệch số bước (tie-breaking) để đem lại lợi thế trên bảng xếp hạng.

### Khó khăn kỹ thuật đã xử lý
* Cắt tỉa độ sâu và tối ưu vòng lặp Heuristic của Ghost để tránh lỗi vượt quá thời gian ra quyết định cho phép.
* Quản lý tốt cấu trúc dữ liệu lưu trữ để tránh tràn giới hạn bộ nhớ khi loang các vùng không gian lớn trong môi trường.
* Tinh chỉnh liên tục các trọng số (weights) cho 5 tiêu chí của Ghost để tác tử không bị vướng bẫy chiến thuật của chính mình.
* Ghost ban đầu thiếu tính tối ưu và khá yếu trong việc sinh tồn.