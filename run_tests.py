import io
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

# Thêm thư mục src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Chặn mọi output khi import arena (nếu có)
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from arena import Arena

SEEKER_ID = "24127555"
HIDER_ID = "example_student"
NUM_RUNS = 10

print(f"🚀 Bắt đầu thực nghiệm {NUM_RUNS} lần giữa Pacman ({SEEKER_ID}) và Ghost ({HIDER_ID})...\n")

print("-" * 55)
print(f"{'Lần Test':<12}{'Kết quả':<20}{'Số Steps'}")
print("-" * 55)

pacman_win = 0
ghost_win = 0

for i in range(1, NUM_RUNS + 1):

    # Chặn toàn bộ output trong lúc Arena chạy
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):

        arena = Arena(
            pacman_id=SEEKER_ID,
            ghost_id=HIDER_ID,
            submissions_dir="submissions",
            visualize=False,
            delay=0,
            max_steps=200,
            pacman_speed=2,
            capture_distance_threshold=2,
        )

        arena.load_agents()
        result, stats = arena.run_game()

    steps = stats["total_steps"]

    if result == "pacman_wins":
        winner = "Pacman Win 👑"
        pacman_win += 1
    else:
        winner = "Ghost Win 👻"
        ghost_win += 1

    print(f"Lần {i:<8}{winner:<20}{steps}")

print("-" * 55)
print(f"Pacman thắng : {pacman_win}")
print(f"Ghost thắng  : {ghost_win}")
print(f"Tổng số trận : {NUM_RUNS}")
print("-" * 55)